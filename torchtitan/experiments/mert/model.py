# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MERT (Music undERstanding model with large-scale self-supervised Training).

Architecture follows the reference training code in
``ReferenceRepos/MERT/mert_fairseq/models/mert/mert_model.py`` (itself a
HuBERT/wav2vec2 derivative): a strided CNN feature extractor over raw audio,
projected into a bidirectional Transformer encoder, trained with a masked
multi-codebook classification objective plus an auxiliary Constant-Q
reconstruction loss. See the experiment README for known, documented
deviations from the reference code.

Shape suffix legend (module-local; see
https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
    B = batch, T = raw audio samples, F = downsampled encoder frames,
    Dc = conv feature-extractor output channels, D = encoder embedding dim,
    H = feed-forward hidden dim, M = number of codebooks (target label sets),
    K = classes per codebook (codebook size), Q = projection dim used for the
    cosine-similarity classification head (``final_dim``), V = number of CQT
    bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, GELU, GroupNorm, LayerNorm
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict, ModuleList

from .cqt import ConstantQTransform
from .masking import compute_mask_indices


def downsample_factor(conv_layers: tuple[tuple[int, int, int], ...]) -> int:
    """Nominal total temporal downsampling (product of strides).

    This is only a nominal ratio, not the exact input-length -> output-length
    map: since every conv layer is unpadded ("valid") convolution and every
    layer's kernel size exceeds its stride, each layer truncates a few extra
    trailing samples beyond what the stride alone accounts for. Use it only
    where an approximate frames-per-sample ratio suffices (e.g. picking a
    label-array offset for an audio crop); use ``conv_output_length`` /
    ``required_input_length`` wherever the exact frame count matters (label
    array sizing, fixed-shape batch construction).
    """
    factor = 1
    for _, _, stride in conv_layers:
        factor *= stride
    return factor


def conv_output_length(
    conv_layers: tuple[tuple[int, int, int], ...], input_length: int
) -> int:
    """Exact number of output frames for ``input_length`` input samples,
    applying each (unpadded, dilation=1) conv layer's length formula in turn:
    ``L_out = floor((L_in - kernel_size) / stride) + 1``.
    """
    length = input_length
    for _, kernel_size, stride in conv_layers:
        length = (length - kernel_size) // stride + 1
    return length


def required_input_length(
    conv_layers: tuple[tuple[int, int, int], ...], num_frames: int
) -> int:
    """Inverse of ``conv_output_length``: the smallest ``input_length`` that
    makes the conv stack produce exactly ``num_frames`` output frames.

    Solves each layer's length formula for its minimal ``L_in`` given a
    target ``L_out``, walking the conv stack in reverse; substituting that
    minimal length forward through ``conv_output_length`` reproduces exactly
    ``num_frames`` at every intermediate layer (not just the last one), since
    ``(L_out - 1) * stride`` is by construction a multiple of ``stride``.
    """
    length = num_frames
    for _, kernel_size, stride in reversed(conv_layers):
        length = (length - 1) * stride + kernel_size
    return length


def _default_linear_bias_init(fan_in: int):
    """``nn.Linear.reset_parameters()``'s default bias init, as a standalone
    callable, for layers whose weight gets a non-default ``param_init`` while
    their bias should keep PyTorch's ordinary default."""
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    return lambda b: nn.init.uniform_(b, -bound, bound)


class Dropout(nn.Dropout, Module):
    """Configurable ``nn.Dropout`` (not provided by ``models.common.nn_modules``)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        p: float = 0.0

    def __init__(self, config: Config):
        super().__init__(p=config.p)


class MERTFeatureExtractor(Module):
    """Strided CNN over raw audio, matching wav2vec2/HuBERT's default-mode
    ``ConvFeatureExtractionModel``: a GroupNorm + GELU after the first conv
    layer, then GELU-only conv layers.

    Only ``extractor_mode="default"`` is implemented: both released MERT
    configs (95M, 330M) use it, and the reference's alternative
    ``layer_norm`` mode is unused by any shipped config.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        conv_layers: tuple[tuple[int, int, int], ...]
        """``((out_channels, kernel_size, stride), ...)`` per conv layer."""
        conv_bias: bool = False

    def __init__(self, config: Config):
        super().__init__()
        # Weights use each layer's default ``reset_parameters()`` (plain
        # PyTorch conv/norm init), matching the reference, which never
        # overrides ``ConvFeatureExtractionModel``'s layer init.
        blocks = []
        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(config.conv_layers):
            conv = Conv1d.Config(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=config.conv_bias,
            ).build()
            layer: list[Module] = [conv]
            if i == 0:
                layer.append(
                    GroupNorm.Config(
                        num_groups=out_channels,
                        num_channels=out_channels,
                    ).build()
                )
            layer.append(GELU.Config().build())
            blocks.append(ModuleList(layer))
            in_channels = out_channels
        self.conv_layers = ModuleList(blocks)
        self.out_channels = in_channels

    def forward(self, waveform_BT: torch.Tensor) -> torch.Tensor:
        x_BDcT = waveform_BT.unsqueeze(1)
        for block in self.conv_layers:
            for layer in block:
                x_BDcT = layer(x_BDcT)
        return x_BDcT


class MERTPositionalConvEmbedding(Module):
    """Grouped-conv relative positional embedding, added to the encoder input.

    The reference applies weight normalization to this conv; that is omitted
    here (see README) since it complicates meta-device init / FSDP2 DTensor
    parameters and is not central to the pretraining objective.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        kernel_size: int = 128
        groups: int = 16

    def __init__(self, config: Config):
        super().__init__()
        padding = config.kernel_size // 2
        self.conv = Conv1d.Config(
            in_channels=config.dim,
            out_channels=config.dim,
            kernel_size=config.kernel_size,
            padding=padding,
            groups=config.groups,
        ).build()
        # Even kernel sizes produce one extra trailing frame; drop it so the
        # output length matches the input (matches wav2vec2's ``SamePad``).
        self.remove_last = config.kernel_size % 2 == 0
        self.activation = GELU.Config().build()

    def forward(self, x_BFD: torch.Tensor) -> torch.Tensor:
        x_BDF = x_BFD.transpose(1, 2)
        x_BDF = self.conv(x_BDF)
        if self.remove_last:
            x_BDF = x_BDF[..., :-1]
        x_BDF = self.activation(x_BDF)
        return x_BDF.transpose(1, 2)


class MERTAttention(Module):
    """Bidirectional (non-causal) multi-head self-attention.

    Unlike causal decoder attention (``torchtitan.models.common.attention``),
    MERT is an encoder: every frame attends to every other frame, and there
    is no rotary positional embedding (position information instead comes
    from ``MERTPositionalConvEmbedding``), matching wav2vec2/HuBERT.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        attention_dropout: float = 0.0

    def __init__(self, config: Config):
        super().__init__()
        assert (
            config.dim % config.n_heads == 0
        ), f"dim ({config.dim}) must be divisible by n_heads ({config.n_heads})"
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.attention_dropout = config.attention_dropout
        # Matches fairseq's ``MultiheadAttention.reset_parameters`` for the
        # (default) same-dim q/k/v case: q/k/v weights use Xavier-uniform
        # with gain 1/sqrt(2) (their outputs are summed inside attention),
        # out_proj uses plain Xavier-uniform with a zeroed bias. q/k/v biases
        # are left at ``nn.Linear``'s default init (fairseq does not touch
        # them either).
        qkv_param_init = {
            "weight": lambda w: nn.init.xavier_uniform_(w, gain=1 / math.sqrt(2)),
            "bias": _default_linear_bias_init(config.dim),
        }
        self.wq = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=True,
            param_init=qkv_param_init,
        ).build()
        self.wk = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=True,
            param_init=qkv_param_init,
        ).build()
        self.wv = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=True,
            param_init=qkv_param_init,
        ).build()
        self.wo = Linear.Config(
            in_features=config.dim,
            out_features=config.dim,
            bias=True,
            param_init={"weight": nn.init.xavier_uniform_, "bias": nn.init.zeros_},
        ).build()

    def forward(
        self,
        x_BFD: torch.Tensor,
        key_padding_mask_BF: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x_BFD.shape
        q_BFD, k_BFD, v_BFD = self.wq(x_BFD), self.wk(x_BFD), self.wv(x_BFD)

        def split_heads(t_BFD: torch.Tensor) -> torch.Tensor:
            return t_BFD.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_BHFK, k_BHFK, v_BHFK = (
            split_heads(q_BFD),
            split_heads(k_BFD),
            split_heads(v_BFD),
        )

        attn_mask = None
        if key_padding_mask_BF is not None:
            # SDPA expects True = attend, so invert the True-at-padding mask.
            attn_mask = (~key_padding_mask_BF)[:, None, None, :]

        out_BHFK = F.scaled_dot_product_attention(
            q_BHFK,
            k_BHFK,
            v_BHFK,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        out_BFD = out_BHFK.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(out_BFD)


class MERTFeedForward(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        activation_dropout: float = 0.0

    def __init__(self, config: Config):
        super().__init__()
        # Default ``nn.Linear`` init, matching the reference (fairseq's
        # ``TransformerSentenceEncoderLayer`` FFN uses plain ``nn.Linear``).
        self.w1 = Linear.Config(
            in_features=config.dim,
            out_features=config.hidden_dim,
            bias=True,
        ).build()
        self.w2 = Linear.Config(
            in_features=config.hidden_dim,
            out_features=config.dim,
            bias=True,
        ).build()
        self.activation = GELU.Config().build()
        self.dropout = Dropout.Config(p=config.activation_dropout).build()

    def forward(self, x_BFD: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x_BFD))))


_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}


class MERTEncoderLayer(Module):
    """One bidirectional Transformer encoder layer.

    Supports both post-LN (``layer_norm_first=False``, used by the 95M
    config) and pre-LN (``layer_norm_first=True``, used by the 330M config),
    matching the reference's ``TransformerSentenceEncoderLayer``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_heads: int
        ffn_dim: int
        dropout: float = 0.0
        attention_dropout: float = 0.0
        activation_dropout: float = 0.0
        layer_norm_first: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm_first = config.layer_norm_first
        self.dropout = Dropout.Config(p=config.dropout).build()
        self.attention = MERTAttention.Config(
            dim=config.dim,
            n_heads=config.n_heads,
            attention_dropout=config.attention_dropout,
        ).build()
        self.feed_forward = MERTFeedForward.Config(
            dim=config.dim,
            hidden_dim=config.ffn_dim,
            activation_dropout=config.activation_dropout,
        ).build()
        self.attention_norm = LayerNorm.Config(
            normalized_shape=config.dim, param_init=_NORM_INIT
        ).build()
        self.ffn_norm = LayerNorm.Config(
            normalized_shape=config.dim, param_init=_NORM_INIT
        ).build()

    def forward(
        self,
        x_BFD: torch.Tensor,
        key_padding_mask_BF: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layer_norm_first:
            attn_out = self.attention(self.attention_norm(x_BFD), key_padding_mask_BF)
            x_BFD = x_BFD + self.dropout(attn_out)
            ffn_out = self.feed_forward(self.ffn_norm(x_BFD))
            x_BFD = x_BFD + self.dropout(ffn_out)
        else:
            attn_out = self.attention(x_BFD, key_padding_mask_BF)
            x_BFD = self.attention_norm(x_BFD + self.dropout(attn_out))
            ffn_out = self.feed_forward(x_BFD)
            x_BFD = self.ffn_norm(x_BFD + self.dropout(ffn_out))
        return x_BFD


class MERTEncoder(Module):
    """Stack of ``MERTEncoderLayer``s with a convolutional positional
    embedding, matching the reference's ``TransformerEncoder``.

    Layers are held in a ``ModuleDict`` (keyed by index, as strings) so the
    shared activation-checkpointing / ``torch.compile`` / FSDP tooling that
    expects a ``model.layers`` submodule (see
    ``torchtitan.distributed.activation_checkpoint`` and
    ``torchtitan.distributed.compile``) works unmodified.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        layers: list[MERTEncoderLayer.Config]
        conv_pos: int = 128
        conv_pos_groups: int = 16
        layer_norm_first: bool = False
        dropout: float = 0.0
        layerdrop: float = 0.0

    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm_first = config.layer_norm_first
        self.layerdrop = config.layerdrop
        self.pos_conv = MERTPositionalConvEmbedding.Config(
            dim=config.dim,
            kernel_size=config.conv_pos,
            groups=config.conv_pos_groups,
        ).build()
        self.layer_norm = LayerNorm.Config(
            normalized_shape=config.dim, param_init=_NORM_INIT
        ).build()
        self.dropout = Dropout.Config(p=config.dropout).build()
        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = layer_config.build()

    def forward(
        self,
        x_BFD: torch.Tensor,
        key_padding_mask_BF: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_BFD = x_BFD + self.pos_conv(x_BFD)
        if not self.layer_norm_first:
            x_BFD = self.layer_norm(x_BFD)
        x_BFD = self.dropout(x_BFD)
        for layer in self.layers.values():
            if (
                self.training
                and self.layerdrop > 0.0
                and torch.rand(()) < self.layerdrop
            ):
                continue
            x_BFD = layer(x_BFD, key_padding_mask_BF)
        if self.layer_norm_first:
            x_BFD = self.layer_norm(x_BFD)
        return x_BFD


class MERTModel(BaseModel):
    """MERT: masked multi-codebook prediction pretraining over raw audio.

    ``forward`` returns a dict of intermediate quantities (dense over all
    frames, not gathered at masked positions) rather than a final loss or
    logits tensor; :class:`torchtitan.experiments.mert.loss.MertLoss` combines
    them with the codebook-index labels. Computing everything densely (rather
    than via a boolean-indexed gather at masked positions, as the reference
    does) keeps every tensor a fixed shape, which this framework's default
    CUDA-graph / ``torch.compile`` paths require.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        # --- feature extractor ---
        conv_layers: tuple[tuple[int, int, int], ...] = (
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        )
        conv_bias: bool = False

        # --- transformer encoder ---
        encoder_embed_dim: int = 768
        encoder_layers: int = 12
        encoder_attention_heads: int = 12
        encoder_ffn_embed_dim: int = 3072
        layer_norm_first: bool = False
        dropout: float = 0.1
        attention_dropout: float = 0.1
        activation_dropout: float = 0.0
        dropout_input: float = 0.0
        dropout_features: float = 0.0
        encoder_layerdrop: float = 0.0
        conv_pos: int = 128
        conv_pos_groups: int = 16

        # --- masked multi-codebook classification head ---
        final_dim: int = 64
        """Projection dim for the cosine-similarity classification head."""
        untie_final_proj: bool = False
        """If True, each codebook gets its own slice of a wider final
        projection (see reference ``untie_final_proj``); if False, all
        codebooks share the same projected representation."""
        num_codebooks: int = 8
        codebook_size: int = 1024
        logit_temp: float = 0.1
        feature_grad_mult: float = 1.0

        # --- masking ---
        mask_prob: float = 0.65
        mask_length: int = 10
        no_mask_overlap: bool = False
        mask_min_space: int = 1

        # --- data / auxiliary CQT reconstruction head ---
        sample_rate: int = 24000
        label_rate: float = 75.0
        audio_cqt_loss_m: bool = False
        audio_cqt_bins: int = 336

        def update_from_config(self, *, config, **kwargs) -> None:
            pass

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """FLOPs per token, where a "token" is one encoder frame.

            ``seq_len`` is therefore the number of encoder frames per sample
            (``training.max_context_length``), not the raw audio sample
            count; see the config registry for the frames <-> raw-samples
            conversion via ``downsample_factor``.
            """
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            head_dim = self.encoder_embed_dim // self.encoder_attention_heads
            attention_op_flops = (
                quadratic_attention_flops_per_token(
                    num_heads=self.encoder_attention_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
                * self.encoder_layers
            )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.feature_extractor = MERTFeatureExtractor.Config(
            conv_layers=config.conv_layers,
            conv_bias=config.conv_bias,
        ).build()
        conv_out_dim = config.conv_layers[-1][0]

        self.feature_layer_norm = LayerNorm.Config(
            normalized_shape=conv_out_dim, param_init=_NORM_INIT
        ).build()
        self.post_extract_proj = (
            Linear.Config(
                in_features=conv_out_dim,
                out_features=config.encoder_embed_dim,
                bias=True,
            ).build()
            if conv_out_dim != config.encoder_embed_dim
            else None
        )
        self.dropout_input = Dropout.Config(p=config.dropout_input).build()
        self.dropout_features = Dropout.Config(p=config.dropout_features).build()

        # Value filled in by ``param_init["mask_embedding_MD"]`` (set on this
        # model's Config by the config registry) during ``init_states()``,
        # not here: this runs under a meta device context during
        # ``model_config.build()``, where an eager in-place random-init call
        # would not produce usable values.
        self.mask_embedding_MD = nn.Parameter(torch.empty(config.encoder_embed_dim))

        layer_configs = [
            MERTEncoderLayer.Config(
                dim=config.encoder_embed_dim,
                n_heads=config.encoder_attention_heads,
                ffn_dim=config.encoder_ffn_embed_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout,
                layer_norm_first=config.layer_norm_first,
            )
            for _ in range(config.encoder_layers)
        ]
        self.encoder = MERTEncoder.Config(
            dim=config.encoder_embed_dim,
            layers=layer_configs,
            conv_pos=config.conv_pos,
            conv_pos_groups=config.conv_pos_groups,
            layer_norm_first=config.layer_norm_first,
            dropout=config.dropout,
            layerdrop=config.encoder_layerdrop,
        ).build()

        final_proj_out = config.final_dim * (
            config.num_codebooks if config.untie_final_proj else 1
        )
        self.final_proj = Linear.Config(
            in_features=config.encoder_embed_dim,
            out_features=final_proj_out,
            bias=True,
        ).build()

        self.codebook_embeddings = ModuleDict(
            {
                str(i): Embedding.Config(
                    num_embeddings=config.codebook_size,
                    embedding_dim=config.final_dim,
                    param_init={"weight": nn.init.uniform_},
                ).build()
                for i in range(config.num_codebooks)
            }
        )

        self.cqt = None
        self.cqt_proj = None
        if config.audio_cqt_loss_m:
            self.cqt = ConstantQTransform.Config(
                sample_rate=config.sample_rate,
                hop_length=int(config.sample_rate // config.label_rate),
                n_bins=config.audio_cqt_bins,
            ).build()
            self.cqt_proj = Linear.Config(
                in_features=config.encoder_embed_dim,
                out_features=config.audio_cqt_bins,
                bias=True,
            ).build()

    def apply_mask(self, x_BFD: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x_BFD.shape
        mask_BF = compute_mask_indices(
            (bsz, seq_len),
            padding_mask=None,
            mask_prob=self.config.mask_prob,
            mask_length=self.config.mask_length,
            min_masks=2,
            no_overlap=self.config.no_mask_overlap,
            min_space=self.config.mask_min_space,
        )
        mask_BF_t = torch.from_numpy(mask_BF).to(x_BFD.device)
        x_BFD = torch.where(mask_BF_t.unsqueeze(-1), self.mask_embedding_MD, x_BFD)
        return x_BFD, mask_BF_t

    def preprocess_inputs(self, input_dict, *, parallel_dims, parallelism):
        raise NotImplementedError(
            "MERTModel uses MERTTrainer's bespoke forward_backward_step "
            "instead of the generic preprocess_inputs contract."
        )

    def forward(self, source_BT: torch.Tensor) -> dict[str, torch.Tensor]:
        features_BDcF = self.feature_extractor(source_BT)
        features_pen = features_BDcF.float().pow(2).mean()

        features_BFDc = features_BDcF.transpose(1, 2)
        features_BFDc = self.feature_layer_norm(features_BFDc)

        x_BFD = features_BFDc
        if self.post_extract_proj is not None:
            x_BFD = self.post_extract_proj(x_BFD)
        x_BFD = self.dropout_input(x_BFD)

        x_BFD, mask_BF = self.apply_mask(x_BFD)

        x_BFD = self.encoder(x_BFD)

        cqt_target_BFcV = None
        if self.cqt is not None:
            with torch.no_grad():
                cqt_target_BFcV = self.cqt(source_BT)
            # The CQT's own framing (padded, ``hop_length``-strided) does not
            # exactly agree with the feature extractor's frame count (see
            # ``conv_output_length``'s docstring); crop every per-frame
            # output to whichever is shorter so ``logits_BFMK``, ``mask_BF``,
            # and the CQT prediction/target all share one frame count.
            num_frames = min(cqt_target_BFcV.shape[1], x_BFD.shape[1])
            x_BFD = x_BFD[:, :num_frames]
            mask_BF = mask_BF[:, :num_frames]
            cqt_target_BFcV = cqt_target_BFcV[:, :num_frames]

        proj_BFQ = self.final_proj(x_BFD)
        num_codebooks = self.config.num_codebooks
        if self.config.untie_final_proj:
            proj_per_codebook = proj_BFQ.chunk(num_codebooks, dim=-1)
        else:
            proj_per_codebook = [proj_BFQ] * num_codebooks

        logits_per_codebook = []
        for i in range(num_codebooks):
            codebook_KQ = self.codebook_embeddings[str(i)].weight
            proj_BFQ_i = F.normalize(proj_per_codebook[i].float(), dim=-1)
            codebook_KQ_n = F.normalize(codebook_KQ.float(), dim=-1)
            logits_BFK = (
                torch.matmul(proj_BFQ_i, codebook_KQ_n.t()) / self.config.logit_temp
            )
            logits_per_codebook.append(logits_BFK)
        logits_BFMK = torch.stack(logits_per_codebook, dim=2)

        output = {
            "logits_BFMK": logits_BFMK,
            "mask_BF": mask_BF,
            "features_pen": features_pen,
        }

        if cqt_target_BFcV is not None:
            output["cqt_target_BFV"] = cqt_target_BFcV
            output["cqt_pred_BFV"] = self.cqt_proj(x_BFD)

        return output
