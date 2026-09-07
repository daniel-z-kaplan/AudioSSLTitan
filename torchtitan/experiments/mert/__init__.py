# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MERT (Music undERstanding model with large-scale self-supervised Training).

See README.md for background, scope, and known deviations from the
reference training code in ``ReferenceRepos/MERT``.
"""

from torchtitan.protocols.model_spec import ModelSpec

from .model import MERTModel
from .parallelize import parallelize_mert

__all__ = ["parallelize_mert", "MERTModel", "mert_configs", "model_registry"]


def _95m() -> MERTModel.Config:
    return MERTModel.Config(
        encoder_embed_dim=768,
        encoder_layers=12,
        encoder_attention_heads=12,
        encoder_ffn_embed_dim=3072,
        layer_norm_first=False,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        dropout_input=0.1,
        dropout_features=0.1,
        encoder_layerdrop=0.05,
        final_dim=64,
        untie_final_proj=True,
        feature_grad_mult=0.1,
        mask_prob=0.8,
        mask_length=5,
        audio_cqt_loss_m=True,
        audio_cqt_bins=336,
        param_init={"mask_embedding_MD": lambda p: p.uniform_()},
    )


def _330m() -> MERTModel.Config:
    return MERTModel.Config(
        encoder_embed_dim=1024,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_embed_dim=4096,
        layer_norm_first=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        dropout_input=0.0,
        dropout_features=0.0,
        encoder_layerdrop=0.0,
        final_dim=128,
        untie_final_proj=False,
        feature_grad_mult=1.0,
        mask_prob=0.8,
        mask_length=5,
        audio_cqt_loss_m=False,
        param_init={"mask_embedding_MD": lambda p: p.uniform_()},
    )


def _debugmodel() -> MERTModel.Config:
    return MERTModel.Config(
        conv_layers=((32, 10, 5), (32, 3, 2), (32, 3, 2), (32, 2, 2)),
        encoder_embed_dim=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_embed_dim=64,
        final_dim=16,
        untie_final_proj=True,
        mask_prob=0.65,
        mask_length=3,
        num_codebooks=4,
        codebook_size=32,
        # sample_rate / downsample_factor(conv_layers) = 16000 / 40 = 400,
        # matched to label_rate so the CQT head's hop_length lines up with
        # the (nominal) conv downsampling -- see model.py's
        # ``conv_output_length``/``downsample_factor`` docstrings for why
        # this is only approximate, not exact.
        sample_rate=16000,
        label_rate=400.0,
        audio_cqt_loss_m=True,
        audio_cqt_bins=32,
        param_init={"mask_embedding_MD": lambda p: p.uniform_()},
    )


# (config factory, default num_frames) per flavor. ``num_frames`` is the
# model's sequence length in encoder frames -- see model.py's
# ``get_nparams_and_flops`` docstring; raw audio crop length is
# ``num_frames * downsample_factor(conv_layers)``.
mert_configs = {
    "95M": (_95m, 375),  # 5s @ 24kHz / 320
    "330M": (_330m, 384),  # 5.12s @ 24kHz / 320, matches the reference config
    "debugmodel": (_debugmodel, 50),
}


def model_registry(flavor: str, *, num_frames: int | None = None) -> ModelSpec:
    get_config, default_num_frames = mert_configs[flavor]
    config = get_config()
    return ModelSpec(
        name="mert",
        flavor=flavor,
        model=config,
        max_context_length=num_frames or default_num_frames,
        parallelize_fn=parallelize_mert,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
