# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.experiments.mert import _debugmodel
from torchtitan.experiments.mert.cqt import ConstantQTransform
from torchtitan.experiments.mert.dataset import MERTDataLoader
from torchtitan.experiments.mert.loss import MertLoss
from torchtitan.experiments.mert.masking import compute_mask_indices
from torchtitan.experiments.mert.model import MERTModel, required_input_length


def _build_debug_model() -> MERTModel:
    """Build MERTModel the same way the real Trainer does: construct on meta
    device, materialize, then run ``init_states()`` -- this is the path that
    exercises every ``param_init`` callback and every ``_init_self_buffers``
    override (CQT kernels, mask embedding)."""
    config = _debugmodel()
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_weights()
    return model


def test_compute_mask_indices_shape_and_min_masks():
    mask = compute_mask_indices(
        (4, 50), padding_mask=None, mask_prob=0.5, mask_length=3, min_masks=2
    )
    assert mask.shape == (4, 50)
    assert mask.dtype == bool
    assert mask.sum(axis=1).min() >= 2


def test_constant_q_transform_output_shape():
    config = ConstantQTransform.Config(sample_rate=24000, hop_length=320, n_bins=32)
    with torch.device("meta"):
        cqt = config.build()
    cqt.to_empty(device="cpu")
    with torch.no_grad():
        cqt.init_states()
    waveform = torch.randn(2, 16000)
    out = cqt(waveform)
    assert out.shape[0] == 2
    assert out.shape[2] == 32
    assert torch.isfinite(out).all()


def test_mert_model_forward_shapes():
    model = _build_debug_model()
    model.eval()
    num_frames = 50
    crop_samples = required_input_length(model.config.conv_layers, num_frames)
    source = torch.randn(2, crop_samples) * 0.1

    with torch.no_grad():
        out = model(source)

    # The CQT head's own framing can force a small crop relative to the
    # feature extractor's exact frame count (see MERTModel.forward), so
    # assert mutual consistency and closeness to num_frames rather than
    # exact equality.
    actual_frames = out["logits_BFMK"].shape[1]
    assert abs(actual_frames - num_frames) <= 2
    assert out["logits_BFMK"].shape == (
        2,
        actual_frames,
        model.config.num_codebooks,
        model.config.codebook_size,
    )
    assert out["mask_BF"].shape == (2, actual_frames)
    assert out["mask_BF"].dtype == torch.bool
    assert torch.isfinite(out["features_pen"])
    assert out["cqt_pred_BFV"].shape == out["cqt_target_BFV"].shape
    assert out["cqt_pred_BFV"].shape[:2] == (2, actual_frames)
    assert out["cqt_pred_BFV"].shape[2] == model.config.audio_cqt_bins


def test_mert_forward_backward_updates_all_parameters():
    model = _build_debug_model()
    model.train()
    num_frames = 50
    crop_samples = required_input_length(model.config.conv_layers, num_frames)
    source = torch.randn(2, crop_samples) * 0.1

    pred = model(source)
    labels_MBF = torch.randint(
        0, model.config.codebook_size, (model.config.num_codebooks, 2, num_frames)
    )

    loss_fn = MertLoss.Config(feature_pen_weight=10.0, cqt_loss_weight=1.0).build(
        compile_config=None
    )
    loss, metrics = loss_fn(pred, labels_MBF)
    assert torch.isfinite(loss)
    assert "masked_ce" in metrics and "cqt_mse" in metrics

    loss.backward()

    missing_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_synthetic_dataloader_batch_shapes():
    num_frames = 20
    batch_size = 3
    crop_samples = 20 * 320  # exact value doesn't matter for the synthetic source
    loader = MERTDataLoader.Config(
        num_codebooks=4, codebook_size=32, crop_samples=crop_samples
    ).build(
        dp_world_size=1,
        dp_rank=0,
        max_context_length=num_frames,
        num_tokens_per_batch=batch_size * num_frames,
    )
    input_dict, labels_MBF = next(iter(loader))
    assert input_dict["source"].shape == (batch_size, crop_samples)
    assert labels_MBF.shape == (4, batch_size, num_frames)
