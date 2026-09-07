# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer configs for the MERT experiment.

Run with, e.g.::

    python -m torchtitan.train --module mert --config mert_debugmodel

``mert_95m`` / ``mert_330m`` reproduce the hyperparameters of
``ReferenceRepos/MERT/mert_fairseq/config/pretrain/MERT_RVQ-VAE_CQT_{95M,330M}.yaml``,
but train on synthetic random audio by default (no acoustic-teacher labels
are shipped with this repo). Point ``dataloader.manifest_path`` /
``dataloader.label_paths`` at real data prepared per the reference's
``scripts/prepare_codecs_from_manifest.py`` to pretrain for real; see the
experiment README.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC

from . import model_registry
from .dataset import MERTDataLoader
from .loss import MertLoss
from .model import downsample_factor, required_input_length
from .trainer import MERTTrainer

# Real audio manifests are unrelated to text, but the base Trainer
# unconditionally builds a tokenizer; point it at the repo's existing tiny
# test tokenizer rather than teaching Trainer to skip tokenizer construction.
_UNUSED_TOKENIZER_PATH = "./tests/assets/tokenizer"


def _mert_trainer_config(
    flavor: str,
    *,
    num_frames: int,
    batch_size: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    warmup_steps: int,
    max_norm: float,
    steps: int,
    feature_pen_weight: float,
    cqt_loss_weight: float,
) -> MERTTrainer.Config:
    model_spec = model_registry(flavor, num_frames=num_frames)
    conv_layers = model_spec.model.conv_layers  # pyrefly: ignore[missing-attribute]
    return MERTTrainer.Config(
        loss=MertLoss.Config(
            feature_pen_weight=feature_pen_weight,
            cqt_loss_weight=cqt_loss_weight,
        ),
        hf_assets_path=_UNUSED_TOKENIZER_PATH,
        model_spec=model_spec,
        optimizer=default_adamw(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=batch_size * num_frames,
            max_context_length=num_frames,
            max_norm=max_norm,
            steps=steps,
        ),
        dataloader=MERTDataLoader.Config(
            num_codebooks=model_spec.model.num_codebooks,  # pyrefly: ignore[missing-attribute]
            codebook_size=model_spec.model.codebook_size,  # pyrefly: ignore[missing-attribute]
            crop_samples=required_input_length(conv_layers, num_frames),
            downsample_factor=downsample_factor(conv_layers),
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        # See the "Parallelism" section of the README: MERT's non-causal
        # encoder is parallelized with classic FSDP2, not the spmd_types
        # sharding-config annotations used by the core decoder-only models.
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
        checkpoint=CheckpointManager.Config(interval=1000, last_save_model_only=False),
        activation_checkpoint=SelectiveAC.Config(),
    )


def mert_debugmodel(num_frames: int | None = None) -> MERTTrainer.Config:
    return _mert_trainer_config(
        "debugmodel",
        num_frames=num_frames or 50,
        batch_size=4,
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
        warmup_steps=2,
        max_norm=10.0,
        steps=10,
        feature_pen_weight=10.0,
        cqt_loss_weight=1.0,
    )


def mert_95m(num_frames: int | None = None) -> MERTTrainer.Config:
    return _mert_trainer_config(
        "95M",
        num_frames=num_frames or 375,
        batch_size=32,
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
        warmup_steps=32000,
        max_norm=10.0,
        steps=400000,
        feature_pen_weight=10.0,
        cqt_loss_weight=1.0,
    )


def mert_330m(num_frames: int | None = None) -> MERTTrainer.Config:
    return _mert_trainer_config(
        "330M",
        num_frames=num_frames or 384,
        batch_size=16,
        lr=1.5e-3,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
        warmup_steps=32000,
        max_norm=1.0,
        steps=400000,
        feature_pen_weight=10.0,
        cqt_loss_weight=1.0,
    )
