# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelize MERT with FSDP2, following the same pattern as
``torchtitan.models.flux.parallelize`` (a non-decoder model with no tensor
parallelism support): activation checkpointing and ``fully_shard`` are
applied directly, without going through the SPMD-types ``Module.parallelize``
sharding-config machinery (MERT declares no ``sharding_config`` on any
submodule, so that call would be a no-op walk in any case).
"""

from __future__ import annotations

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.tools.logging import logger

from .model import MERTModel


def parallelize_mert(
    model: MERTModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> MERTModel:
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model.encoder)

    if parallelism.spmd_backend == "spmd_types":
        # No submodule declares a ``sharding_config``, so this only walks the
        # module tree; it exists for parity with models (e.g. Flux) that do
        # mix in TP-sharded submodules under this backend.
        model.parallelize(parallel_dims)
        annotate_replicated_parameters(model, parallel_dims)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(
            model.encoder, compile_config=compile_config, parallel_dims=parallel_dims
        )

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    assert dp_mesh is not None, "MERT requires a data-parallel mesh (FSDP or DDP)."

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cpu_offload=training.enable_cpu_offload,
    )

    logger.info("Applied fully_shard to MERT")
    return model


def apply_fsdp(
    model: MERTModel,
    dp_mesh: DeviceMesh,
    *,
    param_dtype,
    reduce_dtype,
    cpu_offload: bool = False,
) -> None:
    from torch.distributed.fsdp import CPUOffloadPolicy

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        "default", pp_enabled=False
    )

    fully_shard(
        model.feature_extractor,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    for transformer_block in model.encoder.layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # Group the remaining (non-transformer-block) parameters -- projections,
    # codebook embeddings, positional conv, norms -- into one FSDP unit.
    # Unlike token-normalized LM losses, MertLoss is a plain per-rank mean,
    # so FSDP's default gradient *averaging* (not sum) across ranks is kept
    # as-is here (no ``disable_fsdp_gradient_division``).
    fully_shard(model, **fsdp_config)
