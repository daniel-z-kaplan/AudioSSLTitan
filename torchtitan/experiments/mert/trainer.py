# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MERT's bespoke training loop.

MERT is not an autoregressive next-token decoder, so it does not fit the
generic ``Trainer``'s ``preprocess_inputs`` / single-``labels``-tensor
contract (see ``BaseModel.preprocess_inputs``'s docstring, which calls out
exactly this case). Following the precedent set by
``torchtitan.models.flux.trainer.FluxTrainer`` for the same reason, this
overrides only the training-step methods; model construction, parallelization,
checkpointing, and validation still go through the base ``Trainer.__init__``
unmodified.

Known limitation (matching ``FluxTrainer``): gradient accumulation and
pipeline parallelism are not supported.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any

import spmd_types as spmd
import torch

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer


class MERTTrainer(Trainer):
    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Override to count encoder frames (``batch_size * num_frames``)
        rather than assuming ``labels`` are per-sample token ids."""
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            input_dict, labels_MBF = batch
            num_codebooks, bsz, num_frames = labels_MBF.shape
            self.metrics_processor.ntokens_since_last_log += bsz * num_frames
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield input_dict, labels_MBF

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert global_valid_tokens is None, (
            "MertLoss normalizes internally; MERTTrainer does not compute "
            "global_valid_tokens (see MertLoss docstring)."
        )
        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)

        model = self.model_parts[0]
        source_BT = input_dict["source"]
        self.ntokens_seen += labels.shape[1] * labels.shape[2]

        with self.train_context():
            pred = model(source_BT)
            loss, metrics = self.loss_fn(pred, labels)
            del pred
            with spmd.no_typecheck():
                loss.backward()

        return loss, metrics

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> None:
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        lr_metrics = self.lr_schedulers.get_metrics()
        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError(
                "MERTTrainer does not support gradient accumulation "
                "(training.num_tokens_per_train_step must equal one "
                "microbatch's worth of tokens across the data-parallel group)."
            )
        if parallel_dims.pp_enabled:
            raise ValueError("MERTTrainer does not support pipeline parallelism.")

        input_dict, labels = next(data_iterator)
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        loss, metrics = self.forward_backward_step(input_dict=input_dict, labels=labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        if not self.metrics_processor.should_log(self.step):
            return

        loss = loss.detach()
        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_mean(loss, loss_mesh)
            global_max_loss = dist_utils.dist_max(loss, loss_mesh)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
            )
        else:
            global_avg_loss = global_max_loss = float(loss.item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics: dict[str, Any] = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **{k: float(v.item()) for k, v in metrics.items()},
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
