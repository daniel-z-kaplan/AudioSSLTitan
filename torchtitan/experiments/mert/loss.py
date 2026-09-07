# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MERT's masked multi-codebook prediction loss, plus the auxiliary CQT
reconstruction and feature-penalty terms from the reference training code
(``mert_fairseq.models.mert.mert_model.MERTModel.forward`` /
the fairseq ``hubert`` criterion).

Shape suffix legend: B = batch, F = encoder frames, M = number of codebooks,
K = classes per codebook, V = number of CQT bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.components.loss import BaseLoss
from torchtitan.config import CompileConfig


class MertLoss(BaseLoss):
    """Combines masked multi-codebook cross-entropy with the auxiliary
    feature-penalty and (optional) CQT reconstruction terms.

    Unlike most ``BaseLoss`` subclasses, ``pred`` here is the dict returned
    by ``MERTModel.forward`` (not a single logits tensor): MERT's objective
    needs several model outputs (per-codebook logits, the mask, the CQT
    prediction/target) together with the label tensor, so a single
    ``(pred, labels) -> loss`` reduction does not fit.

    ``labels_MBF`` stacks the per-codebook target label sets along a new
    leading dim (see ``dataset.py``), matching how the reference training
    code is given ``target_list: List[Tensor]``.

    Uses a plain per-rank mean reduction (rather than the reference
    fairseq criterion's sum-then-global-sample_size normalization): with
    FSDP/DDP's default gradient averaging and a fixed per-rank batch shape
    (see ``dataset.py``), this is the more portable choice for this
    framework and does not require ``global_valid_tokens``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        feature_pen_weight: float = 10.0
        """Weight on ``features_pen`` (matches the reference's
        ``criterion.loss_weights[0]``)."""
        cqt_loss_weight: float = 1.0
        """Weight on the CQT reconstruction MSE (matches the reference's
        ``criterion.loss_weights[1]``); unused when the model has no CQT
        head."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.feature_pen_weight = config.feature_pen_weight
        self.cqt_loss_weight = config.cqt_loss_weight

    def __call__(
        self,
        pred: dict[str, torch.Tensor],
        labels_MBF: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert global_valid_tokens is None, (
            "MertLoss normalizes internally (per-rank mean); it does not use "
            "global_valid_tokens."
        )
        logits_BFMK = pred["logits_BFMK"]
        mask_BF = pred["mask_BF"]
        num_codebooks = logits_BFMK.shape[2]

        # The model may emit fewer frames than the dataset's label window
        # (e.g. the auxiliary CQT head's own framing can force a small crop;
        # see ``MERTModel.forward``). Align on the model's frame count.
        num_frames = logits_BFMK.shape[1]
        labels_MBF = labels_MBF[:, :, :num_frames]

        labels_BFM = labels_MBF.permute(1, 2, 0)
        mask_flat = mask_BF.reshape(-1).float()
        num_masked = mask_flat.sum().clamp(min=1.0)

        masked_ce = logits_BFMK.new_zeros(())
        for codebook in range(num_codebooks):
            logits_NK = logits_BFMK[:, :, codebook, :].reshape(
                -1, logits_BFMK.shape[-1]
            )
            labels_N = labels_BFM[:, :, codebook].reshape(-1)
            per_token_ce = F.cross_entropy(
                logits_NK.float(), labels_N, reduction="none"
            )
            masked_ce = masked_ce + (per_token_ce * mask_flat).sum() / num_masked
        masked_ce = masked_ce / num_codebooks

        features_pen = pred["features_pen"]
        loss = masked_ce + self.feature_pen_weight * features_pen

        cqt_mse = None
        if "cqt_pred_BFV" in pred:
            cqt_pred_BFV = pred["cqt_pred_BFV"]
            cqt_target_BFV = pred["cqt_target_BFV"]
            se_BFV = (cqt_pred_BFV.float() - cqt_target_BFV.float()) ** 2
            cqt_mse = (se_BFV.mean(dim=-1) * mask_BF.float()).sum() / num_masked
            loss = loss + self.cqt_loss_weight * cqt_mse

        metrics = {
            "masked_ce": masked_ce.detach(),
            "features_pen": features_pen.detach(),
        }
        if cqt_mse is not None:
            metrics["cqt_mse"] = cqt_mse.detach()
        return loss, metrics
