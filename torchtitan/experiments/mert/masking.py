# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Span masking for MERT/HuBERT-style masked prediction pretraining.

Self-contained reimplementation of the span masking algorithm used by the
reference training code (``fairseq.data.data_utils.compute_mask_indices``,
via ``mert_fairseq.models.mert.mert_model.MERTModel.apply_mask``). Vendored
here instead of depending on fairseq, which this repo does not otherwise use.
"""

from __future__ import annotations

import numpy as np


def compute_mask_indices(
    shape: tuple[int, int],
    padding_mask: np.ndarray | None,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 1,
) -> np.ndarray:
    """Compute boolean span-masking indices.

    Args:
        shape: ``(batch_size, sequence_length)``.
        padding_mask: ``(batch_size, sequence_length)`` bool array, True at
            padded positions, or None if there is no padding.
        mask_prob: fraction of the (unpadded) sequence to mask, spread across
            spans of length ``mask_length`` (so the total masked fraction is
            approximately ``mask_prob``, not exactly, since spans can overlap
            or run past the sequence boundary).
        mask_length: span length.
        mask_type: "static" (fixed span length), or "uniform" (span length
            drawn from ``[mask_other, 2 * mask_length]``).
        mask_other: secondary parameter for non-static mask types.
        min_masks: minimum number of spans per sample.
        no_overlap: if True, sample non-overlapping spans via interval
            splitting; otherwise spans are sampled independently and may
            overlap or merge.
        min_space: minimum gap enforced between spans when ``no_overlap``.

    Returns:
        ``(batch_size, sequence_length)`` bool array, True at masked
        positions.
    """
    batch_size, sequence_length = shape
    mask = np.full((batch_size, sequence_length), False)

    all_num_mask = max(
        min_masks,
        int(mask_prob * sequence_length / float(mask_length) + np.random.rand()),
    )

    mask_indices_per_sample: list[np.ndarray] = []
    for i in range(batch_size):
        if padding_mask is not None:
            valid_size = sequence_length - padding_mask[i].astype(np.int64).sum()
            num_mask = max(
                min_masks,
                int(mask_prob * valid_size / float(mask_length) + np.random.rand()),
            )
        else:
            valid_size = sequence_length
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        else:
            raise ValueError(f"unknown mask_type {mask_type!r}")

        if lengths.sum() == 0:
            lengths[0] = min(mask_length, valid_size - 1)

        if no_overlap:
            span_starts: list[int] = []

            def arrange(
                start: int, end: int, length: int, keep_length: int
            ) -> list[tuple[int, int]]:
                span_start = np.random.randint(start, end - length)
                span_starts.extend(span_start + offset for offset in range(length))
                new_parts = []
                if span_start - start - min_space >= keep_length:
                    new_parts.append((start, span_start - min_space + 1))
                if end - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, end))
                return new_parts

            parts = [(0, valid_size)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    dtype=np.int64,
                )
                total = int(lens.sum())
                if total == 0:
                    break
                probs = lens / total
                choice = np.random.choice(len(parts), p=probs)
                start, end = parts.pop(choice)
                parts.extend(arrange(start, end, length, min_length))
            mask_idc = np.asarray(span_starts, dtype=np.int64)
        else:
            min_len = min(lengths)
            if valid_size - min_len <= num_mask:
                min_len = valid_size - num_mask - 1
            span_starts = np.random.choice(
                valid_size - min_len, num_mask, replace=False
            )
            mask_idc = np.asarray(
                [
                    span_starts[j] + offset
                    for j in range(len(span_starts))
                    for offset in range(lengths[j])
                ],
                dtype=np.int64,
            )

        mask_idc = np.unique(mask_idc[mask_idc < valid_size])
        mask_indices_per_sample.append(mask_idc)

    min_len = min(len(m) for m in mask_indices_per_sample)
    for i, mask_idc in enumerate(mask_indices_per_sample):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask
