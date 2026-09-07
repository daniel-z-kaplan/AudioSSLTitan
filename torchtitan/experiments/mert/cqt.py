# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Constant-Q Transform (CQT), used as MERT's "musical teacher" reconstruction
target (``audio_cqt_loss_m`` in the reference training code).

The reference implementation (``ReferenceRepos/MERT``) computes this via the
third-party ``nnAudio`` library. This module reimplements the same standard
kernel-based CQT algorithm (Brown 1991; Schoerkhuber & Klapuri 2010) directly
with ``torch.nn.functional.conv1d``, so the experiment has no dependency on
an unmaintained third-party package. Constants (kernel normalization, hann
window) follow the common convention but are not bit-exact with ``nnAudio``;
see the experiment README for this and other noted deviations from the
reference code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.protocols.module import Module


class ConstantQTransform(Module):
    """Magnitude Constant-Q Transform, implemented as grouped 1D convolution.

    Shape suffix legend: B = batch, T = input samples, F = output frames,
    K = number of CQT bins (``n_bins``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        sample_rate: int
        hop_length: int
        n_bins: int = 84
        bins_per_octave: int | None = None
        """Defaults to ``n_bins // 7`` (one octave per 7 bins), matching the
        reference training code's CQT configuration."""
        fmin: float = 32.7
        filter_scale: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_bins = config.n_bins
        self.hop_length = config.hop_length
        self.bins_per_octave = config.bins_per_octave or max(1, config.n_bins // 7)
        self.q_factor = config.filter_scale / (2 ** (1.0 / self.bins_per_octave) - 1.0)
        lengths = self._kernel_lengths()
        self.max_kernel_length = max(lengths)
        kernel_real, kernel_imag = self._compute_kernels(lengths)
        self.register_buffer("kernel_real", kernel_real, persistent=False)
        self.register_buffer("kernel_imag", kernel_imag, persistent=False)

    def _kernel_lengths(self) -> list[int]:
        # Plain Python/``math`` (not ``torch``) so this is unaffected by the
        # ambient ``torch.device("meta")`` context used during model
        # construction: it fixes structural values (loop bounds, padding
        # amounts) that must be real numbers even before real device
        # materialization, unlike the kernel tensor *contents* below.
        lengths = []
        for k in range(self.n_bins):
            freq_k = self.config.fmin * 2.0 ** (k / self.bins_per_octave)
            lengths.append(math.ceil(self.q_factor * self.config.sample_rate / freq_k))
        return lengths

    def _compute_kernels(self, lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = self.max_kernel_length
        kernel_real = torch.zeros(self.n_bins, 1, max_len, dtype=torch.float32)
        kernel_imag = torch.zeros(self.n_bins, 1, max_len, dtype=torch.float32)
        for k in range(self.n_bins):
            n_k = lengths[k]
            n = torch.arange(n_k, dtype=torch.float64)
            window = torch.hann_window(n_k, periodic=False, dtype=torch.float64)
            phase = 2.0 * math.pi * self.q_factor * n / n_k
            envelope = window / n_k
            start = (max_len - n_k) // 2
            kernel_real[k, 0, start : start + n_k] = (
                envelope * torch.cos(phase)
            ).float()
            kernel_imag[k, 0, start : start + n_k] = (
                envelope * torch.sin(phase)
            ).float()
        return kernel_real, kernel_imag

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.kernel_real.device
        lengths = self._kernel_lengths()
        kernel_real, kernel_imag = self._compute_kernels(lengths)
        self.kernel_real = kernel_real.to(buffer_device)
        self.kernel_imag = kernel_imag.to(buffer_device)

    def forward(self, waveform_BT: torch.Tensor) -> torch.Tensor:
        """Compute the CQT magnitude spectrogram.

        Args:
            waveform_BT: raw audio, shape ``(B, T)``.

        Returns:
            Magnitude CQT, shape ``(B, F, K)`` (frame-major, matching the
            transformer's ``(B, T, C)`` convention).
        """
        pad = self.max_kernel_length // 2
        x_B1T = F.pad(waveform_BT, (pad, pad), mode="constant").unsqueeze(1)
        real_BKF = F.conv1d(x_B1T, self.kernel_real, stride=self.hop_length)
        imag_BKF = F.conv1d(x_B1T, self.kernel_imag, stride=self.hop_length)
        magnitude_BKF = torch.sqrt(real_BKF.pow(2) + imag_BKF.pow(2) + 1e-10)
        return magnitude_BKF.transpose(1, 2)
