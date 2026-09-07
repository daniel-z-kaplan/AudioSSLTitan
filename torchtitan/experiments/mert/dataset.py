# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MERT pretraining data loading.

MERT's masked-prediction targets (the RVQ-VAE acoustic-teacher codebook
indices) are precomputed offline by a separate tokenizer, not produced by
this training loop -- see ``ReferenceRepos/MERT/scripts/prepare_codecs_from_manifest.py``
and the experiment README. This module therefore expects, for real data, the
same manifest + per-codebook label-file layout as the reference's
``MERTDataset``:

- a manifest ``.tsv``: first line is the audio root dir, each following line
  is ``<relative_path>\\t<num_samples>``.
- one label file per codebook (``label_paths``), each with one line per
  manifest entry: whitespace-separated integer codes, at ``label_rate``.

For quick iteration and CI (no on-disk assets required), ``manifest_path=None``
(the default) generates synthetic random audio and labels instead.

Every crop -- real or synthetic -- has the same fixed length, so every batch
has a fixed shape (required for this framework's default CUDA-graph /
``torch.compile`` paths). This differs from the reference's ``pad_audio=False,
random_crop=True`` dataset, which crops each batch to that batch's shortest
sample rather than a fixed length.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.loader import BaseDataLoader


class _SyntheticSource:
    """Deterministic per-rank stream of random audio crops + codebook labels."""

    def __init__(
        self,
        *,
        crop_samples: int,
        num_frames: int,
        num_codebooks: int,
        codebook_size: int,
        seed: int,
    ):
        self.crop_samples = crop_samples
        self.num_frames = num_frames
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.generator = torch.Generator().manual_seed(seed)

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        waveform_T = torch.randn(self.crop_samples, generator=self.generator) * 0.1
        labels_FM = torch.randint(
            0,
            self.codebook_size,
            (self.num_frames, self.num_codebooks),
            generator=self.generator,
        )
        return waveform_T, labels_FM

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.generator.set_state(state_dict["generator_state"])


class _ManifestSource:
    """Reads fixed-length crops from a manifest of on-disk audio files, with
    per-codebook integer labels aligned by crop offset."""

    def __init__(
        self,
        *,
        manifest_path: str,
        label_paths: list[str],
        crop_samples: int,
        num_frames: int,
        downsample_factor: int,
        dp_rank: int,
        dp_world_size: int,
        seed: int,
    ):
        import torchaudio

        self._torchaudio = torchaudio
        self.crop_samples = crop_samples
        self.num_frames = num_frames
        self.downsample_factor = downsample_factor

        with open(manifest_path) as f:
            self.audio_root = f.readline().strip()
            entries = []
            for line in f:
                rel_path, num_samples = line.rstrip("\n").split("\t")
                entries.append((rel_path, int(num_samples)))

        self.label_paths = label_paths
        self._labels_by_codebook: list[list[str]] = []
        for label_path in label_paths:
            with open(label_path) as f:
                lines = f.read().splitlines()
            if len(lines) != len(entries):
                raise ValueError(
                    f"label file {label_path} has {len(lines)} lines, "
                    f"expected {len(entries)} (one per manifest entry)"
                )
            self._labels_by_codebook.append(lines)

        # Filter out clips too short for a single fixed-length crop.
        self.entries = [
            (i, rel_path, num_samples)
            for i, (rel_path, num_samples) in enumerate(entries)
            if num_samples >= crop_samples
        ]
        if not self.entries:
            raise ValueError(
                f"No manifest entries are >= crop_samples ({crop_samples}); "
                "reduce training.max_context_length or use longer audio."
            )

        self._rng = random.Random(seed + dp_rank)
        self._shard = self.entries[dp_rank::dp_world_size] or self.entries[:1]
        self._rng.shuffle(self._shard)
        self._position = 0

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._position >= len(self._shard):
            self._rng.shuffle(self._shard)
            self._position = 0
        entry_idx, rel_path, num_samples = self._shard[self._position]
        self._position += 1

        max_start = num_samples - self.crop_samples
        start_sample = self._rng.randint(0, max_start) if max_start > 0 else 0
        waveform, _ = self._torchaudio.load(
            f"{self.audio_root}/{rel_path}",
            frame_offset=start_sample,
            num_frames=self.crop_samples,
        )
        waveform_T = waveform.mean(dim=0)  # downmix to mono

        start_frame = start_sample // self.downsample_factor
        labels_per_codebook = []
        for lines in self._labels_by_codebook:
            codes = lines[entry_idx].split()
            window = codes[start_frame : start_frame + self.num_frames]
            if len(window) < self.num_frames:
                window = window + [window[-1]] * (self.num_frames - len(window))
            labels_per_codebook.append(torch.tensor([int(c) for c in window]))
        labels_FM = torch.stack(labels_per_codebook, dim=1)
        return waveform_T, labels_FM

    def state_dict(self) -> dict[str, Any]:
        return {"position": self._position, "shard_order": list(self._shard)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._position = state_dict["position"]
        self._shard = state_dict["shard_order"]


class MERTDataLoader(BaseDataLoader):
    """Yields ``(input_dict, labels_MBF)`` batches for MERT pretraining.

    ``input_dict`` has a single key, ``"source"``: raw audio, shape
    ``(B, crop_samples)``. ``labels_MBF`` stacks the per-codebook integer
    targets, shape ``(num_codebooks, B, num_frames)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        num_codebooks: int = 8
        codebook_size: int = 1024
        crop_samples: int = 0
        """Raw audio samples per crop. Must be computed from the model's
        ``conv_layers`` via ``model.required_input_length`` so that the
        resulting number of encoder frames exactly equals
        ``training.max_context_length`` -- the conv feature extractor's
        input/output length ratio is not simply ``num_frames *
        downsample_factor`` (see ``model.conv_output_length``'s docstring).
        Set by the config registry; there is no sane model-independent
        default."""
        downsample_factor: int = 320
        """Nominal stride product, used only to approximate which stretch of
        a manifest label file corresponds to an audio crop's offset (see
        ``model.downsample_factor``'s docstring for why this is approximate,
        not exact)."""
        manifest_path: str | None = None
        """Path to a fairseq-style manifest ``.tsv``. If None, generates
        synthetic random audio and labels (used by the debug config and
        unit tests)."""
        label_paths: list[str] = field(default_factory=list)
        """One label file per codebook, required when ``manifest_path`` is
        set; must have length ``num_codebooks``."""
        seed: int = 42

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.config = config
        # ``max_context_length`` / ``num_tokens_per_batch`` are counted in
        # frames (see model.py's ``get_nparams_and_flops`` docstring), not
        # raw audio samples.
        self.num_frames = max_context_length
        if num_tokens_per_batch % self.num_frames != 0:
            raise ValueError(
                f"num_tokens_per_batch ({num_tokens_per_batch}) must be a "
                f"multiple of max_context_length ({self.num_frames})."
            )
        self.batch_size = num_tokens_per_batch // self.num_frames
        if config.crop_samples <= 0:
            raise ValueError(
                "MERTDataLoader.Config.crop_samples must be set (via "
                "model.required_input_length(conv_layers, num_frames)); "
                f"got {config.crop_samples}."
            )
        self.crop_samples = config.crop_samples

        if config.manifest_path is not None:
            if len(config.label_paths) != config.num_codebooks:
                raise ValueError(
                    f"label_paths has {len(config.label_paths)} entries, "
                    f"expected num_codebooks ({config.num_codebooks})"
                )
            self._source: _SyntheticSource | _ManifestSource = _ManifestSource(
                manifest_path=config.manifest_path,
                label_paths=config.label_paths,
                crop_samples=self.crop_samples,
                num_frames=self.num_frames,
                downsample_factor=config.downsample_factor,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                seed=config.seed,
            )
        else:
            self._source = _SyntheticSource(
                crop_samples=self.crop_samples,
                num_frames=self.num_frames,
                num_codebooks=config.num_codebooks,
                codebook_size=config.codebook_size,
                seed=config.seed + dp_rank,
            )

    def __iter__(self):
        return self

    def __next__(self) -> TrainerBatch:
        waveforms = []
        labels = []
        for _ in range(self.batch_size):
            waveform_T, labels_FM = next(self._source)
            waveforms.append(waveform_T)
            labels.append(labels_FM)
        source_BT = torch.stack(waveforms)
        labels_BFM = torch.stack(labels)
        labels_MBF = labels_BFM.permute(2, 0, 1).contiguous()
        return {"source": source_BT}, labels_MBF

    def state_dict(self) -> dict[str, Any]:
        return self._source.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            self._source.load_state_dict(state_dict)
