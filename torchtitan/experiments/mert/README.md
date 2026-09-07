# MERT

MERT (**M**usic und**ER**standing model with large-scale self-supervised
**T**raining) is a HuBERT/wav2vec2-style self-supervised speech/music model:
a strided CNN feature extractor over raw audio feeds a bidirectional
Transformer encoder, trained with a masked multi-codebook classification
objective (plus an auxiliary Constant-Q reconstruction loss) against
pseudo-labels from a separately trained acoustic tokenizer.

- Paper: [MERT: Acoustic Music Understanding Model with Large-Scale
  Self-supervised Training](https://arxiv.org/abs/2306.00107) (Li et al.,
  ICLR 2024)
- Reference training code: `ReferenceRepos/MERT` (fairseq), specifically
  `mert_fairseq/models/mert/mert_model.py`. Where the paper and the reference
  training code disagree, this implementation follows the training code.

## Scope

This experiment implements the MERT **student model and pretraining loop**:
the CNN + Transformer architecture, span masking, the multi-codebook
cosine-similarity classification head, and the auxiliary CQT reconstruction
head, wired into torchtitan's model/config/parallelism/checkpoint machinery.

It does **not** implement the acoustic tokenizer (RVQ-VAE or k-means) that
produces MERT's pseudo-labels; the reference training code doesn't either --
that tokenizer is trained separately and its output codes are precomputed
onto disk before the run in `ReferenceRepos/MERT/scripts/run_training.sh`.
See `ReferenceRepos/MERT/scripts/prepare_codecs_from_manifest.py` (RVQ-VAE,
via EnCodec) and `simple_kmeans` (k-means) for how the reference produces
these. Without real labels, `mert_95m` / `mert_330m` / `mert_debugmodel`
train against synthetic random audio and labels by default (see
"Training on real data" below).

## Architecture

| Component | File | Notes |
|---|---|---|
| Strided CNN feature extractor | `model.py: MERTFeatureExtractor` | wav2vec2 "default" mode: GroupNorm + GELU after the first conv layer, GELU-only after |
| Positional embedding | `model.py: MERTPositionalConvEmbedding` | grouped conv, added (not concatenated) to the encoder input |
| Bidirectional attention | `model.py: MERTAttention` | no causal mask, no RoPE (encoder, not decoder) |
| Transformer encoder | `model.py: MERTEncoder` / `MERTEncoderLayer` | supports both post-LN (95M) and pre-LN (330M) |
| Span masking | `masking.py` | vendored port of `fairseq.data.data_utils.compute_mask_indices` |
| Multi-codebook classification | `model.py: MERTModel.forward` | cosine similarity to a learned per-codebook embedding table, temperature-scaled -- mathematically equivalent to the reference's noise-contrastive loss against the full codebook as negatives (see below) |
| CQT reconstruction | `cqt.py` | auxiliary "musical teacher" head |
| Loss | `loss.py: MertLoss` | masked cross-entropy (summed over codebooks) + feature penalty + CQT MSE |
| Data loading | `dataset.py: MERTDataLoader` | manifest + per-codebook label files, or synthetic |
| Parallelism | `parallelize.py` | FSDP2 (+ HSDP via `dp_replicate`), activation checkpointing, `torch.compile` |
| Training loop | `trainer.py: MERTTrainer` | bespoke, not the generic decoder `Trainer` (see below) |

### Why a bespoke `Trainer` and dense (not gathered) masked positions

`BaseModel.preprocess_inputs` (`torchtitan/protocols/model.py`) is built
around a single `labels` tensor and next-token-style losses; its docstring
explicitly calls out models with a "bespoke pipeline" (citing Flux) as not
using it. MERT's loss needs several model outputs together (per-codebook
logits, the mask, the CQT prediction/target) plus multiple codebooks worth of
label tensors, so `MERTTrainer` follows `torchtitan.models.flux.trainer`'s
precedent: it overrides `batch_generator` / `forward_backward_step` /
`train_step` directly, while model construction, parallelization,
checkpointing, and (optional) validation still go through the base
`Trainer.__init__` unmodified.

The reference computes classification logits only at the boolean-indexed
masked positions (`x[masked_indices]`). This experiment instead computes
logits (and the CQT prediction) **densely over every frame** and applies the
mask inside the loss (a multiply-then-normalize, not a gather). This keeps
every tensor a fixed shape across steps, which this framework's default
CUDA-graph and `torch.compile` code paths require; the loss is otherwise
identical since only masked positions are averaged into it.

### Multi-codebook classification head, reframed

The reference computes, per codebook, cosine similarity between the
projected encoder output and a positive target embedding plus **every** row
of that codebook's `nn.Parameter` embedding table as negatives (not a
sampled subset), then a cross-entropy against index 0. Because the full
table is always used as negatives, this is mathematically identical to
`logits = cosine_similarity(proj_x, codebook_table) / logit_temp` followed by
`cross_entropy(logits, target_index)` -- an ordinary cosine-similarity
classifier -- which is what `MERTModel.forward` computes directly, without
the reference's separate "positive + all-negatives" construction.

## Known deviations from the reference training code

- **No RVQ-VAE/k-means tokenizer.** See "Scope" above.
- **Fixed-length crops.** The reference (`pad_audio=False, random_crop=True`)
  crops each *batch* to that batch's shortest sample. This experiment always
  crops to one fixed length (derived from `training.max_context_length` via
  `model.required_input_length`, exactly matching the CNN's true
  input/output length formula -- see `model.conv_output_length`'s
  docstring), so every batch has a fixed shape.
- **No in-batch noise-mixing augmentation** (`mixture_prob`, used by the 95M
  config). This is a data augmentation policy, not core architecture; it
  could be added to `dataset.py`'s collation later.
- **No channel masking** (`mask_channel_prob`). Both released configs leave
  it at the default `0.0` (disabled), so there is no behavior difference for
  95M/330M; it's simply not implemented.
- **No `target_glu`, `wav_normalize`, mel-spectrogram reconstruction, or
  `emb_grad_mult`/`attention_relax`/`deepnorm`/`subln`.** Both released
  configs leave these at their disabled defaults.
- **No weight normalization on the positional conv.** The reference applies
  `weight_norm`; this is omitted since it interacts awkwardly with
  meta-device initialization and FSDP2's DTensor parameters, and is not
  central to the pretraining objective.
- **Own Constant-Q Transform**, not `nnAudio`. Same standard kernel-based
  algorithm, but not bit-exact (see `cqt.py`'s module docstring) -- this
  drops an otherwise-unmaintained third-party dependency.
- **Plain per-rank mean loss**, not the fairseq criterion's sum-then-global-
  `sample_size` normalization. With FSDP/DDP's default gradient averaging
  and this experiment's fixed per-rank batch shape, a mean reduction is the
  more portable choice for this framework and needs no cross-rank valid-
  token bookkeeping (see `loss.py`'s docstring).
- **No tensor, context, or pipeline parallelism**, and no gradient
  accumulation (`MERTTrainer` raises if configured) -- FSDP2 (optionally
  HSDP via `dp_replicate`) only, matching the scale of the released configs
  (an 8-A100 node for 95M).
- **Weight init:** attention q/k/v/out-proj use fairseq's
  `MultiheadAttention.reset_parameters` (Xavier-uniform, scaled q/k/v gain);
  every other linear/conv/norm layer uses plain PyTorch `reset_parameters()`
  defaults (the reference never overrides them either). `mask_embedding_MD`
  and the codebook embedding tables use `nn.init.uniform_` (matching the
  reference's `mask_emb` / `label_embs_concat`).

## Usage

Quick correctness check on synthetic data (no external assets needed):

```bash
NGPU=1 MODULE=mert CONFIG=mert_debugmodel ./run_train.sh
```

`mert_95m` and `mert_330m` reproduce the hyperparameters of
`ReferenceRepos/MERT/mert_fairseq/config/pretrain/MERT_RVQ-VAE_CQT_{95M,330M}.yaml`
(see `config_registry.py`), but also default to synthetic data:

```bash
NGPU=8 MODULE=mert CONFIG=mert_95m ./run_train.sh
```

### Training on real data

1. Prepare an audio manifest and per-codebook label files in the same
   layout as the reference's `MERTDataset` (see `dataset.py`'s module
   docstring): a manifest `.tsv` (root dir on line 1, then
   `<relative_path>\t<num_samples>` per line) and one label file per
   codebook (whitespace-separated integer codes per line, one line per
   manifest entry, at `label_rate`). See
   `ReferenceRepos/MERT/scripts/prepare_codecs_from_manifest.py` (RVQ-VAE
   codes) or `simple_kmeans` (k-means codes) to produce these from raw
   audio.
2. Point the dataloader at them, e.g.:

   ```bash
   NGPU=8 MODULE=mert CONFIG=mert_95m ./run_train.sh \
     --dataloader.manifest-path /path/to/manifest.tsv \
     --dataloader.label-paths /path/to/train.codec_0 ... /path/to/train.codec_7
   ```

## Tests

```bash
pytest torchtitan/experiments/mert/tests/test_mert.py -v
```

Covers: span masking shapes, the Constant-Q Transform, a full forward pass
through the (meta-device-constructed, then materialized and initialized)
debug model, a forward+backward pass verifying every parameter receives a
gradient, and the synthetic dataloader's batch shapes.
