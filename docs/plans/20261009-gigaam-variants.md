# GigaAM v3 variants

Roadmap phase 3: `gigaam-v3-e2e-ctc`, `gigaam-v3-rnnt` and `gigaam-v3-ctc` on the existing
GigaAM code. Follow `agents.md` → "Adding a model".

## Goal

Each variant reproduces its fixtures exactly on `ru`, `ru-short` and `ru-long`, and its warm,
first call, load and peak footprint are at least on par with `transcribe-bench` on the same
file. A variant is config + head inside `src/gigaam/`, with the encoder shared as is.

## Model facts

Sources: `3rd/transcribe.cpp/scripts/convert-gigaam.py`, `src/arch/gigaam/{weights,decoder,
model}.cpp`, `docs/models/gigaam-v3-*.md`, `catalog/gigaam-v3-*.json`.

- `stt.gigaam.head_kind` selects the head: `rnnt` or `ctc`.
- CTC head: `head.ctc.weight` `[1, 768, C]` + `head.ctc.bias` `[C]`, C =
  `stt.gigaam.head.num_classes` (257 for e2e-ctc, 34 for ctc), blank = C − 1. Greedy: per-frame
  argmax (ties to the lowest id), drop a label equal to the previous frame's, then drop blanks; a
  token's frame is where its run starts.
- e2e variants use SentencePiece pieces (`▁` → space). `rnnt` and `ctc` are charwise
  (`tokenizer.ggml.model` = `char`, 33 symbols + blank); check how the reference turns them into
  text.

## Tasks

- [x] `make fixtures` for the three variants on `ru ru-short ru-long`; `tests/gigaam.rs` covers
      all four models.
- [x] CTC head: logits as the last GEMM on the GPU instead of `joint.enc`, CPU greedy collapse;
      charwise detokenization; RNN-T with the 34-class vocabulary.
- [x] Fixtures exact for all variants; `make check`; `make test`.
- [x] Bench each variant against `transcribe-bench` (warm, first call, load, peak); log in
      `docs/perf.md`; list the variants in `readme.md`; tick roadmap phase 3.
