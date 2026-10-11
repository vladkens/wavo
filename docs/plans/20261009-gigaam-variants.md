# GigaAM v3 variants

Done: 2026-10-09, commit 7720de8

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

## Result

M2, measured at the end of phase 5, two rounds; wavo / `transcribe-bench` on Metal, ms (MiB for
the peak). Warm 1–9% faster than the reference, the CTC models most; a cold first read of a new
GGUF took wavo 0.19–0.47 s to load.

| Model | Clip | Warm median | First call | Load | Peak footprint |
|---|---|---|---|---|---|
| e2e-ctc | ru | 39.9–40.3 / 41.8 | 43.2–43.5 / 46.9–59.4 | 74–78 / 123–136 | 289–290 / 313–314 |
| e2e-ctc | ru-short | 85.5 / 92.6–92.7 | 88.7–89.0 / 95.0–96.3 | 75 / 121–130 | 295 / 315 |
| e2e-ctc | ru-long | 270.0–270.3 / 295.5–296.1 | 275.3–276.2 / 303.9–312.0 | 74–77 / 122–131 | 314–316 / 323–324 |
| rnnt | ru | 42.6 / 42.7–43.3 | 45.7–46.1 / 46.3–54.6 | 74–75 / 123–127 | 291 / 318–319 |
| rnnt | ru-short | 96.1 / 97.1–97.7 | 99.0–99.4 / 99.0–100.1 | 76 / 121–123 | 298–299 / 321–322 |
| rnnt | ru-long | 296.2–296.5 / 305.7–306.3 | 299.8–301.0 / 308.8–322.1 | 76 / 123–126 | 320 / 329–330 |
| ctc | ru | 39.7–39.8 / 41.3–41.4 | 43.0–43.3 / 44.3–60.3 | 73–75 / 121–126 | 287–289 / 312–314 |
| ctc | ru-short | 85.0 / 91.5–91.6 | 89.0–89.9 / 93.0–95.1 | 74–75 / 125–132 | 293 / 314 |
| ctc | ru-long | 269.0–269.2 / 290.2–292.5 | 273.5–275.8 / 289.5–301.1 | 75–79 / 123–129 | 314 / 322 |
