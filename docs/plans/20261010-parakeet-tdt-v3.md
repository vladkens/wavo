# Parakeet TDT V3

Done: 2026-10-10, commit 31bfebf

Roadmap phase 6: `parakeet-tdt-0.6b-v3` (25 European languages) as a variant of the Parakeet
family. Follow `agents.md` → "Adding a model"; read `docs/perf.md` before speed work.

## Goal

`make test` reproduces the reference fixtures `jfk`, `ru-short` and `uk-short` exactly (text,
token pieces, start ms); V2 and GigaAM fixtures stay exact. Against `transcribe-bench` on the same
Q8_0 GGUF, warm, first call, load and peak footprint at least on par, faster where possible. The
variant adds config, not code: core + `src/parakeet/` stays ~3.1k lines.

## Model facts that differ from V2

Sources in `3rd/transcribe.cpp`: `catalog/parakeet-tdt-0.6b-v3.json`
(`handy-computer/parakeet-tdt-0.6b-v3-gguf`, `parakeet-tdt-0.6b-v3-Q8_0.gguf`, 697 tensors),
`scripts/convert-parakeet.py` (profile `parakeet-tdt-0.6b-v3`), `docs/models/parakeet*.md`,
`src/arch/parakeet/{model,decoder,weights}.cpp`, `src/transcribe-tokenizer.cpp`.

- Encoder, frontend and TDT metadata are identical to V2 (same keys and values, same tensor
  shapes and types). New keys: `stt.capability.lang_detect` true, `general.languages` 25 codes,
  `stt.variant` `tdt-0.6b-v3`. No prompt dictionary, `kestrel` length masking or global symbol
  budget (`weights.cpp:515`, `:526`): those are `parakeet-ultra`'s.
- Vocabulary 8193 pieces, blank 8192 (`tokenizer.ggml.blank_token_id`,
  `stt.parakeet.predictor.vocab`); `pred.embed` [640, 8193] and `joint.out` [640, 8198] + bias
  [8198], Q8_0 / F32 as in V2.
- Ids 1–273 are NeMo prompt pieces (`<|nospeech|>`, `<pad>`, `<|en|>`, `<|spk0|>`, ...) typed
  NORMAL (1); only `<unk>` (0, UNKNOWN) and `<blank>` (CONTROL) are special. The reference strips
  CONTROL, UNKNOWN, `<unk>` and `<ll-RR>` locale tags (`model.cpp:654–697`,
  `transcribe-tokenizer.cpp:114`), so a prompt piece would print as-is, as V2's filter does. No
  `<0xHH>` byte-fallback pieces (`decode_sentencepiece`, `transcribe-tokenizer.cpp:157`), so text
  is V2's: pieces with `▁` → space, runs of spaces collapsed, trimmed. The language hint does not
  change the output (`docs/models/parakeet.md`).
- Joint output 8198 rows: 21 MB of F32 weights per decision (V2 2.6 MB), and the embedding table
  21 MB. The reference keeps both F32 and runs the joint as a ggml CPU graph on up to 8 threads
  (`decoder.cpp:116`, `:1018`), plus a softmax over 8193 tokens per emission for `p=`.
- V3's pre-encode amplifies the fp64 STFT gap ~3.5× more than V2's
  (`docs/models/parakeet-tdt-0.6b-v3.md` → "Numerical Validation"); it only matters if a fixture
  mismatches.

## Tasks

### Task 1: model, fixtures, correctness

- [x] `parakeet-tdt-0.6b-v3` is already in `MODELS`; `make models`.
- [x] `make fixtures MODEL=parakeet-tdt-0.6b-v3 SAMPLES="jfk ru-short uk-short"`.
- [x] `tests/models.rs`: `parakeet_tdt_v3` over the three clips.
- [x] V3 fixtures exact on the V2 code path; V2 and GigaAM fixtures exact; `make check`;
      `make test`.

### Task 2: decoder

- [x] Profile the decoder on the three clips (direct timer: LSTM steps, joint, decisions).
- [x] Joint weight format and threads by measurement, bit-exact: F32 as now, Q8_0 or F16
      dequantized in registers (fits in L2), rows split over a scoped second thread. Embedding
      rows read from Q8_0 if it saves memory without cost.

### Task 3: bench and docs

- [x] Benchmark against `transcribe-bench` with the phase-5 protocol (alternating, two rounds;
      warm median and min, first call, load, peak footprint).
- [x] `docs/perf.md`: V3 rows in "Benchmark", "Current state" note, Log entries.
- [x] `readme.md`: model table, "Works now", speed table rows. Tick roadmap phase 6.
- [x] Check: `make check`; `make test`.

## Result

M2, end of phase 6 (Q8_0 decoder weights), two rounds; wavo / `transcribe-bench` on Metal, ms
(MiB for the peak). Decoder alone V3 21 / 31–36 / 29–31 ms, from 76 / 68 / 70 in F32.

| Model | Clip | Warm median | First call | Load | Peak footprint |
|---|---|---|---|---|---|
| V2 | jfk | 142.2–142.9 / 198.1–200.3 | 147.1–150.0 / 204.7–259.0 | 158–159 / 272–276 | 740 / 820 |
| V2 | dots | 459.7–461.5 / 691.8–706.9 | 467.8–483.6 / 675.3–857.7 | 158–175 / 270–280 | 766–768 / 833–834 |
| V2 | jobs-silence | 83.4 / 95.4–98.9 | 87.3–88.0 / 115.7–115.8 | 168–181 / 274–285 | 734–736 / 816 |
| V3 | jfk | 153.0–155.2 / 212.7–218.2 | 155.9–161.5 / 277.9–284.2 | 167–170 / 280–287 | 762–764 / 883–884 |
| V3 | ru-short | 159.7–162.3 / 238.5–253.0 | 172.9 / 321.6–376.6 | 165–170 / 280–290 | 762–763 / 884 |
| V3 | uk-short | 159.1–175.6 / 232.6–275.3 | 169.4–170.6 / 273.6–333.8 | 184–192 / 281–354 | 761–762 / 884 |

Warm is 14–34% faster for V2 and 28–35% for V3 (uk-short's 175.6 was a background burst for both
engines), load 35–45% below, peak 66–82 MiB (V2) and ~120 MiB (V3) below. The reference's decoder
runs on CPU threads and gains most from a quiet machine (dots 671–690 ms here vs 799–821 at load
average 3–5).
