# Parakeet TDT V3

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
