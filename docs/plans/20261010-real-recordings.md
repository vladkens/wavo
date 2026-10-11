# Real recordings vs transcribe.cpp

Done: `make compare` on 2,923 dictation recordings, the same text as the reference in 99.86%
(GigaAM) and 98.9% (Parakeet V3), PR #6.

Roadmap phase 10: check on a few thousand real dictation recordings, not the 15 fixture clips,
that wavo gives transcribe.cpp's text and is faster. The recordings are private voice notes: only
aggregates go into the repository. Follow `agents.md`; log the measurements in `docs/perf.md`.

## Design

- **`examples/batch.rs`**: `batch MODEL.gguf LIST` loads the model once and transcribes every
  16 kHz mono WAV in LIST with `Model::transcribe` (one pass, as the reference's default), printing
  JSON lines shaped like `transcribe-cli --batch LIST --batch-jsonl`'s (header with `load_ms`,
  then `file`, `text`, `audio_ms`, `ms` per file). `batch compare A.jsonl B.jsonl` prints speed,
  then exact-text agreement and word error rate by duration bucket, then the files that differ.
  Only the library and `hound` (dev-dependency): no new dependencies.
- **Reference**: `transcribe-cli --batch LIST --batch-jsonl`, its built-in batch mode (one model
  load, `transcribe_run` per file); per-file time is its `mel_ms + encode_ms + decode_ms`.
- **`make compare MODEL= LIST= OUT=`** runs both under `/usr/bin/time -l` (process wall time, peak
  footprint), then the comparison.
- **Sample**: stratified by duration, seeded, shuffled; every recording of 60 s or more. Models
  `gigaam-v3` (e2e-rnnt) and `parakeet-v3` on all of it, A/B/A (wavo, reference, wavo) under the
  GPU lock.

## Tasks

- [x] Inventory of the recordings (count, format, duration distribution) and the sample.
- [x] `examples/batch.rs`, `make compare`, a pointer in `agents.md`.
- [x] `gigaam-v3` and `parakeet-v3` A/B/A over the sample.
- [x] Mismatches inspected and categorized; agreement with Handy's stored transcripts.
- [x] `docs/perf.md` "Real recordings" and a Log entry; roadmap phase 10 ticked.

## Result

2,923 of the person's recordings (8.6 h, mostly Russian, median 3.75 s), M2, A/B/A:

- Text: the same as the reference in 99.86% (GigaAM e2e-rnnt) and 98.9% (Parakeet V3) of the
  recordings. Every difference starts at a near-tie (top two logits within 0.03): float
  rounding, not a bug.
- Parakeet V3: 36–37% less compute, faster at every length (V2 checked on 87 recordings).
- GigaAM: on par overall; ~5% slower per file under 2 s, faster from 5 s. Its fixed cost per call
  is ~1 ms higher (`docs/perf.md` → "Not tried yet").
- Model loaded per recording (fresh process): GigaAM 136 vs 188 ms, Parakeet V3 278 vs 429 ms.
- Peak memory over a batch: Parakeet 975 vs 1067 MiB; GigaAM 452 vs 421 MiB, since wavo keeps the
  arena of the longest input so far.
