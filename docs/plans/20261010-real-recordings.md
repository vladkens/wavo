# Real recordings vs transcribe.cpp

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
