# Long audio

Roadmap phase 9: `wavo run` transcribes recordings of any length by splitting them at pauses into
segments each model handles well. Follow `agents.md`; log every measurement in `docs/perf.md`.

## Goal

- `wavo run` works on any length that fits in memory (16 kHz f32, ~230 MB per hour) with every
  model. GigaAM keeps its words past one minute: 10 minutes of `ru-long` on repeat give ~923
  words, not 177. Parakeet keeps its one-pass text, apart from differences next to the cuts.
- Time grows linearly with length and peak footprint stays flat (the longest segment's plus the
  decoded audio). At 10 minutes both are below the one-pass numbers: 15.9 s / 858 MiB for GigaAM,
  22.5 s / 1516 MiB for Parakeet V3.
- Audio within the segment length goes through exactly one `transcribe` call, as now. `--segment
  0` gives the reference's single pass at any length. The library fixtures and `wavo bench` (one
  call on the whole file, compared with `transcribe-bench`) stay unchanged.
- The CLI stays within ~700 lines without tests (636 now): the splitter and merge take ≤ ~70.

## Reference (transcribe.cpp 5bb2deb)

- `docs/input-limits.md` sorts families into three buckets, and the catalog's
  `long_form_strategy` names the bucket. GigaAM is "soft-window": every call is one pass, and past
  25 s it logs a WARN and proceeds (`src/arch/gigaam/model.cpp:72-83`, `:325-341`), reporting
  `max_audio_ms` = 25 000 (`:116`). Upstream GigaAM rejects audio over 25 s and offers
  `transcribe_longform`: pyannote VAD, chunks merged to ~15–22 s, anything over 30 s split evenly.
  The reference does not port it (`docs/models/gigaam.md:86-90`).
- Parakeet TDT V2/V3 is "chunked-unbounded", which here also means one pass at any length with no
  limit (`docs/models/parakeet.md:87-94`). Only `parakeet-ultra` segments, using its VAD head
  (`needs_longform`, `src/arch/parakeet/model.cpp:1749-1751`). The V2/V3 GGUFs have no
  `stt.parakeet.vad.*` keys.
- parakeet-ultra's segmenter (`src/arch/parakeet/longform.{h,cpp}`, a port of kestrel) treats an
  80 ms frame as speech when the VAD probability is ≥ 0.5, and a pause is ≥ 0.2 s. Over 30 s it cuts
  at the midpoint of the last pause 1–30 s into the segment, else at 30 s. Segments are contiguous
  with no overlap, and segments without speech are skipped (`longform.cpp:84-105`, `:129-189`).
  Each segment is decoded as a short clip; the texts are joined with a space, and times are
  shifted by the segment start and clamped to its end (`model.cpp:1607-1747`).
- `transcribe-cli` has no long-form flags: segmentation is automatic where a model has it. The
  fixtures (default flags, ≤ 35.3 s) are single passes for every model we support.

So the reference has no split output to compare with for our models. Russian is checked against
the fixture text on repeat; Parakeet against its own one-pass text, which matches the reference's.

## Design

- **Library: `Model::max_audio_ms() -> Option<u32>`**, the window a model was trained for, like
  transcribe.cpp's `transcribe_capabilities::max_audio_ms`: `Some(25_000)` for GigaAM (hardcoded
  as in the reference, no GGUF key holds it), `None` for Parakeet. Callers who chunk for themselves
  get the window from the model, and the CLI gets it for any `.gguf` path. The library still
  leaves chunking to the caller.
- **CLI: `src/cli/split.rs`** holds the splitter, the merge and their unit tests; `run` in
  `src/cli/main.rs` calls them.
- **Segment length L**, for every model: `max_audio_ms()`, else the CLI's default for models
  without a window. Parakeet runs coherently to 10 minutes in one pass, but its cost grows past a
  minute (attention is quadratic: 5× the length takes 6.8× the time) and long inputs misbehave in
  other apps. The default is picked from 30 / 60 / 120 s by measurement (time, peak, word diff
  against one pass); 60 s also keeps every Parakeet fixture (≤ 35.3 s) in one pass.
- **`wavo run --segment SECS`** overrides L. `0` means one pass at any length, as the reference
  does; checks against the reference's single pass use it (e.g. `ru-long`, 33.8 s, against its
  fixture). Other values must be at least 5 s. The arguments are still parsed by hand.
- **Splitting**, with no VAD model and no threshold. Energy per 10 ms frame (160 samples, sum of
  x²). While the rest from `s` is longer than L, look in [s + L/2, min(s + L, end − 1 s)] for the
  300 ms window (30 frames) with the least energy (ties go to the later one), and cut at the start
  of that window's quietest frame. The remainder is the last segment.
  - Segments are contiguous slices of the decoded PCM with no copies and no overlap. Each is ≤ L,
    all but the last are ≥ L/2, and the last is ≥ 1 s.
  - The quietest window needs no threshold, so recording level and steady noise don't matter. A
    300 ms window prefers sentence pauses to gaps between words. That matters for the e2e models,
    which punctuate and capitalize each segment as an utterance: a cut mid-sentence can add a
    period and a capital.
  - If the range has no pause (music, run-on speech), the same rule cuts at its quietest 10 ms,
    usually a gap between words. There is no separate hard cut.
  - The intent is the same as parakeet-ultra's rule (long segments, cut inside a pause), but
    without its speech detector. Outputs can be compared by text, not by cut points.
- **Merging.** The one loaded model transcribes the segments in order.
  - `text`: the non-empty segment texts joined with one space.
  - `tokens`: concatenated, with cut / 16 added to `start_ms` (cuts are multiples of 160 samples,
    so the offset is whole ms). A token starts inside its own segment, so times keep increasing.
  - `--json`: same shape. With character vocabularies (`gigaam-v3-rnnt`, `gigaam-v3-ctc`) the
    space between two segments appears only in `text`; no token marks it.
  - `--srt`: `output::srt` also gets the indices of the tokens that start segments and starts a
    word at each one (character vocabularies have no `▁`). `audio_ms` stays the whole file.
  - Plain text: one line, as now.
- **Not in this plan:** VAD models, overlapping windows, skipping silent stretches, streaming the
  input decode or the output, and a `transcribe_long` in the library (worth it once an app built
  on the library needs long files; `split.rs` would move over as is).

## Tasks

### Task 1: plan and rules

- [x] This plan; the user chose `max_audio_ms` in the library, splitting for every model and a
      `--segment` override.
- [x] `agents.md`: the public API gains `max_audio_ms`, `wavo run` gains `--segment`, the CLI
      splits long audio at pauses. Roadmap phase 9.

### Task 2: library

- [x] `Model::max_audio_ms()` with a doc comment.

### Task 3: `wavo run` on long audio

- [ ] `src/cli/split.rs`: `segments(pcm, l) -> Vec<Range<usize>>` and `join(parts)` as designed.
- [ ] `run`: `--segment SECS` parsed by hand; segments and join (one segment within L, so one
      `transcribe` call); `srt` starts a word at each segment start; usage text.
- [ ] Unit tests on synthetic PCM, with tone bursts as speech and digital silence or −40 dB noise
      as pauses: input within L gives one range; cuts fall inside the last pause before L; ranges
      are contiguous and cover the input, each ≤ L, the last ≥ 1 s, cuts multiples of 160; with
      no pause the cuts fall within [L/2, L]; the input × 0.01 gives the same cuts. Join: offsets,
      empty segment texts, segment starts. SRT across a boundary between character tokens gives
      two words.
- [ ] `make check`; `make test`.

### Task 4: verify and measure

- [ ] Long files built with `ffmpeg` outside the repo, as in `docs/perf.md`: `ru-long` on repeat,
      and `dots-full` + `death` + `whole-earth` back to back, as 16 kHz mono PCM16, cut to 1, 2,
      3, 5, 10 and 60 minutes (English repeated for the 60).
- [ ] The default L for models without a window: 30 / 60 / 120 s with `parakeet-v3` on the
      10-minute English file (time, peak, word diff against `--segment 0`).
- [ ] Words, compared in lowercase without punctuation: `gigaam-v3` and `gigaam-v3-rnnt` against
      the fixture text on repeat; `parakeet-v3` and `parakeet-v2` against `--segment 0` up to 10
      minutes. `wavo run --segment 0` on the fixture clips gives the fixture text. Differences
      only next to cuts.
- [ ] `/usr/bin/time -l wavo run` at every length for `gigaam-v3` and `parakeet-v3`: a split table
      next to the one-pass one in `docs/perf.md` → "Long audio", and a log entry.
- [ ] `readme.md`: long audio, each model's L and `--segment`; tick roadmap phase 9; `make check`;
      `make test`.
