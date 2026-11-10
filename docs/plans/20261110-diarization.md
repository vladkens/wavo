# Speaker diarization

Who spoke when, and speaker-attributed transcripts, approved by the person on 2026-11-10. The
model is NVIDIA's streaming Sortformer 4spk v2.1, the diarizer transcribe.cpp ports at
`REFERENCE_REV` (`src/arch/sortformer/`, `scripts/convert-sortformer.py`,
`docs/models/diar_streaming_sortformer_4spk-v2.1.md`), published by `handy-computer` as
`diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf` (139 MB) under the NVIDIA Open Model License
(commercial use and redistribution allowed with the licence and a notice). Nemotron 3 Diarization
(8 speakers, OpenMDW) is another architecture (a 31-layer RoPE transformer) that transcribe.cpp
doesn't port: out of scope until it has a reference.

## Model

- A FastConformer (17 layers, d 512, 8× conv subsampling of 128 mels, relative positions, biases,
  BatchNorm conv, x·√d before the blocks), a projection to 192, 18 post-LN transformer layers
  (8 heads, ReLU FFN 768, no positions), then ReLU, fc1, ReLU and a 4-way sigmoid: per 80 ms
  frame the probability that each of 4 speakers talks, in order of first appearance.
- Frontend: Parakeet's log-mel without normalization, ⌈n / 160⌉ frames, those from ⌊n / 160⌋ on
  zero.
- Streaming (`stream.cpp`): chunks of the GGUF's `chunk_len` (188 frames, 15 s) with one frame
  of context each side; per chunk the subsampling runs on the chunk's mel window, the rest over
  `[speaker cache | FIFO | chunk]`, and the speaker cache (188 frames) is compressed to the frames
  that best represent each speaker (AOSC). Audio of any length, in linear time.
- Turns: per speaker, each run of frames with probability > 0.5.

## API (for the person's review in the PR)

- `wavo::Diarizer::load(path)`, `diarize(&pcm) -> Vec<Turn>`, `speakers()`, `device()`.
- `wavo::Turn { speaker, start_ms, end_ms }`, sorted by start.
- `Transcript::utterances(&turns) -> Vec<Utterance { speaker, start_ms, end_ms, text, tokens }>`:
  each word goes to the speaker whose turns overlap it most, consecutive words of a speaker join.
- `hub::MODELS` gains `sortformer`; `ModelInfo::diarization()` tells it apart.

## CLI

- `wavo run [MODEL] AUDIO --diarize`: text as `Speaker 1: …` lines, `--json` with a speaker per
  token and the utterances; the diarizer is `sortformer`, downloaded as `run` downloads a model.
- `--srt` with `--diarize`: speakers as cue boundaries, not labels (as Subtitle Edit does with
  Sortformer): no cue spans two speakers, and two short lines of different speakers back to back
  share a cue as a dialogue, `- line one` / `- line two`. Labels in cues would be opt-in.
- `wavo run sortformer AUDIO`: the turns alone.
- `wavo serve` stays out (OpenAI's `diarized_json` is a follow-up).

## Checklist

- [x] `sortformer` in `hub::MODELS`, the model table, `ALL_MODELS` and `tests/models.rs`
- [x] `scripts/fixtures.sh` keeps the reference's speaker segments; fixtures for
      `sortformer-2spk-mix`, `multitalker-2spk-mix`, `jfk`, `dots`, `dots-full` and `noise`
- [x] `src/sortformer/`: frontend, encoder (the Parakeet subsampling and the shared conformer with
      biases), transformer and head on the GPU, the streaming state on the CPU
- [x] `Diarizer`, `Turn`, `Transcript::utterances`, rustdoc with units and errors
- [x] Fixtures pass on the M2: exact up to 35 s; `dots-full` (306 s) within 1% of its speech
      frames, as transcribe.cpp's own CPU and Metal backends differ there
- [x] CLI: `--diarize`, the turns of `wavo run sortformer`
- [x] Speed against transcribe.cpp on the M2
- [x] `docs/roadmap.md`, `readme.md`
