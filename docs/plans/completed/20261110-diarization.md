# Speaker diarization

Done: 2026-11-10, PR #100

Who spoke when, and speaker-attributed transcripts, approved by the person on 2026-11-10. The
model is NVIDIA's Nemotron 3 Diarization (up to 8 speakers, OpenMDW 1.1), the diarizer
parakeet.cpp runs (`src/diarization*.cpp`), in its author's GGUF
`mudler/Nemotron-3-Diarization-GGUF` (`nemotron-3-diarization-q8_0.gguf`, 109 MB). parakeet.cpp is
the reference.

## Model

- 8 mel frames stacked and projected to 512, a LayerNorm, 31 pre-LN transformer layers (8 heads,
  fused qkv without bias, rotary positions in GPT-NeoX pairs, GELU FFN 2048), a final LayerNorm;
  the head: a projection to 192, a conv1d (kernel 3) to 8 frames of 192 per frame, ReLU, a hidden
  layer with ReLU and an 8-way sigmoid: per 10 ms frame the probability that each of 8 speakers
  talks, in order of first appearance.
- Frontend: Parakeet's log-mel without normalization, ⌊n / 160⌋ frames.
- Streaming (`diarization_streaming.cpp`): chunks of 264 embedding frames (21.12 s), no context;
  per chunk the stacking runs on the chunk's mel, the rest over `[speaker cache | chunk]`, and the
  speaker cache (264 frames) is compressed to the frames that best represent each speaker (AOSC),
  the model's silence embedding in empty slots. Audio of any length, in linear time.
- Turns: per speaker, each run of frames with probability ≥ 0.5.

## API (for the person's review in the PR)

- `wavo::Diarizer::load(path)`, `diarize(&pcm) -> Vec<Turn>`, `device()`.
- `wavo::Turn { speaker, start_ms, end_ms }`, sorted by start.
- `Transcript::utterances(&turns) -> Vec<Utterance { speaker, start_ms, end_ms, text, tokens }>`:
  each word goes to the speaker whose turns overlap it most, consecutive words of a speaker join.
- `hub::MODELS` gains `nemotron-diarization`; `ModelInfo::diarization()` tells it apart.

## CLI

- `wavo run [MODEL] AUDIO --diarize`: text as `Speaker 1: …` lines, `--json` with the
  utterances; the diarizer is `nemotron-diarization`, downloaded as `run` downloads a model. Not
  with Moonshine, which has no token times.
- `--srt` with `--diarize`: speakers as cue boundaries, not labels (as Subtitle Edit does with
  Sortformer): no cue spans two speakers.
- `wavo run nemotron-diarization AUDIO`: the turns alone, as text or `--json`.
- `wavo serve` stays out (OpenAI's `diarized_json` is a follow-up).

## Checklist

- [x] `nemotron-diarization` in `hub::MODELS`, the model table, `ALL_MODELS` and `tests/models.rs`
- [x] Fixtures from parakeet.cpp's segments for `sortformer-2spk-mix`, `multitalker-2spk-mix`,
      `jfk`, `dots`, `dots-full` and `noise`
- [x] `src/sortformer/`: the transformer (the shared encoder blocks, rotary in GPT-NeoX pairs) and
      the head on the GPU, the streaming state on the CPU
- [x] `Diarizer`, `Turn`, `Transcript::utterances`, rustdoc with units and errors
- [x] Fixtures pass within 1% of their speech frames
- [x] CLI: `--diarize`, the turns of `wavo run nemotron-diarization`
- [x] Speed against parakeet.cpp
- [x] `docs/roadmap.md`, `readme.md`

## Result

- Of the speech frames of parakeet.cpp's turns: `sortformer-2spk-mix` 1 of 874 apart,
  `multitalker-2spk-mix` 1 of 1209, `jfk` 2 of 858, `dots` 1 of 2756, `dots-full` (306 s, 15
  chunks) 42 of 21585 (portable kernels: 20); `noise` none.
- On an M5, the whole CLI run with the load, best of 5, against parakeet.cpp's `diarize` on
  Metal: jfk 0.04 vs 0.08 s, dots 0.08 vs 0.14 s, dots-full 0.52 vs 0.76 s. parakeet.cpp's own
  Metal and CPU turns of dots-full are 10 speech frames apart, wavo's from its Metal ones 34.
- DER, collar 0, against the datasets' own labels: AMI test SDM (16 meetings, 8.9 h, 3–4
  speakers) wavo 28.2%, parakeet.cpp 27.5% on Metal and on the CPU; VoxConverse dev, its first 44
  files (4.2 h, 1–17 speakers) wavo 5.2%, parakeet.cpp 6.1% on Metal and on the CPU. A long
  recording now and then flips a speaker for minutes on one build and not another: AMI's ES2004a
  53.5% on the M5's tensor kernels against 32–35% on the others, VoxConverse's vmaiq (17 speakers)
  28% against 45–46%.
