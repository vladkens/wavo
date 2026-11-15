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
- Streaming (`diarization_streaming.cpp`, NeMo's `streaming_update`): the model card's 30.4 s
  config, chunks of 340 embedding frames (27.2 s) with 40 more ahead; per chunk the stacking runs
  on the chunk's mel and those ahead, the rest over `[speaker cache | FIFO | chunk | ahead]`, the
  FIFO (40 frames) moves 300 frames at a time to the speaker cache (264 frames), which is
  compressed to the frames that best represent each speaker (AOSC), the model's silence embedding
  in empty slots. Audio of any length, in linear time.
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
- [x] CLI: `--diarize`, the turns of `wavo run nemotron-diarization`, `--device`
- [x] The diarizer on the CPU (`--backend cpu`, `Backend::Auto` without a hardware GPU)
- [x] Speed against parakeet.cpp
- [x] `docs/roadmap.md`, `readme.md`

## Result

- Of the speech frames of parakeet.cpp's turns: `sortformer-2spk-mix` 1 of 874 apart,
  `multitalker-2spk-mix` 1 of 1209, `jfk` 2 of 858, `dots` 1 of 2756, `dots-full` (306 s, 15
  chunks) 42 of 21585 (portable kernels: 20); `noise` none. With the model card's chunking,
  `dots` 25 and `dots-full` 214 apart; on the CPU 25 and 218, on `matrix8` 217, at the edge of
  the 1%. The fixtures became wavo's own turns on the M5's GPU in parakeet.cpp's format: 0 apart
  on every tier and the CPU, 3 on `matrix8`.
- The CPU on an M5 (10 threads): `dots-full` in 6.2–9.4 s, against 0.6 s on its GPU.
- AMI test Mix-Headset on a MacBook Air M5 (`scripts/diarization.py`, the table in
  `docs/benchmarks.md`): 9.2% at 581× on the GPU, 9.2% at 38× on the CPU; FluidAudio's Nemotron
  9.5% at 499×, pyannote community-1 34.7% at 13× (17.1% against the manual labels, wavo 26.0%).
- On an M5, the whole CLI run with the load, best of 5, against parakeet.cpp's `diarize` on
  Metal: jfk 0.04 vs 0.08 s, dots 0.08 vs 0.14 s, dots-full 0.52 vs 0.76 s. parakeet.cpp's own
  Metal and CPU turns of dots-full are 10 speech frames apart, wavo's from its Metal ones 34.
- The same against parakeet.cpp on Vulkan: RTX 4060 jfk 0.27 vs 0.29 s, dots 0.30 vs 0.33 s,
  dots-full 0.65 vs 0.78 s; Radeon 610M 0.41 vs 0.40 s, 0.89 vs 0.97 s, 7.3 vs 8.5 s; Intel N100
  (Linux) 0.52 vs 0.95 s, 1.9 vs 3.5 s, 20.1 vs 37.6 s. The fixtures' frames apart are the M5's
  on the M2, the Radeon 610M (Vulkan and DX12), the RTX 4060 (DX12 too) and the N100, on each
  kernel tier of the last two.
- DER, collar 0, on an M5. Against the datasets' own labels: AMI test SDM (16 meetings, 8.9 h,
  3–4 speakers) wavo 27.7%, parakeet.cpp 27.5%; VoxConverse dev (216 files, 20.1 h, 1–20 speakers)
  5.7% and 6.2%, test (232 files, 43.3 h, 1–21) 8.5% and 8.3%. Against the forced-alignment labels
  of the model card (nttcslab-sp/diar-forced-alignment), AMI test Mix-Headset 9.2% (the card:
  9.25%), SDM 11.7%; FluidAudio's CoreML port with the same chunking 9.5% and 11.4%, VoxConverse
  test 8.2%. parakeet.cpp's 21.12 s chunks with nothing ahead gave wavo 28.2%, 6.0%, 8.4%, 9.3%
  and 12.3%. The chunking alone moves a long recording's DER by up to 27 points either way, as a
  speaker flips for minutes: AMI's ES2004a (SDM) 43.4% with the old chunks, 16.7% with the new.
- Words given to the wrong speaker (WDER, diarizationlm) by `--diarize` with parakeet-v3 against
  AMI's word-level labels, its 16 test meetings: 6.06%, 5.63% once a word's span ends 200 ms after
  its last token starts rather than at the next word (better on 15 meetings, the same on 1), 5.55%
  with the new chunks.
