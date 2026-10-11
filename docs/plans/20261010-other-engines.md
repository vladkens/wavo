# Other engines

After Whisper lands. On the M2, the same audio, each engine in wavo's setup (greedy, the same
language) and with its own defaults: warm median, first call, load (cold and cached), peak memory
and, where possible, power (`powermetrics` or `macmon`). Numbers and conclusions go to Result
below; the readme compares only with transcribe.cpp.

- [x] Whisper large-v3-turbo: whisper.cpp (Metal).
- [x] Whisper large-v3-turbo: WhisperKit (Core ML, Neural Engine).
- [x] Parakeet TDT v3: FluidAudio (Core ML, Neural Engine), parakeet-mlx (MLX).
- [ ] GigaAM v3 and Parakeet: sherpa-onnx (ONNX Runtime, CPU / Core ML).
- [ ] Clips under 1 s: what each engine returns, and wavo too.
- [x] Neural Engine for wavo: whether it can be reached from Rust (Core ML through `objc2`) and
      what that would cost (model conversion, F16, fixed shapes) against the speed and power gain.

## Result

Whisper large-v3-turbo against whisper.cpp v1.9.5 (d1be6fd, Metal, its 874 MB
`ggml-large-v3-turbo-q8_0.bin`), M2, two rounds alternating; wavo `bench -n 10` / `whisper-cli
-bs 1` (greedy) with the language given, ms (MiB for the peak). The last column is `-l auto`, which
wavo always does.

| Clip | Warm median | First call | Load | Peak footprint | whisper.cpp `-l auto` |
|---|---|---|---|---|---|
| jfk, 11 s | 1244–1281 / 1242–1260 | 1243–1284 / 1242–1260 | 208–277 / 268–287 | 1017 / 1038–1039 | 2357–2448 |
| german, 29 s | 1319–1321 / 1317–1321 | 1322–1323 / 1317–1321 | 216–224 / 265–269 | 1019 / 1036–1038 | 2432–2447 |
| ru-long, 34 s | 2696 / 2674–2710 | 2693–2695 / 2674–2710 | 212–214 / 264–276 | 1019–1021 / 1048–1050 | 3783–3869 |
| dots-full, 306 s | 16583 / 20240–20633 | 16567–16637 / 20240–20633 | 220–226 / 268–279 | 1072 / 1133–1134 | 21512–21740 |

- Speed is a tie on one or two windows (1.11 s per encoder window, 3.0 ms per token). With `-l
  auto` whisper.cpp runs one extra encoder pass per file; wavo reads the language from the first
  window's logits. On dots-full wavo is 18% faster: not investigated.
- Text: the same words on all four. dots-full (808 words) differs in 8 punctuation marks at window
  joins (whisper.cpp feeds the previous text as a prompt, wavo doesn't). Beam 5: the same text,
  +2–13% time, +70 MiB.
- `whisper-cli` reads wav, mp3, flac and ogg without ffmpeg; m4a needs a build against ffmpeg.
  Setup: cmake configure 11 s and build 28 s, 8.2 MB in 7 files, its Metal library compiled on
  the first run in 5.8 s. wavo: a clean `cargo build --release` in 41 s, one 13.2 MB binary.

Neural Engine engines, M2, 2026-10-11: WhisperKit 9f99b5c and FluidAudio e560c64 (built
from source, release), driven by a small Swift harness over their libraries with the same 16 kHz
samples wavo reads; parakeet-mlx 0.5.3 (bf16, its CLI's 120 s chunks). Two or three rounds
alternating, 10 warm calls each, ms; peak is the process's peak footprint in MiB. WhisperKit runs
its M2 default `openai_whisper-large-v3-v20240930` (F16 weights, 1.6 GB), greedy, language
given, no temperature fallback, sequential 30 s windows; its default compute units already put
both the encoder and the decoder on the ANE (`.cpuAndNeuralEngine`), the mel on the GPU.
FluidAudio runs `parakeet-tdt-0.6b-v3-coreml` (encoder 6-bit palettized, 15 s windows with
overlap); its default is the ANE for encoder, decoder and joint, `gpu` moves the encoder only.

| Whisper turbo, warm median | wavo | WhisperKit ANE | WhisperKit GPU |
|---|---|---|---|
| jfk, 11 s | 1250–1322 | 1778–2001 | 1929–1985 |
| german, 29 s | 1313–1322 | 2027–2052 | 2458–2530 |
| ru-long, 34 s | 2676–2758 | 4280–4380 | 5406–5662 |
| dots-full, 306 s | 16641–17065 | 29687–29748 | 42045–43469 |
| first call (jfk) | 1284–1345 | 1939–4000 | 2397–2449 |
| load, cached | 327–427 | 464–824 | 772–4953 |
| load, first ever | – | 6868 | 5187 |
| peak (jfk / dots-full) | 1007 / 1057–1061 | 197–204 / 220–225 | 344–351 / 877–878 |

| Parakeet V3, warm median | wavo | FluidAudio ANE | FluidAudio GPU | parakeet-mlx |
|---|---|---|---|---|
| jfk, 11 s | 145–146 | 62–65 | 134–135 | 259–266 |
| dots, 35 s | 435–439 | 193–194 | 371–375 | 792–794 |
| ru-long, 34 s | 408–412 | 186–188 | 368–375 | 705 |
| dots-full, 306 s | 6712–6972 (one pass) | 1462–1535 | 3209–3219 | 7792–7831 |
| first call (jfk) | 166–169 | 75–80 | 420–434 | 275–2519 |
| load, cached | 86–315 | 111–173 | 241–1821 | 1187–1841 |
| load, first ever | – | 16408 | 6614 | – |
| peak (jfk / dots-full) | 762–764 / 1143–1145 | 38 / 108–122 | 148–149 / 763–765 | 2229–2258 / 5691–5787 |

Energy: `macmon pipe -i 200` while the bench loops on one clip for 30–55 s, CPU + GPU + ANE power
averaged over the warm calls, the idle level measured just before (≤ 1.5 W, runs on a busy machine
discarded) subtracted, times the median call; two to four clean rounds. W are the busy averages.

| Clip | Engine | CPU / GPU / ANE W | J per transcription |
|---|---|---|---|
| Parakeet dots, 35 s | wavo | 4.3–6.6 / 8.4–8.6 / 0 | 5.4–6.3 |
| | FluidAudio ANE | 5.8 / 0.1 / 3.4 | 1.7–1.8 |
| | FluidAudio GPU | 4.2–4.4 / 7.7–7.9 / 0 | 4.4–4.5 |
| Parakeet dots-full, 306 s | wavo (one pass) | 2.5–2.9 / 8.0–8.5 / 0 | 68–75 |
| | FluidAudio ANE | 5.1–5.3 / 0.1 / 4.2–4.4 | 13.5–13.7 |
| | FluidAudio GPU | 3.9–4.6 / 9.1–9.4 / 0 | 41–44 |
| Whisper ru-long, 34 s | wavo | 0.7–1.8 / 11.7–12.0 / 0 | 32–35 |
| | WhisperKit ANE | 7.3–8.0 / 0.2 / 1.7–1.9 | 38–41 |
| | WhisperKit GPU | 1.0–1.7 / 9.7–10.2 / 0 | 58–62 |

One Core ML model at a time on zero inputs (`MLModel` from Swift, median of 20, ms): Parakeet
encoder per 15 s window ANE 31, GPU 103 (wavo's whole call on the 15 s `noise.wav`: 168); its
decoder and joint steps 0.19 and 0.23 on either. Whisper encoder per 30 s window ANE 1434 (the
ANE-reshaped `_turbo` variant 1557–1634), GPU 1147, wavo ~1110; decoder per token ANE 11.5, GPU
25, wavo 3.0. First compile on the ANE: Parakeet encoder 16.6 s, Whisper encoder 4.1 s and
decoder 2.1 s.

- Parakeet on the ANE is the one clear win: FluidAudio is 2.2–2.3× faster than wavo on 11–35 s
  (1.5× on the 4.5 s `ru`, 46 against 71 ms, though padded to 15 s), takes a third of the energy
  per transcription, and its first transcript after a cached load comes in 190–250 ms against
  wavo's 250–480. The ANE runs Parakeet's encoder at ~7 TFLOP/s against ~1.7 for wavo's GPU
  kernels. On 306 s wavo's one pass is quadratic; `wavo run` in 60 s segments takes 4.1 s with
  load, still 2.7× FluidAudio's 15 s windows.
- Whisper on the ANE is no win on the M2: WhisperKit is 1.4–1.8× slower than wavo and uses
  ~15% more energy. The ANE runs Whisper's encoder (1280 wide, 1500 frames) at ~1.6 TFLOP/s,
  slower than the GPU, and its decoder step is 4× wavo's; the CPU draws 7–8 W driving it.
- Both Core ML engines keep a CPU busy at 5–8 W around the ANE's 2–4 W, so the ANE's saving is
  only as large as the speedup. The process footprint leaves out what Core ML maps from the model
  files and the ANE's own memory (RSS 1.5–2.6 GB for WhisperKit), so peaks don't compare.
- Load: the first Core ML load compiles for the ANE (6.9 s Whisper, 16.4 s Parakeet); the OS
  caches the result by model path and configuration, so a moved or updated model compiles again.
  Cached loads are 0.1–0.8 s, but WhisperKit's first ANE call still costs up to 2 s more than a
  warm one. WhisperKit's GPU load stays at 0.8–5 s in every process.
- Text: the same words on jfk, german, dots and ru-long for every engine (normalized,
  punctuation aside); ru-long matches the FLEURS reference exactly for all. On dots-full
  WhisperKit differs in 2–4 words from wavo, FluidAudio in 3 (`wanna`, a doubled `roadad` at a
  window join), wavo's own 60 s split in 3; parakeet-mlx drops two whole sentences at its chunk
  joins (102 edits).
- parakeet-mlx is 1.7–1.8× slower than wavo on 11–35 s and 1.1–1.2× on dots-full (its 120 s
  chunks against wavo's one pass), loads in 1.2–1.8 s and peaks at 2.2–5.8 GB.
- Setup: WhisperKit and FluidAudio build with `swift build` in 1–2.5 min (FluidAudio also fetches
  an 87 MB prebuilt text-normalization framework); models 1.6 GB (Whisper) and 0.47 GB (Parakeet)
  of `.mlmodelc` directories from Hugging Face.

Neural Engine for wavo: feasible, worth it only for the Conformer families (Parakeet, then
GigaAM), as an opt-in macOS backend whose output is approximate; not for Whisper.

- Reach: `objc2-core-ml` 0.3.2 binds `MLModel`, `MLMultiArray` (from our buffers), compute units,
  `MLModel.compileModelAtURL` and `MLComputePlan`; it is the same `objc2` 0.6 / 0.3 family wgpu
  already builds, so the build stays pure Rust and grows little.
- Model: Core ML takes only its own format, so an offline conversion has to publish `.mlmodelc`
  (or `.mlpackage`, compiled on device at first load): a `scripts/` coremltools step from the GGUF
  weights or the NeMo checkpoint, run with `uv` and never in the build. Reusing FluidInference's
  CC-BY-4.0 encoder is the shortest path; a pure-Rust MIL writer (as `mil_spec` does) avoids
  Python but is ~1–1.5k lines more. Q8_0 maps onto Core ML's blockwise int8 weights
  (`constexpr_blockwise_shift_scale`, macOS 15).
- Precision: the ANE computes in F16, the published encoder is 6-bit palettized, so the fixtures
  can't stay exact; the backend needs agreement tests against the GPU path instead.
- Shapes: the ANE wants fixed shapes. One 15 s window means padding short clips (31 ms of
  encoder each) and windows with overlap and token merging past 15 s, the part where FluidAudio
  and parakeet-mlx both slip. Enumerated shapes (e.g. 2, 5, 10, 15 s) cut the padding, each
  compiled on first use.
- Stays on the CPU/GPU: the mel and the TDT/RNN-T decoder loops (wavo's NEON decoder takes the
  encoder output; the joint's encoder projection moves to the CPU or into the Core ML graph).
- Cost: ~400–600 lines of Rust (Core ML wrapper, windowing, a directory model in `hub`, a CLI
  switch) plus a ~300–500-line conversion script, about one to two weeks of agent work with
  validation; a one-time 16 s compile per model, best done by `pull`. Gain for Parakeet: 2.2–2.3×
  warm, ~3× less energy per transcription, a first transcript in 190–250 instead of 250–480 ms.
