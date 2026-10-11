# Other engines

After Whisper lands. On the M2, the same audio, each engine in wavo's setup (greedy, the same
language) and with its own defaults: warm median, first call, load (cold and cached), peak memory
and, where possible, power (`powermetrics` or `macmon`). Numbers and conclusions go to Result
below; the readme compares only with transcribe.cpp.

- [x] Whisper large-v3-turbo: whisper.cpp (Metal).
- [ ] Whisper large-v3-turbo: WhisperKit (Core ML, Neural Engine).
- [ ] Parakeet TDT v3: FluidAudio (Core ML, Neural Engine), parakeet-mlx (MLX).
- [ ] GigaAM v3 and Parakeet: sherpa-onnx (ONNX Runtime, CPU / Core ML).
- [ ] Clips under 1 s: what each engine returns, and wavo too.
- [ ] Neural Engine for wavo: whether it can be reached from Rust (Core ML through `objc2`) and
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
