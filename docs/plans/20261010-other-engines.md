# Other engines

After Whisper lands. On the M2, the same audio, each engine with its defaults: warm median, first
call, load (cold and cached), peak memory and, where possible, power (`powermetrics` or `macmon`).
Conclusions go to `docs/perf.md` and the readme's Speed section.

- [x] Whisper large-v3-turbo: whisper.cpp (Metal).
- [ ] Whisper large-v3-turbo: WhisperKit (Core ML, Neural Engine).
- [ ] Parakeet TDT v3: FluidAudio (Core ML, Neural Engine), parakeet-mlx (MLX).
- [ ] GigaAM v3 and Parakeet: sherpa-onnx (ONNX Runtime, CPU / Core ML).
- [ ] Clips under 1 s: what each engine returns, and wavo too.
- [ ] Neural Engine for wavo: whether it can be reached from Rust (Core ML through `objc2`) and
      what that would cost (model conversion, F16, fixed shapes) against the speed and power gain.
