# Fast build

After the models are in. Keep the fixtures exact; log each attempt in `docs/perf.md`, the numbers
in Result below.

- [x] One FFT module for every size the models use (2, 3 and 5 factors: 320, 400, 512), shared
      by all frontends; drop `rustfft` if GigaAM and Parakeet fixtures stay exact.
- [x] `cargo build --timings` before and after: clean build and library rebuild.
- [ ] wgpu features per target (Vulkan only on Linux) if it shortens the macOS build.
- [ ] What else `--timings` shows on wavo's own crate.

## Result

The own FFT (PR #15), Windows 7945HX, `cargo build --release --timings`: clean build 40.0 → 29.5 s,
library rebuild after touching `src/lib.rs` 11.9 → 2.1 s (the rest is the CLI, whose `rubato`
still pulls `rustfft`), library alone clean 29.8 → 19.8 s.
