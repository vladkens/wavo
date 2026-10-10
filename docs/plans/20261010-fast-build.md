# Fast build

After the models are in. Keep the fixtures exact; log the numbers in `docs/perf.md`.

- [ ] One FFT module for every size the models use (2, 3 and 5 factors: 320, 400, 512), shared
      by all frontends; drop `rustfft` if GigaAM and Parakeet fixtures stay exact.
- [ ] `cargo build --timings` before and after: clean build and library rebuild.
- [ ] wgpu features per target (Vulkan only on Linux) if it shortens the macOS build.
- [ ] What else `--timings` shows on wavo's own crate.
