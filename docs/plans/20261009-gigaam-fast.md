# GigaAM v3 e2e-rnnt, fast

Done: warm faster than the reference on all three clips, load and peak below it, commit 7b02350
(before pull requests).

Roadmap phase 2. Follow `agents.md`; read `docs/perf.md` first and log every attempt there.

## Goal

On the M2, against `transcribe-bench` on the same GGUF and WAV, measured A/B/A back to back:

- Warm median faster than the reference on `ru`, `ru-short` and `ru-long`, by more than the noise
  between A runs.
- First call (fresh process) and peak footprint (`/usr/bin/time -l`) no worse than the reference.
- Load stays ≤ 0.25 s.
- Fixtures stay exact. If a change alters them, stop and show the user the diff.

Baseline (phase 1): warm 45 / 102 / 340 ms vs 44 / 103 / 315 ms, first call 71–84 / 131–135 /
372–388 ms vs 59–70 / 103–118 / 319–331 ms, peak 0.88–0.93 GB vs 0.33 GB, load 70 ms vs 130 ms.

## Tasks

### Task 1: measure precisely

- [x] GPU timestamps per dispatch kind (GEMM by layer type, attention, row kernels) and CPU
      timers for mel, upload, readback and decoder, as a temporary local tool that is not kept.
      Record the ru-long and ru profiles in `docs/perf.md`; the order of the tasks below follows
      from them.

### Task 2: memory and first call

- [x] Peak footprint ≤ the reference: no whole-file read, Q8_0 repacked straight into mapped
      upload memory, no extra staging copy on unified memory.
- [x] First call ≤ the reference: split its extra cost (arena allocation, bind groups, buffer
      zero-fill, first pipeline use) and remove the largest parts, e.g. an arena sized for ~30 s
      at load.

### Task 3: warm speed

- [x] GEMM (~60% of GPU time): tile shape by M, Q8_0 prepacked in fragment order at load,
      double-buffered staging. Compare with ggml-metal `mul_mm`'s layout.
- [x] Attention: 64-key blocks, a kernel specialized for head_dim 48, skip the rescale when no row
      max moved.
- [x] Decoder (38.5 ms vs 19 ms on ru-long): a 4-row joint dot with `std::arch::aarch64` NEON
      intrinsics, plain dot elsewhere.
- [x] Check: `make check`; `make test`; the bench table in `docs/perf.md` and `readme.md`
      updated; roadmap phase 2 ticked if the goal is met.

Decision point: if warm is still slower than the reference on any clip after these, stop and
report to the user with the profile.

## Result

M2, measured at the end of phase 5 on the same GigaAM code, two rounds; wavo / `transcribe-bench`
on Metal, ms (MiB for the peak):

| Clip | Warm median | First call | Load | Peak footprint |
|---|---|---|---|---|
| ru | 43.5–44.1 / 43.7–45.4 | 46.6–47.0 / 46.5–48.7 | 76–78 / 122–124 | 294 / 323 |
| ru-short | 95.8–96.4 / 99.2–101.0 | 99.2–101.4 / 101.1–103.4 | 76–78 / 122–125 | 302 / 325–326 |
| ru-long | 299.3 / 310.4 | 301.4–302.9 / 310.5–334.7 | 76–77 / 122–128 | 321–322 / 334 |
