# Kernel tests on one machine

Done: 2026-11-07, PR #91

Run the fixtures on every kernel tier a machine can execute, so a kernel change is checked for correctness without other hardware. PR #88 showed the gap: `tests/models.rs` loads through `Backend::Gpu`, which takes the fastest tier, so on a Mac the portable GEMM never ran a fixture, and ivrit he-long drifted there unnoticed. Forcing the portable path on an M5 reproduced that bug exactly.

This tests correctness only. Speed stays as `docs/perf.md` says: A/B on the hardware that matters.

## Decisions

- `WAVO_KERNELS` selects the tier in `Gpu::new(true)`; unset or empty changes nothing. Values: `portable` (128×64 tile), `portable-32` (128×32, the small GEMMs of the 16×16 tier), `portable-cpu` (64×64, `gemm_cpu.wgsl`), `matrix8` (8×8 without the tensor GEMM), `tensor`.
- `Gpu::new(false)`, used only by unit tests, ignores it. Unit tests need it unset: `gpus()` and `device` assert the default tiers, so the Makefile sets it only for `--test models` runs.
- It is a test switch: not in `wavo --help`, not in the rustdoc, named in `AGENTS.md`. `wavo debug` stays as it is, so it reports the tier from before the override.
- No silent fallback: an unknown value, a tier the adapter lacks or a failed probe is `Error::Gpu` from `Gpu::new`, or a green run could test the wrong tier. No new error variant.
  - The check mirrors `gpus()`: the GEMM and all four ops kept, `tensor` present or absent as requested. `info.kernels` can't show this, as it doesn't show the four ops.
  - `matrix8` needs `fast_size == Some(8)`, so that NVIDIA doesn't run 16×16 under that name.
  - `portable` and `portable-32` are refused on `DeviceType::Cpu`: 128-wide tiles crashed WARP on an ARM64 VM, and a crash can't become a clean error.
- The tiers to run are the Makefile's `KERNELS`, an explicit list, not which `Gpu::new` calls succeed, which would hide a failed probe: `portable`, `portable-32`, `portable-cpu`, plus `matrix8` on macOS. A tier the adapter lacks fails; a CPU adapter passes `KERNELS=portable-cpu`.
- All tiers run on every kernel change, with no map from files to tests: a spike took 72–169 s per tier on an M5, about 11 minutes for all five, all 24 tests passing on each.
- `make test` doesn't change. CI adds `make test-portable`, the smoke models on `portable`. `make test-all` stays the real guard: ivrit isn't a smoke model.
- `make test-kernels KERNELS=portable` is the quick mode for work on a kernel: only the named tiers, without the plain `cargo test`; `TESTS=` passes test names through. It stops at the first failing tier.
- A tier that misses a fixture is a finding: report it, never change the fixture.

Out of reach on one Mac, so left to the checks before a release:

- `Matrix16` (NVIDIA). `portable-32` covers its small-GEMM kernel, not how the tier behaves: it runs every GEMM at 128×32 with the non-subgroup decoder.
- Vulkan and DX12 drivers, FXC, compiler differences such as FMA contraction.
- The `COPY_DST` staging upload: an M5 stays `mappable` on every tier.

Before a release, without a GPU: smoke models in VMs on Linux with llvmpipe (SPIR-V) and Windows with WARP (HLSL). With one: the real adapters.

## Checklist

- [x] `src/gpu/mod.rs`: `WAVO_KERNELS` in `Gpu::new` as decided above. Parsing takes a plain `Option<&str>`, unit-tested with no `set_var`. `make check`, `make test-unit`.
- [x] The Makefile and CI: `make test-all` runs `cargo test $(CARGO_FLAGS)` as today, then `make test-kernels`: `cargo test --release --locked --test models -- --skip _ane` per tier in `KERNELS`. CI runs `make test-portable`. Needs `make samples` and `make samples-he`. `make test-all` passes on the M5 on all five tiers.
- [x] `src/gpu/gemm.wgsl`: the `to_half` comment covers activations, which can exceed the f16 range. `docs/perf.md`: a GEMM Log entry for #88 (portable, M5 on Metal, main → PR: `ru` 53.2 → 54.4, `ru-long` 303.8 → 308.5, Parakeet V3 `jfk` 211.1 → 216.0, whisper-tiny `jfk` 56.3 → 56.9 ms). `AGENTS.md`: `make test-kernels` and `WAVO_KERNELS`.

## Result

`make test-all` on an M5: 503 s, all fixtures exact on every tier. `matrix8` 75 s, `portable` 108 s, `portable-32` 126 s, `portable-cpu` 138 s, the plain `cargo test` the rest. `make test-portable` takes 6 s.
