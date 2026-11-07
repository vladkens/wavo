# Kernel tests on one machine

Done: 2026-11-07, PR #91

Run the fixtures on every kernel tier a machine can execute, so a kernel change is checked for correctness without other hardware. PR #88 showed the gap: `tests/models.rs` loads through `Backend::Gpu`, which takes the fastest tier, so on a Mac the portable GEMM never ran a fixture, and ivrit he-long drifted there unnoticed. Forcing the portable path on an M5 reproduced that bug exactly.

This tests correctness only. Speed stays as `docs/perf.md` says: A/B on the hardware that matters.

## Decisions

- `WAVO_KERNELS` selects the tier in `Gpu::new(true)`; unset or empty changes nothing. Values: `portable` (128×64 tile), `portable-32` (128×32, the small GEMMs of the 16×16 tier), `portable-cpu` (64×64, `gemm_cpu.wgsl`), `matrix8` (8×8 without the tensor GEMM), `tensor`.
- `Gpu::new(false)`, used only by unit tests, ignores it. Unit tests need it unset: `gpus()` and `device` assert the default tiers, so the script sets it only for `--test models` runs.
- It is a test switch: not in `wavo --help`, not in the rustdoc, named in `AGENTS.md`. `wavo debug` stays as it is, so it reports the tier from before the override.
- No silent fallback: an unknown value, a tier the adapter lacks or a failed probe is `Error::Gpu` from `Gpu::new`, or a green run could test the wrong tier. No new error variant.
  - The check mirrors `gpus()`: the GEMM and all four ops kept, `tensor` present or absent as requested. `info.kernels` can't show this, as it doesn't show the four ops.
  - `matrix8` needs `fast_size == Some(8)`, so that NVIDIA doesn't run 16×16 under that name.
  - `portable` and `portable-32` are refused on `DeviceType::Cpu`: 128-wide tiles crashed WARP on an ARM64 VM, and a crash can't become a clean error.
- The tiers to run come from an explicit list, not from which `Gpu::new` calls succeed, which would hide a failed probe. The script maps the `kernels` line of `target/release/wavo debug`, leaving out the default tier: Tensor → `matrix8` and the three portable values; Matrix8 or Matrix16 → the three portable values; Portable → `portable-32`, `portable-cpu`.
- All tiers run on every kernel change, with no map from files to tests: a spike took 72–169 s per tier on an M5, about 11 minutes for all five, all 24 tests passing on each.
- `make test` adds the smoke models on `portable` where that isn't already the default and the adapter allows it, so CI needs no workflow change (about a minute more). `make test-all` stays the real guard: ivrit isn't a smoke model.
- `make test-all KERNELS=portable` is the quick mode for work on a kernel: only the named tiers, without the plain `cargo test`; `TESTS=` passes test names through.
- A tier that misses a fixture is a finding: report it, never change the fixture.

Out of reach on one Mac, so left to the checks before a release:

- `Matrix16` (NVIDIA). `portable-32` covers its small-GEMM kernel, not how the tier behaves: it runs every GEMM at 128×32 with the non-subgroup decoder.
- Vulkan and DX12 drivers, FXC, compiler differences such as FMA contraction.
- The `COPY_DST` staging upload: an M5 stays `mappable` on every tier.

Before a release, without a GPU: smoke models in VMs on Linux with llvmpipe (SPIR-V) and Windows with WARP (HLSL). With one: the real adapters.

## Checklist

- [x] `src/gpu/mod.rs`: `WAVO_KERNELS` in `Gpu::new` as decided above. Parsing takes a plain `Option<&str>`, unit-tested with no `set_var`. `make check`, `make test-unit`.
- [x] `scripts/test-kernels.sh` and the Makefile: `make test-all` runs `cargo test $(CARGO_FLAGS)` as today, then `cargo test --release --locked --test models -- --skip _ane` per listed tier, naming the tier on a failure. `make test` adds the smoke models as decided. Needs `make samples` and `make samples-he`. `make test-all` passes on the M5 on all five tiers.
- [x] `src/gpu/gemm.wgsl`: the `to_half` comment covers activations, which can exceed the f16 range. `docs/perf.md`: a GEMM Log entry for #88 (portable, M5 on Metal, main → PR: `ru` 53.2 → 54.4, `ru-long` 303.8 → 308.5, Parakeet V3 `jfk` 211.1 → 216.0, whisper-tiny `jfk` 56.3 → 56.9 ms). `AGENTS.md`: `WAVO_KERNELS`, what `make test` and `make test-all` cover, and the checks before a release.

## Result

`make test-all` on an M5: 653 s, all fixtures exact on every tier. `matrix8` 95 s, `portable` 149 s, `portable-32` 166 s, `portable-cpu` 176 s, the plain `cargo test` the rest. `make test` gains 6 s of tests.
