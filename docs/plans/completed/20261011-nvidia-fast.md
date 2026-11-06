# Fast GEMM on 16×16 cooperative matrices

Done: 2026-10-11, PR #28

wavo was 1.1–3.4× slower than transcribe.cpp's Vulkan build on the Windows RTX 4060 (see the
[Windows plan](20261010-windows.md)): its fast kernels need Apple's 8×8 F32 cooperative matrices,
so every kernel there was the portable one. NVIDIA's driver offers 16×16 F16 matrices with F32
accumulators instead (AMD's and Intel Arc's report them too). Everything stays in our WGSL through
wgpu: a fast tier chosen at load from adapter features, as Apple's is.

- [x] Profile the RTX 4060 with GPU timestamps (Whisper turbo and large-v3 on `jfk`, Parakeet V3
      on `dots`, GigaAM on `ru`).
- [x] Check what wgpu 30.0.1 exposes: adapter features, cooperative-matrix properties, WGSL
      16×16 f16 matrices with an f32 accumulator.
- [x] gemm_fast.wgsl in the fragment size F, built with F = 8 (Apple) or 16; A/B/A against the
      portable GEMM and main; fixtures on NVIDIA.
- [x] Apple's path unchanged (A/B/A on the M2, fixtures); the N100 keeps the portable kernels.
- [x] The laptop's AMD iGPU, which reports 16×16 matrices too: its driver crashes compiling them,
      so the tier is NVIDIA's only.

## Result

RTX 4060 Laptop, NVIDIA 610.62, Vulkan. Only the GEMM runs on 16×16 matrices. Warm median, ms,
`wavo bench` main → this (two A/B rounds, within 3%) and `transcribe-bench` on Vulkan in the
same session:

| Model | Clip | main | this | Reference |
|---|---|---|---|---|
| GigaAM e2e-rnnt | ru / ru-long | 23.5 / 193 | 20.2 / 123 | 134 / 1064 |
| Parakeet TDT V3 | jfk / dots | 106 / 357 | 79 / 233 | 86 / 245 |
| whisper-small | jfk | 266 | 198 | 102 |
| whisper-medium | jfk | 745 | 513 | 233 |
| whisper-large-v3-turbo | jfk | 1082 | 646 | 224 |
| whisper-large-v3 | jfk | 1411 | 909 | 427 |

Turbo `jfk`, GPU timestamps per dispatch, ms per call, portable → 16×16: kernels 1055 → 633,
GEMMs 536 → 119 (FFN up / down 173 / 174 → 34 / 44, ~18.5 TFLOP/s for FFN up), attention 475
unchanged (scores 208, softmax 45, values 222), decoder 37. The reference's turbo encoder takes
160 ms. Fixtures exact for all 19 models; load unchanged, first call lower (turbo 1153 → 733 ms);
the M2 at main's speed.
