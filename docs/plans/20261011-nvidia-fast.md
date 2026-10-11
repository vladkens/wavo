# Fast GEMM on 16×16 cooperative matrices

PR #28

wavo was 1.1–3.4× slower than transcribe.cpp's Vulkan build on the Windows RTX 4060 (see the
[Windows plan](20261010-windows.md)): its fast kernels need Apple's 8×8 F32 cooperative matrices,
so every kernel there was the portable one. NVIDIA's driver offers 16×16 F16 matrices with F32
accumulators instead (AMD's and Intel Arc's report them too). Everything stays in our WGSL through wgpu: a fast tier
chosen at load from adapter features, as Apple's is.

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

RTX 4060 Laptop, Windows 11, NVIDIA 610.62, wgpu's Vulkan backend. The driver reports 16×16×16,
16×8×16 and 16×8×8 matrices of F16 into F16 or F32, and 32-lane subgroups; no F32 operands. The
laptop's Radeon 610M (AMD 25.10.30.02) reports 16×16×16 F16 → F32 as well, but its driver kills
the process compiling any shader that uses them, so the tier checks NVIDIA's vendor id. Only the
GEMM runs on 16×16 matrices: the subgroup LayerNorm and conv_glu were slower there, and the
flash kernels need 8×8 F32 matrices. Warm median, `wavo bench -n 20` (`-n 10` for Whisper), main
→ this, two A/B rounds each; transcribe.cpp's Vulkan build (`transcribe-bench --warmup 1 --iters
10`, median `wall_ms`) in the same session, ms:

| Model | Clip | main | 16×16 GEMM | Reference Vulkan |
|---|---|---|---|---|
| GigaAM e2e-rnnt | ru | 23.5 / 23.4 | 20.3 / 20.2 | 133.6 |
| GigaAM e2e-rnnt | jfk | 62.6 / 62.4 | 41.8 / 41.7 | |
| GigaAM e2e-rnnt | ru-long | 193.0 / 193.2 | 122.9 / 123.5 | 1064 |
| Parakeet TDT V3 | jfk | 105.9 / 106.2 | 78.4 / 79.8 | 86.1 |
| Parakeet TDT V3 | ru-long | 316.6 / 318.5 | 203.6 / 202.8 | |
| Parakeet TDT V3 | dots | 355.6 / 357.8 | 230.7 / 236.7 | 244.6 |
| whisper-tiny | jfk | 51.3 / 50.8 | 45.8 / 46.7 | |
| whisper-small | jfk | 265.7 / 266.1 | 198.1 / 197.8 | 102.1 |
| whisper-medium | jfk | 745.2 / 745.7 | 513.4 / 512.8 | 232.5 |
| whisper-large-v3-turbo | jfk | 1080 / 1085 | 645.7 / 647.3 | 223.7 |
| whisper-large-v3 | jfk | 1411 / 1412 | 907.4 / 911.2 | 426.8 |

- 10–40% faster on every model; Parakeet now beats the reference on every clip measured. Whisper
  stays 2–3× slower: the reference's turbo encoder takes 160 ms, wavo's portable attention alone
  471 ms.
- Fixtures exact for all 19 models there. The CLI test `cues_keep_the_fixture_text` fails on that
  machine on main too: its checkout has CRLF fixtures (`core.autocrlf`).
- Load and first call, 7 fresh processes each, interleaved, medians, main → this: turbo 353 → 356
  and 1153 → 733 ms, GigaAM `ru` 238 → 239 and 41.1 → 37.9, Parakeet `jfk` 298 → 291 and 182 →
  162. The 16×16 pipelines compile on the thread that compiles the fast kernels anyway.
- whisper-tiny, -small and -medium ran one commit earlier, before GEMMs of fewer than 16
  workgroups went back to the portable kernel; none of theirs is that small.
- M2 against main, A/B/A/B: GigaAM `ru` 40.8 / 40.7 / 40.8 / 40.8, Parakeet V3 `jfk` 152.8 /
  152.4 / 149.5 / 148.3, turbo `jfk` 1255 / 1235 / 1244 / 1249 ms, the long clips within the noise
  of a busy machine; fixtures exact. The N100 (ANV) reports no cooperative matrices; its GigaAM,
  Parakeet V3 and turbo fixtures pass.

Profile, GPU timestamps per dispatch (temporary tool), warm, ms per call, portable → 16×16 GEMM:

| | turbo `jfk` | Parakeet V3 `dots` |
|---|---|---|
| Kernels total | 1055 → 633 | 222 → 120 |
| GEMMs | 536 → 119 | 167 → 63 |
| FFN up / FFN down | 173 / 174 → 34 / 44 | 49 / 54 → 18 / 21 |
| `[q \| k \| v]` / attention out | 129 / 44 → 24 / 12 | 18 / 7 → 7 / 3 |
| Attention: scores / softmax / values | 208 / 45 / 222, unchanged | 49 / 1.5 / 7, unchanged |
| Decoder gemv / attend | 24 / 13 | on the CPU |

- FFN up runs at ~18.5 TFLOP/s, FFN down (1280 wide, 120 workgroups) at ~14.
- Whisper large-v3 `jfk`: kernels 1386 → 888 ms, of which attention 471 and the decoder's gemv
  and attend 140 + 115 (32 layers, 31 tokens).
