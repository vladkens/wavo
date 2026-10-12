# Flash attention on every GPU

Done: 2026-10-12, PR #36

Whisper on an RTX 4060 (Vulkan) ran 2–3× slower than transcribe.cpp, most of it in the portable
attention that keeps all scores in memory (471 of 633 ms of kernels on turbo `jfk`, see the
[16×16 GEMM plan](20261011-nvidia-fast.md)). The same scores made Moonshine's peak on
long clips higher than the reference's (tiny `dots` 187 vs 138 MiB on the M2). One portable flash
kernel for any GPU, then fast tiers on cooperative matrices where they measure faster.

- [x] Profile with GPU timestamps: whisper-turbo `jfk` and `ru-long` on the RTX 4060 and the M2.
- [x] Read the reference's flash attention (ggml Vulkan `flash_attn*.comp`, Metal `fa.metal`).
- [x] Portable flash kernel (no scores in memory) for attention without relative positions:
      Whisper and Moonshine encoders, GigaAM without the fast tier; fixtures on the M2, the RTX
      4060 and the N100.
- [x] Fast tier on 16×16 F16 cooperative matrices (NVIDIA); A/B/A against the portable kernel.
- [x] Apple's 8×8 tier: `flash.wgsl` against the portable kernel, extended where it pays.
- [x] Decoder attention (self and cross) over key blocks, if the profile asks for it.
- [x] Parakeet's relative attention: measured or reasoned out.
- [x] A/B/A on Parakeet V3 and GigaAM v3 (M2 and RTX 4060); `make check` and `make test` on
      both.

## Result

RTX 4060 Laptop (NVIDIA, Vulkan), warm median, ms, main → this branch (A/B/A, within 3%), and
`transcribe-bench` on Vulkan:

| Model | Clip | main | this | Reference |
|---|---|---|---|---|
| whisper-large-v3-turbo | jfk / dots / ru-long | 637 / 1405 / 1370 | **201 / 508 / 474** | 226 / 542 / 503 |
| whisper-large-v3 | jfk | 910 | **420** | 427 |
| whisper-tiny | jfk / dots | 46 / 153 | **23 / 89** | 43 / 135 |
| moonshine-tiny | dots | 138 | **90** | 144 |
| moonshine-base | dots | 229 | **141** | 227 |
| gigaam-v3-e2e-rnnt | ru / ru-long | 20.3 / 123 | 18.9 / 70 | 134 / 1064 |
| parakeet-tdt-0.6b-v3 | jfk / dots | 74 / 229–247 | the same | 86 / 245 |

Turbo `jfk` per call (GPU timestamps): attention 473 → 39 ms on 16×16 matrices (86 with the
portable kernel); the encoder now takes about the reference's 160 ms. The decoder's attention
on large-v3 `jfk`: 115 → 65 ms.
Parakeet's relative attention stays on the portable scores (~58 of ~120 ms of kernels on V3
`dots`): a flash version is left open in docs/perf.md.

Moonshine `dots`, M1 A/B/A (main / this / main): tiny 1010 / 769 / 1005, base 1980 / 1374 / 1958
ms; on a busy M2, rough A/B: tiny 416 → 236, base 669 → 386 ms. Peak footprint on the M2 187 → 122
and 260 → 194 MiB (reference 138 / 236); whisper-tiny `jfk` 70.5 → 64.9 ms; turbo,
GigaAM and Parakeet unchanged. Fixtures exact for all 21 models on the M2 and the RTX 4060, and
for whisper-tiny and moonshine-tiny on the N100 (portable kernels), where whisper-tiny `jfk` /
`dots` went 2269 / 5016 → 759 / 1988 ms (the reference's Vulkan build 1340 / 3052) and
moonshine-tiny `jfk` 389 → 305 ms.
