# Flash attention on every GPU

Whisper on an RTX 4060 (Vulkan) ran 2–3× slower than transcribe.cpp, most of it in the portable
attention that keeps all scores in memory (471 of 633 ms of kernels on turbo `jfk`, see the
[16×16 GEMM plan](completed/20261011-nvidia-fast.md)). The same scores made Moonshine's peak on
long clips higher than the reference's (tiny `dots` 187 vs 138 MiB on the M2). One portable flash
kernel for any GPU, then fast tiers on cooperative matrices where they measure faster.

- [x] Profile with GPU timestamps: whisper-turbo `jfk` and `ru-long` on the RTX 4060 and the M2.
- [x] Read the reference's flash attention (ggml Vulkan `flash_attn*.comp`, Metal `fa.metal`).
- [x] Portable flash kernel (no scores in memory) for attention without relative positions:
      Whisper and Moonshine encoders, GigaAM without the fast tier; fixtures on the M2, the RTX
      4060 and the N100.
- [x] Fast tier on 16×16 F16 cooperative matrices (NVIDIA); A/B/A against the portable kernel.
- [ ] Apple's 8×8 tier: the same kernel against `flash.wgsl`, kept where faster.
- [x] Decoder attention (self and cross) over key blocks, if the profile asks for it.
- [ ] Parakeet's relative attention: measured or reasoned out.
- [ ] A/B/A on Parakeet V3 and GigaAM v3 (M2 and RTX 4060); `make check` and `make test` on
      both.
