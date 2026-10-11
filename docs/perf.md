# Performance

The log of every optimization attempt, kept or rejected, so that no idea is tried twice without
knowing it. Read it before speed work and add each attempt to the Log. Current speed is the
readme's Speed table; a task's benchmark numbers go in its plan's "## Result" section and its PR.

Unless stated: Apple M2 (8 CPU and 10 GPU cores, 24 GiB), macOS 27, wgpu 30.0.1 on Metal, warm
medians without model load, against transcribe.cpp 5bb2deb on Metal on the same Q8_0 GGUF. Clips:
GigaAM `ru` 4.5 s, `ru-short` 11 s, `ru-long` 33.8 s; Parakeet `jfk` 11 s, `dots` 35.3 s,
`jobs-silence` 5.3 s, for V3 `jfk`, `ru-short` and `uk-short`; Whisper `jfk` (one 30 s window).
Numbers a / b / c follow that clip order.

## How each family runs

**GigaAM.** Load reads the GGUF header and streams each tensor from the file into weight buffers
mapped on unified memory (no staging copy); weights stay Q8_0/F16 on the GPU, Q8_0 scales as f16.
Load runs nothing on the GPU but the fast-kernel probes, in one submission. Metal caches compiled
pipelines across runs by shader source, so new shaders cost 0.2–0.4 s once. Per call: CPU log-mel
(own FFT, one frame at a time), then the encoder over an arena sized for the longest input so far,
bind groups cached per dispatch: one submission with the fast kernels, one per layer on the
portable path (`Pass::layers`). The fast kernels (cooperative-matrix GEMM on f16 operands with
Q8_0/F16 dequantized while staging and W rounded to f16 as the reference does, barrier-free flash
attention with the output in registers, one subgroup per row for LayerNorm and the conv module)
each turn on after a probe dispatch. On NVIDIA GPUs, which have 16×16 F16 matrices instead of
8×8 F32 ones, the GEMM (a GEMM of fewer than 16 such workgroups runs the portable one) and flash
attention (its own kernel, `flash_f16.wgsl`, on f16 operands) are fast; without cooperative
matrices attention is the portable flash kernel. The head's linear layer is the last GEMM: the joint's encoder
projection for RNN-T, the logits (rows padded to 64) for CTC. The RNN-T loop (LSTM, joint, argmax)
runs on one CPU thread with NEON (AVX2 + FMA on x86) tiles that score two frames per pass over the
joint weights; CTC is an argmax and collapse per frame.

**Parakeet.** The shared Conformer block (`src/conformer.rs`) with relative attention: one GEMM for
`[q | k | v]` whose epilogue writes `[q + u | k | v | q + v]`, `P = linear_pos(pos_emb)` per block,
then flash attention that scores each 16-query × 64-key block against the 80 positions it needs
and skews them onto the keys in shared memory. The sinusoidal table is cached on the CPU for the
longest T so far; its 2T + 15 rows are written per call. Subsampling runs in 8 chunks of output
rows, with conv0 + ReLU fused into the first depthwise conv and the pointwise convs through the
GEMM; BatchNorm is folded into a per-channel affine at load. The TDT loop and two LSTM layers run
on the CPU over Q8_0 weights (`cpu::Q8`: int8 values and an f32 scale per 32) dequantized in NEON
registers; `q·d` is exact in f32 and the lanes are `dot`'s, so results equal the F32 path bit for
bit. A step's LSTM (6.6 MB) and V3 joint (5.2 MB) fit in the performance cores' 16 MB L2. Each
product goes out in blocks of 64 rows to four scoped threads that take the next block when free
(`matvecs`); a layer's `Wx·x` and `Wh·h` share one queue.

**Whisper.** CPU log-mel; per 30 s window one submission: the conv stem (im2col and GEMM with GELU,
conv2 added onto the positions), pre-LN blocks with one `[q | k | v]` GEMM and flash attention, the
cross `[K | V]` of every decoder layer packed into F16 caches, and the SOT step. Shapes are fixed
(1500 frames, 448 positions), so the arena is allocated at load. The decoder runs on the GPU, one
pass per token (gemv on Q8_0 rows, attention over the F16 self and cross caches, the tied head over
the vocabulary), and reads the logits back; suppression, timestamp rules, argmax, language and
no-speech run on the CPU, and windows advance as the reference's seek loop.

**Moonshine.** No frontend: the whole input in one pass, with the arena and the rotary table
(computed on the CPU) sized for the longest input so far. One submission (one per layer on the
portable path) holds the conv stem over the raw samples (im2col and GEMM with tanh, a one-workgroup
GroupNorm, two more convs with GELU), the shared pre-LN blocks of `src/transformer.rs` with rotary
q and k and flash attention (heads of 36 and 52 dims), the cross `[K | V]` packed into F16
caches, and the `<s>` step. The decoder is Whisper's step on the GPU with SwiGLU and rotary q and
k, ending in an argmax, so a token reads back 4 bytes; EOS, the 194-position cap and the
repetition guard run on the CPU.

## Not tried yet

GigaAM:

1. Attention: q fragments in registers in a kernel specialized for head_dim 48 (only constant
   loop bounds were tried). Its softmax (~17 of 50 ms on ru-long before the output moved to
   registers) ran 32 subgroup reductions per 64-key block; re-profile first.
2. q·k and v projections as one GEMM (concatenated weights, one A buffer holding `yr | y`): one
   dispatch less per block and more workgroups at small M.
3. Q8_0 prepacked in fragment order: low priority, W traffic is not the GEMM's limit.
4. Decoder: `Wx · embed[token]` depends only on the token, so a per-call cache would skip half of
   the LSTM step (~68 µs) for repeated tokens. On Parakeet V2 it was within noise.
5. Short recordings: under 2 s (24% of real dictation) wavo's call costs ~1 ms (5%) more than the
   reference's, at 2–5 s (37%) the same. At T ≤ 64 frames (2.5 s) FFN down has 12–24 workgroups,
   under two per GPU core. Profile a 1–2 s clip first; candidates are split-K or a 16-row tile at
   small M, item 2, and fixed CPU costs per call.

Parakeet: conv_glu's depthwise taps as [tap][ch] (≤ 2% of GPU time, shared with GigaAM); GEMM tile
shapes at M = 138 (1.5 vs 1.8 TFLOP/s at M = 442 before the f16 64×64 GEMM; re-profile first).
FluidAudio on Core ML's GPU path (F16 activations, 6-bit weights) runs V3 end to end in 8–15%
less time than wavo (jfk 134 vs 145, dots 372 vs 437 ms); its encoder alone takes 103 ms per 15 s
window: time wavo's encoder on the same window to find the gap. An opt-in ANE backend for the
Conformer families (31 ms per window, a third of the energy) is costed in [Other
engines](plans/20261010-other-engines.md).

Whisper, open in [the Whisper plan](plans/completed/20261010-whisper.md) (Task 6): decoder gemv bandwidth
against ~100 GB/s, LayerNorm folded into the next gemv, the two prompt tokens as one two-row step.

GPUs without cooperative matrices, open in [the Linux plan](plans/20261010-linux.md) (Tasks 5
and 7): profile the portable encoder with GPU timestamps; subgroup LayerNorm and conv_glu at any
subgroup width, apart from the cooperative-matrix gate; a register-tiled GEMM with Q8_0/F16
dequantized in registers (optionally F16 math);
Parakeet's fixed cost per call (`first_depthwise`, frames padded to 64); with the person's
approval, since it changes arithmetic, an int8 GEMM (`dot4I8Packed`) on Q8_0 activations as the
reference's CPU path does.

GPUs with 16×16 F16 cooperative matrices ([plan](plans/completed/20261011-nvidia-fast.md)):
Whisper's decoder on large models (gemv 140 + attend 65 ms on large-v3 `jfk`, the reference's
whole decoder 249 ms); Parakeet's relative attention, still the portable scores, softmax and
values (~58 of ~120 ms of kernels on V3 `dots`), as a flash kernel on 16×16 matrices or a portable
one (which would help the N100 too); the GEMM under 16 workgroups (split-K or a 64-wide tile); the
tier on AMD (a Radeon 610M's driver crashed, see the Log; RDNA3 and newer drivers untested) and
Intel Arc.

Load: compiling the pipelines while the weights upload (~7 ms, but every kernel lookup would
wait on a thread); starting the device while the GGUF header is read (≤ 3 ms left).

Memory: in one pass, peak footprint grows with length faster than the reference's (1 → 10 minutes
+509 vs +185 MiB for GigaAM, +695 vs +376 MiB for Parakeet); `wavo run` splits by default. Over a
batch GigaAM keeps the arena of the longest input so far (452 vs 421 MiB).

Research: a fused attention with a larger query tile; items 1–3 of its rejected list (see the
Log), whose verdicts came from a busy machine.

## Generic GPU lessons

- Bound in-flight work: queuing all 32 encoder blocks at once kept every intermediate alive
  (22.7 GB footprint). Waiting every 4 blocks cut it to 9 GB and was faster.
- Dynamic indexing of private arrays in WGSL is slow on Metal. Explicitly unrolled scalar
  accumulators made a GEMM 2× faster.
- M=1 (decoder step, output head) needs a dedicated matvec kernel. A 32-row GEMM tile with one
  useful row was 2× slower (16.6 → 8.6 ms on a 51866×1280 head).
- Q8_0 storage is about memory and load time, not speed: transcribe.cpp on F32-expanded weights
  ran GigaAM 45 / 110 / 368 ms against 44–45 / 102–103 / 312–326 on Q8_0, so Q8 buys it 0–14%
  warm.

## Naga / wgpu quirks (30.0.1)

- `enable subgroups;` is rejected even when `SUBGROUP` is supported; just use subgroup builtins.
- Cooperative matrix splat `CM(0.0)` is rejected: use a zero-initialized `var c: CM`.
- Cooperative ops inside a workgroup-position-dependent branch fail uniformity validation. Keep
  them and barriers outside bounds branches; zero-fill partial tiles instead.
- Cooperative matrix addition compiles in Naga but emits Metal ops that don't exist.
- Verify cooperative support by running a tiny probe dispatch with the real pipeline. Adapter
  subgroup min/max (M2 reports 4..64) is not enough.
- Naga sizes a module's immediates (`var<immediate>`) from the first immediate variable it finds,
  so a second, larger struct in the same module overruns: use one params struct per module.
- A dispatch must set every immediate byte its entry point reads, but bytes it doesn't read may
  stay unset.
- Workgroup arrays sized by override constants (`array<f32, N>` with `override N: u32 = ...`)
  and `@workgroup_size(32 * GROUPS)` work: one source gives pipelines with different shared
  memory layouts.
- `PipelineCompilationOptions { zero_initialize_workgroup_memory: false }`: Naga otherwise clears
  shared memory from a single lane.
- WGSL has square `coop_mat8x8` and `coop_mat16x16` only, of f16 or f32, and takes an f32
  accumulator for f16 operands; the SPIR-V backend emits `OpCooperativeMatrixMulAddKHR`. The
  Vulkan backend lists the driver's subgroup-scope shapes of sizes 8 and 16 (NVIDIA's 16×8 too,
  which WGSL can't name) and offers `EXPERIMENTAL_COOPERATIVE_MATRIX` only with
  `vulkanMemoryModelDeviceScope`. A type can't be an override constant: one source serves 8×8
  and 16×16 by text replacement at load.

## Where to look for kernel ideas

- `3rd/transcribe.cpp/ggml/src/ggml-metal/` is what the reference runs on Apple Silicon:
  `kernels/mul_mm.metal` (matmul), `kernels/mul_mv.metal` (matrix-vector), `kernels/fa.metal`
  (flash attention), `ggml-metal-fusion.cpp` (which ops it fuses), `ggml-metal-tuning.cpp`
  (per-device parameters). Its Q8_0 matmul (`kernel_mul_mm_q8_0_f32`) dequantizes weights to
  half, casts activations to half, multiplies with `simdgroup_half8x8` and accumulates in F32.
- `3rd/transcribe.cpp/ggml/src/ggml-webgpu/wgsl-shaders/` is ggml's WebGPU backend, written in
  WGSL like ours, with Q8_0 `mul_mat`, `mul_mat_vec` and flash attention. transcribe.cpp doesn't
  use this backend, but its shaders can be adapted directly (research did this for the Whisper
  output head).

## Log

Each entry sits under its topic, newest first: date, model, what was tried, the result in one or
two numbers (A/B/A, warm median unless stated), kept or rejected and why.

### GEMM

- 2026-10-11, all models, Windows RTX 4060: the fast GEMM on 16×16 F16 cooperative matrices (F32
  accumulators), gemm_fast.wgsl written in F and built with F = 8 or 16 (at 16: 64 or 128 rows ×
  128 columns). GEMMs on turbo `jfk` 536 → 119 ms; warm 10–40% faster on every model (turbo 1082
  → 646, Parakeet V3 `dots` 357 → 233 ms); fixtures exact. Kept.
  [Plan](plans/completed/20261011-nvidia-fast.md).
  - Under 16 workgroups the 128-wide tiles lose (GigaAM `ru`'s 768-wide layers get 12: FFN down
    6.96 → 10.54 ms, `ru` 23.5 → 25.4 ms), so those GEMMs stay portable: `ru` 20.2 ms. Kept.
  - Checking columns past n (W is padded to 64 rows) at F = 8 too cost the M2 8% on `ru-long`;
    staging in loops of one or two steps cost it 3% on `ru`. Both stay off the F = 8 path.
  - The laptop's Radeon 610M (AMD driver 25.10.30.02) reports 16×16 F16 → F32 matrices, but the
    process dies while the driver compiles any shader using them, even a 10-line one; a probe
    can't catch that. The tier is limited to NVIDIA's vendor id.

- 2026-10-10, all models, Windows RTX 4060: NVIDIA's Vulkan driver folds the portable GEMM's
  `pack2x16float` round trip into a no-op (AMD's truncates), so W was not rounded to f16 and
  e2e-ctc ru-short missed its fixture. Rounding to nearest even by hand: fixtures exact, warm
  unchanged (ru-long 1749 / 1755 ms). Kept.
- 2026-10-10, all models, fast GEMM on f16 operands, 64×64 tiles, K step 8 (32×64 for short
  inputs): Whisper jfk 1495 → 1298, GigaAM ru-long 301 → 248, Parakeet dots 498 → 448 ms; fixtures
  exact. Kept.
- 2026-10-10, Parakeet V2, q computed once: the qkv GEMM computes `[q | k | v]` and its epilogue
  writes q again with `pos_bias_v` after v (a bias longer than the rows), exact. Peak −25.5 MiB of
  weights (jfk 807 → 782), jfk 164 → 161 ms. Kept.
- 2026-10-09, GigaAM variants, GEMM rounds dequantized W to f16, as the reference's
  `kernel_mul_mm` does: without it e2e-ctc ru-short emits a token a frame late (a near-tie, mean
  |Δ logits| 0.0026). W only: all four models' fixtures exact, warm unchanged. Kept. Rounding a as
  well: +1.0 ms on ru and no fixture needs it, rejected. Rounding with integer ops instead of
  `pack2x16float`: +2.4 ms (W) / +4 ms (both) on ru, rejected.
- 2026-10-09, GigaAM, skipping the MMAs of a subgroup's lower 8 rows when all are past m (6% of
  the MMAs at M = 113): the fast GEMM no longer passes its probe (a branch on the subgroup id
  around cooperative-matrix calls). Rejected.
- 2026-10-09, GigaAM, GEMM limits (timing-only ablations with wrong outputs, ru-long, FFN down
  72–74 ms): all W loads from one 2 KB region 71.3, all a loads from one region 73.8, staging only
  on the first K step 60.9, MMAs only on the first K step 37.3 ms. Device memory traffic costs
  nothing, staging ~15%, the rest is the on-chip loop (FFN up ~1.8 TFLOP/s, half the M2's FP32
  peak). A 64×64 tile with F32 operands (each subgroup 32×32, 16 accumulators): 73.3 ms, no gain,
  rejected. Prepacking W in fragment order not tried: W traffic is not the limit.
- 2026-10-09, GigaAM, split-K for short clips (timing-only: FFN down with 4 K-slices as `wg.z`,
  no reduction): at M = 113 11.7–12.4 vs 12.3–12.6 ms, only at M = 2 13.8 → 9.6 ms. Its 48
  workgroups already run at FFN up's throughput. Rejected without the reduction pass.
- 2026-10-09, GigaAM, register prefetch (the next K step's a and raw W words loaded before this
  step's multiplies): FFN down on ru 12.4–12.7 vs 12.3–13.4 ms, ru-long noise. Rejected.
- 2026-10-09, GigaAM, Q8_0 cost (timing ablations): q·k 1536←768 Q8_0 27.2 ms vs 16.7 for the
  F16 pointwise1 of the same shape; without the scale lookup 25.6, without the int8 unpack 25.4,
  q·k stored as F16 ~25 ms. So dequantizing costs ~8% (5% the scale lookup); the rest of the gap
  depends on where the GEMM sits in the block (per-dispatch timestamps likely charge the previous
  kernel's write-back). Scales next to the data would add ~12 MB for ≤ 5%: nothing kept. At M = 2
  (load warm-up) FFN down takes 0.37 ms per dispatch: 12 workgroups each walk 192 K steps.
- 2026-10-09, GigaAM, F16 operands (a and dequantized W rounded to f16 in shared memory,
  f16×f16→f32 cooperative matrices, 32×64 K16 tile): fixtures unchanged but ru-long 329.3 →
  353.1, ru 45.4 → 45.9 ms. Rejected (the 64×64 K8 version above kept them).
- 2026-10-09, GigaAM, K step 32 instead of 16 (one Q8_0 block per step, half the barriers):
  ru-long 330.0 → 347.8 ms. Rejected, as in research.
- 2026-10-09, GigaAM, weight type as a pipeline-overridable constant (one pipeline per type)
  instead of a runtime switch: fast ru 56.9 → 50.0, ru-long 431 → 396 ms; portable neutral. Kept.
  The one-entry portable GEMM is slower than Task 1's three-entry one (ru 85 → 118 ms); cause not
  found, fallback only.
- 2026-10-09, GigaAM, fast paths: cooperative-matrix GEMM (research's `gemm_fast` layout: 32×64
  tile, K16, F32 8×8, Q8_0/F16 dequantized while staging W), flash attention (64 queries × 32 keys
  per workgroup, online softmax, accumulator in shared memory), one subgroup per row for
  LayerNorm(+RoPE) and conv_glu. Portable → fast: ru 118.0 → 56.4, ru-long 1046 → 427 ms. Kept.
  After it on ru-long: GEMM ~245 ms (~1.5 TFLOP/s), attention ~109, other GPU ~8; CPU mel 25,
  decoder 43 ms.

### Attention

- 2026-10-12, Moonshine, M2: Apple's 8×8 flash kernel for heads of a multiple of 4 (36, 52)
  instead of the portable one: a last fragment of 4 columns takes q from a zero-padded copy in
  shared memory (k's extra columns then add zeros), reads 4 floats past each v row and writes
  only its first 4 output columns; Moonshine's qkv rows padded to 64 (+4 floats). A/B/A: `dots`
  tiny 271 / 236 / 277, base 445 / 386 / 442 ms; fixtures exact. Kept. Staging that copy only
  for such heads: GigaAM `ru` 41.1 / 41.9 / 41.2 with it always, within noise without.
  - The partial fragment as a branch inside the d loop fails validation: Naga treats the loop
    variable as non-uniform around cooperative-matrix loads, so the tail is its own block.
- 2026-10-12, Whisper and Moonshine decoder: `attend` split over workgroups of 256 keys per head
  (each writes its maximum, sum and unnormalized output) and a `combine` dispatch per head that
  merges them, always both dispatches so a step's dispatch list (and its cached bind groups)
  doesn't depend on the number of keys. One workgroup per head had left most of the GPU idle (8
  heads on Moonshine). RTX 4060 A/B/A: whisper-turbo `jfk` 206.8 / 201.4 / 207.2, `ru-long`
  493.5 / 472.4 / 495.0, large-v3 `jfk` 472.9 / 420.7 / 472.5, tiny 29.2 / 23.3 / 28.0, Moonshine
  `dots` tiny 112.4 / 90.7 / 112.4, base 190.3 / 140.6 / 190.4 ms. M2: Moonshine `dots` tiny 292 /
  278 / 295, base 489 / 447 / 488, whisper-tiny `jfk` 74.5 / 68.7 / 74.5 ms, turbo within noise.
  Fixtures exact. Kept.
- 2026-10-12, Whisper, Moonshine and GigaAM, RTX 4060: flash attention on 16×16 F16 cooperative
  matrices (`flash_f16.wgsl`, after ggml's `flash_attn_cm1.comp`): four subgroups take 64
  queries, k and v staged as f16 per 64-key block, q and output fragments in registers, the
  scores and f16 probabilities through shared memory (45 KB). A/B/A against the portable flash
  kernel: whisper-turbo `jfk` 260 / 207 / 260, `ru-long` 602 / 495 / 603, large-v3 `jfk` 528 /
  471 / 528, tiny 31.2 / 29.2 / 30.1, GigaAM `ru-long` 75.2 / 70.4 / 75.0, Moonshine `dots` tiny
  114.9 / 112.2 / 114.7, base 194.5 / 190.3 / 194.4 ms; fixtures exact. Kept. Turbo `jfk` per
  call: attention 86 → 39 ms (~9.5 TFLOP/s).
  - A row maximum that moves only when passed by more than 8 (flash.wgsl's, fewer rescales):
    the same speed, but GigaAM e2e-ctc `ru-short` emitted a token a frame late. Rejected for the
    exact maximum.
- 2026-10-12, Whisper, Moonshine and GigaAM, portable flash attention for heads of ≤ 64 dims (a
  multiple of 4) without relative positions: a workgroup of 128 threads takes 32 queries of one
  head and walks the keys in blocks of 32 (q, k transposed and v in shared memory, 21 KB), each
  thread scoring 2 queries × 4 keys and summing 2 × 8 output columns, online softmax; no scores
  in memory. RTX 4060 A/B/A: whisper-turbo `jfk` 639 / 260 / 642, `ru-long` 1365 / 603 / 1370,
  GigaAM `ru-long` 124 / 79 / 124, Moonshine tiny `dots` 138 / 115 / 138, base 228 / 194 / 228
  ms; Parakeet unchanged. M2 Moonshine `dots` tiny 424 → 302, base 687 → 498 ms, peak 187 → 120
  and 259 → 194 MiB (reference 138 / 236). Fixtures exact (all 21 models on the RTX 4060, M2).
  Kept. Turbo `jfk` on the RTX 4060 after it, GPU timestamps per call: attention 473 → 86 ms,
  GEMM 122, decoder gemv 23.5 and attend 13 ms.
  - On the M2 in place of the 8×8 `flash.wgsl`: turbo `jfk` 1232 → 1926, GigaAM `ru-long` 248 →
    333 ms. Not used there.
  - Writing a vector component by a runtime index (`a[i][r % 2u] = x` on a workgroup array of
    vectors) gave NaN on the M2: the staging writes whole vectors instead.
- 2026-10-11, Whisper and Moonshine, decoder `attend` in blocks of 2048 keys with an online
  softmax, so cross-attention takes any length (Moonshine runs one pass over the whole input):
  bit-identical up to 2048 keys; whisper-tiny `jfk` on the RTX 4060 50.0 / 51.2 / 51.5 ms on main,
  51.4 / 51.3 with it. Kept.
- 2026-10-10, Whisper and GigaAM, flash attention with the output in registers, 9.5 KB shared:
  Whisper jfk 1550 → 1403, GigaAM ru-long 297 → 276 ms, Parakeet unchanged; fixtures exact. Kept.
- 2026-10-10, Parakeet V2 and GigaAM, one flash kernel for both attentions. One layout (one
  subgroup, scores [16][80], accumulator [16][128]) slowed GigaAM ru-long 296 → 328.6 ms (305.7
  with a [16][64] accumulator): shared memory per query (occupancy) is the cost. Kept: one source
  whose override constant `REL` sizes the shared arrays and workgroup per pipeline; GigaAM
  unchanged. The portable relative attention adds `(q + v)·P` inside `scores` instead of a
  `pos_scores` kernel and buffer.
- 2026-10-10, Parakeet V2, flash attention with relative positions for head_dim 128: one 32-lane
  subgroup owns 16 queries and walks 64-key blocks; per block it scores its q + v rows against the
  80 positions the block needs (two passes of 5 × 2 fragments, 10% over the 72 + 72 needed),
  stores them with a row stride of 81 so position c + 15 − r of row r lands on key c, and loads
  them as the initial accumulators of the q + u · k scores; then online softmax and P·V with a
  [16][128] accumulator. jfk 180.5 → 160.3, dots 762 → 544 ms, no `ps` and `s` buffers. Kept. Two
  subgroups of 16 queries per workgroup: within noise, rejected for the simpler one.
- 2026-10-09, GigaAM, softmax with two lanes per row (each lane loops over 32 keys of one parity,
  one `subgroupShuffleXor` per reduction): rows 66 floats apart (unaligned) 98.5 ms, 72 apart
  (4-way bank conflicts) 55.5, column-major 16-float columns 56, vs 50.5 ms (GPU timestamps,
  ru-long). Rejected. Cooperative-matrix loads and stores on rows not 16-byte aligned cost ~2×.
  Ablations of the kept kernel: scores ~20, softmax ~17 and P·V ~13 ms of the 50.5.
- 2026-10-09, GigaAM, kernel specialized for head_dim 48 (constant loop bounds): 48.9 → 49.8 ms.
  No gain, rejected.
- 2026-10-09, GigaAM, 64-key blocks (scores in two 32-key halves, softmax with two keys per lane,
  one output load/store and rescale check per 64 keys): attention 56.4 → 48.9 ms, warm ru-long
  322.3 → 316.6 ms. Kept.
- 2026-10-09, GigaAM, skip the output rescale when a row's running max didn't move (exact, the
  factor would be 1): attention 71.4 → 56.4 ms, warm ru-long 337 → 322 ms. Kept.
- 2026-10-09, GigaAM, flash attention without workgroup barriers: each subgroup walks all keys on
  its own, loads k and v fragments from memory and syncs only with `subgroupBarrier` around its
  softmax. Attention ~111 → ~85 ms, ru-long 357.9 → 331.3 ms. Kept. Rows of qk/v past t (masked)
  must be finite: the arena's are.
- 2026-10-09, GigaAM, variants of the barrier version (ru-long): q tile staged once in shared
  memory 330.4 → 329.0, no gain, rejected; q fragments in an array of cooperative matrices 375.9 →
  394.2 (spills), rejected; shared memory 32 → 26 KB 376.5 → 374.4, no gain, rejected; k/v
  staging loops without division by head_dim 377.5 → 370.4, kept until the barrier-free kernel
  removed staging; softmax with lane = key and subgroupMax/Add instead of 2 lanes per row (16-way
  bank conflicts) no change alone, kept in the barrier-free kernel. Without the softmax/rescale
  phase attention saved 47 of 111 ms: the serial phase between barriers was the cost.

### Other encoder kernels and dispatches

- 2026-10-11, all models, Windows RTX 4060: the subgroup LayerNorm and conv_glu next to the 16×16
  GEMM: LayerNorm 5.0 → 8.2 ms on turbo `jfk`, conv_glu 0.89 → 1.16 ms on Parakeet V3 `dots` (GPU
  timestamps). Rejected there.

- 2026-10-10, Parakeet V2 and GigaAM, simplification (core + Parakeet 3171 → 3103 lines): one
  LayerNorm kernel whose rotary output runs behind a uniform head_dim check (without rotary the
  table binding aliases x and `yr` is a 4-byte spare, as WebGPU forbids aliased writable
  bindings), one `Pass::depthwise` with an optional first conv, `Norm` and `Depthwise` as one
  `Channels`, the decoder reading the joint's rows at the GEMM's padded width. Speed and fixtures
  unchanged. Kept. The rotary check inside the fast kernel's per-element loop cost GigaAM ru
  +0.3 ms, so the rotary output got its own loop.
- 2026-10-10, Parakeet V2, `first_depthwise` (conv0 + ReLU fused into conv2): 5.7 / 18.9 ms
  (jfk / dots) → 4.8 / 12.4 with weights as [tap][ch] → 3.3 / 11.3 with the 7×7 input pixels in
  shared memory → 0.88 / 3.06 with conv0's nine weights held as scalars (they were reloaded per
  tap). Kept. One workgroup per output row (mel rows in shared memory, weights in private arrays):
  5.1 / 14.7 ms, rejected.
- 2026-10-10, Parakeet V2, shared core for it (Conformer block, relative attention, conv_glu
  kernel size and affine mode, attention kernels in their own module): GigaAM unchanged, ru 43.1 /
  43.3, ru-long 296.1 / 296.3 ms. Kept.
- 2026-10-09, GigaAM, small encoder experiments on ru (encoder wall time with a temporary timer;
  the machine drifts ±1 ms over minutes), all rejected:
  - 48 extra one-workgroup im2col dispatches: 39.9–40.4 vs 39.8–40.5 ms, so a dispatch boundary
    costs little. (48 extra one-row LayerNorms +1.4 ms and 48 fewer −1.2 ms couldn't be told
    apart from the drift.)
  - Each block's output norm fused with the next block's first norm (the first output recomputed
    per element, −15 dispatches): 38.8–38.9 / 38.9–39.3 / 38.9–39.1 ms, no gain.
  - LayerNorm statistics with eight loads issued before their sums: 44.4 / 44.2 / 44.1 ms, no gain.
    LayerNorm rows held in a 32-entry register array: encoder +1.5 ms (likely spilled).

### CPU decoders

- 2026-10-10, all models, x86_64: runtime-dispatched AVX2 + FMA versions of the decoder tiles and
  `Q8::rows`, bit-identical. N100 default build decoder ru-long 1243 → 55, dots 2031 → 236 ms;
  warm ru-long 8515 → 7418 ms. Kept.
- 2026-10-10, x86_64, `-C target-cpu=native` (`mul_add` without FMA is a libm call): N100 decoder
  13–15× faster on GigaAM, 2.6–2.8× on Parakeet V2 (warm ru-short 2536 → 2133, dots 13628 → 12337
  ms); RTX 4060 box warm ru 234 → 24, V3 dots 2799 → 585 ms. Not kept as a build flag: the
  runtime-dispatched AVX2 tiles above replace it.
- 2026-10-10, Parakeet V3 and V2, decoder weights Q8_0 on four threads (direct timer, median of 18
  calls). F32 before: V3 decode 76 / 68 / 70 ms (the 8198 × 640 joint 31–42 ms, the predictor
  0.9–1.1 ms per token against V2's 0.52: the joint's 21 MB evicted the LSTM from L2), V2 20.5 /
  114 ms (jfk / dots).
  - Every decoder weight Q8_0, dequantized in NEON registers, bit-identical: one thread per
    product V3 51 / 77 / 73, with each layer's `Wx·x` on a second thread 41 / 61 / 58 ms. The
    kernel alone (8200 × 640): 486 µs, ~10.8 G weights/s, near its limit of 19 NEON ops per 16
    weights.
  - Rows split evenly over scoped threads: the kernel on 2 / 3 / 4 / 6 / 8 threads 265 / ~197 /
    ~161 / 211 / ~182 µs (past four the efficiency cores set the pace; a scoped spawn costs 7–15
    µs); in place the V3 joint on four threads took 8–17 ms from run to run. Rejected.
  - Blocks of 64 rows handed to four threads as they come free (a locked chunk iterator; a
    layer's two products share it): the kernel 149–152 µs; decode V3 21–25 / 31–36 / 29–35, V2
    13–15 / 71–78 ms; 3, 6 and 8 threads slower. Kept at four. `joint.pred` on one thread or four:
    the same, so `matvecs` always uses four.
  - Dequantizing each group of 4 rows into a buffer for the F32 `tile::<4, 1>` (~29 lines less):
    1965 vs 486 µs per kernel call. Rejected.
  - Warm: V3 jfk 175 → 150, ru-short 197 → 160, uk-short 194 → 162 ms; V2 jfk 150 → 144, dots
    498 → 461 ms; GigaAM unchanged. Peak V3 808 → 762, V2 794 → 766 MiB (dots). Kept (+95 lines
    in `src/cpu.rs`).
- 2026-10-10, Parakeet V2, LSTM (direct timer, dots / jfk; the F32 step streams 28 MB: 129 / 24.0
  ms):
  - Each layer's `Wx·x` on a scoped thread while `Wh·h` runs on the caller (bit-identical, no
    waiting thread between steps): 113 / 20.4 ms, warm dots 517 → 501 ms. Kept (6 lines).
  - Four threads (row halves of `Wx` and `Wh`): 122–130 / 22.3–22.8 ms. Rejected: three spawns per
    layer and one memory bus.
  - LSTM weights Q8_0 (7 MB, fits in L2) dequantized in NEON registers: one thread 142 / 25.5 ms
    (compute-bound on widen, convert and scale), two threads 96.7 / 17.5 ms and −24 MiB. Rejected
    then for ~40 lines of NEON against 16 / 3 ms; adopted with V3 (above).
  - Dequantizing 16 rows at a time into a buffer for the F32 tiles: 129 / 23.3 ms with threads.
    Rejected.
  - Layer 0's `Wx·embed[token]` cached per call (dots repeats 92 of its 194 tokens, jfk 11 of 33):
    warm jfk 149.6 / 148.6 / 149.8, dots 503.8 / 499.5 / 500.1 ms. Within noise, rejected.
- 2026-10-09, GigaAM, NEON tiles (`std::arch::aarch64`): a tile computes 2 rows × 2 frames or 4
  rows × 1 frame with `dot`'s 16 lanes and pairwise sum, bit-identical; the joint scores frames in
  pairs against one predictor output (~8% rescored). Decoder ru-long 38.5 → 27.2 ms, warm 315 →
  306 ms. Kept. 4 × 1 tiles alone 31.3 ms; spans of 4 frames with 1 × 4 tiles 29.8 ms (18%
  rescored). After it: pair spans 13.9, LSTM + predictor 9.8 (~57 GB/s from L2), single-frame
  joints 3.2 ms; the reference spends 19 ms (Accelerate sgemm on 32-frame spans).
- 2026-10-09, GigaAM, removed the helper thread below (a library must not busy-wait a core, and a
  panic in the main loop would hang `thread::scope` on it): decoder 26 → 38.5 ms on ru-long.
  Single-threaded replacements, all slower than the plain matvec (38.7 ms):
  - 4 joint rows per pass over z, so each z chunk loads once for 4 rows: generic `dots<const R>`
    94 ms (accumulators spilled), index loops 43.5 ms (bounds checks, z reloaded per row), zipped
    row iterators 61.6 ms (LLVM's SLP vectorizer transposes across rows with shuffles). Rejected;
    needed NEON intrinsics (the tiles above).
  - Spans of frames sharing one predictor output, rows outer and frames inner (the reference's
    lookahead): not retried after the 8-frame try below. The matvec is bound by L1 loads and FMAs,
    not L2 traffic, so a weight row kept in L1 across frames saves nothing unless its registers
    are reused across frames, and the frames after each emission are rescored (~20% more dots).
- 2026-10-09, GigaAM, a helper thread scores the upper half of the joint vocabulary, spinning on
  an atomic step counter during `decode`: decoder 38 → 26 ms, warm ru-long 343.1 → ~332 ms,
  bit-exact. Kept, then removed (above).
- 2026-10-09, GigaAM, decoder dot: 32 accumulators 43 → 48 ms (the 32 scalar adds of the final sum
  dominate a 320-long dot), rejected; 16 FMA accumulators summed pairwise 43 → 38.5 ms, kept.
- 2026-10-09, GigaAM, joint scored for spans of 8 frames per weight pass while the predictor
  output is unchanged: ru-long 397.5 → 417.2 ms. Rejected: rescoring the frames after each token
  outweighs the cache reuse.

### Whisper decoder

- 2026-10-10, Whisper, decoder gemv over 4 rows per 32 threads: jfk 1603 → 1563 ms. Kept.

### Frontend and FFT

- 2026-10-10, all models, the own FFT with two radix-2 stages per pass (same butterflies, half
  the memory passes): 7945HX Parakeet jfk mel 4.24 / 4.22 / 4.28 / 4.16 ms, FFT 320 0.41 → 0.54
  µs. Rejected: no gain.
- 2026-10-09, GigaAM, one frame at a time in a reused buffer (no 8.6 MB spectrum on ru-long, no
  div/mod per sample), each mel filter's `dot` only over its nonzero bins in 16-bin chunks (the
  skipped terms are exact zeros): mel bit-identical, frontend ru-long 7.6 → 4.4 ms (the rest is the
  FFT ~3 ms and `ln`), warm ru-long 307.8 → 303.2 ms. Kept.
- 2026-10-09, GigaAM, mel filterbank with the vectorized dot: mel 25 → 8 ms on ru-long. Kept.

### Load and first call

The reference's `load_ms` covers less than wavo's load: not its first GGUF parse (`Loader::open`
runs before the timer) and no Metal pipelines, which ggml compiles in the first call. Compare load
plus first call.

- 2026-10-11, all models, no warm-up at load (the encoder run on 8 silent frames; for Whisper a
  64-frame window and a decoder step), decoder pipelines compiled in the first call: load +
  first call on jfk 5–89 ms lower on every model (GigaAM e2e-rnnt 162.1 → 142.8, Parakeet V3
  298.4 → 253.3, turbo 1486 → 1430 ms), the first call 4–40 ms higher. Kept: the warm-up cost
  12–115 ms at load and saved the first call only 4–40, since its 8-frame arena is allocated again
  by the first call and the weights are mapped since the 2026-10-09 warm-up entry below.
- 2026-10-11, all models, four fixed load costs and a copy: load whisper-base 67.1 → 43.3,
  large-v3 319.0 → 269.2, GigaAM e2e-rnnt 76.6 → 65.4, Parakeet V3 163.6 → 148.5 ms; first call
  unchanged, fixtures exact. Kept. Per part (temporary timers):
  - GGUF metadata decoded when read, not all at open (Whisper's 1.9 MB header holds ~150k values:
    51,866 tokens, their types and 50,000 merges): open 8.1 → 1.2 ms, the tokenizer 0.9 → 1.8 ms.
  - Whisper's 99 language tokens searched from the end of the vocabulary, where the special
    tokens are: 8.1 → 0.2 ms.
  - Fast pipelines compiled on a second thread while the portable ones compile: 7.8 → 4.6 ms.
  - The ten fast-kernel probes in one submission instead of one each: 4.7 → 2.5 ms.
  - Q8_0 blocks written straight into the mapped buffers instead of through a scale and a data
    `Vec` per 278 KB piece: large-v3 upload 170 → 138 ms. The upload is the floor for large
    models: ~10 GB/s from one thread per block, a `read` from the page cache, after wgpu zeroes
    each buffer it maps at creation (the reference: ~3 GB/s). The adapter and device take ~10 ms,
    as in the reference.
- 2026-10-10, all models, warm-up at load only with the fast kernels (the portable GEMM pads one
  frame to 64 rows, 0.4–1 s spent to save 8–20 ms): N100 load ru 586 → 191, jfk 1389 → 390 ms;
  RTX 4060 GigaAM 272 → 230, Parakeet ~440 → ~345 ms, first call +13 / +35–39 ms. Kept; the
  2026-10-11 entry dropped the warm-up everywhere.
- 2026-10-09, GigaAM, warm-up at load, an encoder run on 8 silent frames: first call ru 92.1 /
  71.9 → 50.8 / 51.1, ru-long 369.9 → 348.1 ms; load 65–68 → 77–81 ms. Kept then, dropped
  2026-10-11. With all weights flushed at load (`queue.submit([])` + wait) the first call fell to
  53 ms and load rose to 96 ms: the cost was wgpu's deferred staging copies. Mapped weight buffers
  removed those, but the first GPU use of pipelines and weights still cost ~41 ms of GPU time even
  on 8 frames.
- 2026-10-09, GigaAM, arena allocated for 30 s at load (warm-up on it) instead of on demand:
  first call ru 51.5 / 50.1, ru-short 120.6 / 108.4 ms, no better, +26 MB on short clips.
  Rejected.
- 2026-10-09, GigaAM, Q8_0 repack by per-block memcpy instead of a byte iterator (encoder upload
  181 → 77 ms), the 16 blocks repacked and uploaded from 16 threads (77 → 28 ms), GGUF read in 8
  parallel chunks (36 → 20 ms): load 233 → 60–90 ms. Kept.

### Memory

- 2026-10-10, all models, `MemoryHints::MemoryUsage`: Vulkan suballocates from 8–64 MiB blocks
  instead of 128–256; Metal ignores it. N100 iGPU buffers 529 → 324 (ru) / 436 (ru-long), 914 →
  777 (jfk) / 810 MiB (dots), speed within 0.35%, load and RSS unchanged; M2 unchanged; fixtures
  exact on both. Kept. (A first one-run check on the N100 left it pending A/B/A.)
- 2026-10-10, Parakeet V2, the memory task as a whole (q once, chunked subsampling, flash
  attention without `ps` and `s`): peak 808 → 762, 857 → 791, 796 → 760 MiB (reference 820 / 834 /
  815), warm jfk 161 → 153, dots 568 → 531 ms; GigaAM unchanged. Kept.
- 2026-10-10, Parakeet V2, subsampling in 8 chunks of ⌈T/8⌉ output rows (always 8, empty ones
  dispatch nothing, so the cached bind groups match): conv0 + conv2 into a chunk buffer, conv3
  into `qk`, conv5 into its rows of `h`, so `h` and `qk` shrink to the blocks' size. Peak 780 →
  765, 830 → 789, 771 → 758 MiB, warm unchanged. Kept.
- 2026-10-09, GigaAM, the GGUF header read alone and every tensor streamed from the file in
  278 KB pieces into weight buffers created mapped (`MAPPABLE_PRIMARY_BUFFERS`, no staging copy);
  Q8_0 scales kept f16 (−12 MB); the arena drops its im2col and v buffers (−12 MB on ru-long). Peak
  920 / 922 / 961 → 307 / 318 / 346 MB (reference 338 / 340 / 349); warm unchanged (shared vs
  private weight buffers 334.6 vs 334.0 ms). Kept. Skipping the unused portable pipelines saved
  0.5 MB: rejected.

### Submissions and drivers

- 2026-10-10, all models, `Pass::layers` submits each encoder layer on its own on the portable
  path (Whisper had one submission per window, which i915 cancelled on the N100). M2 unchanged:
  ru 41.3 / 41.4, jfk 137.8 / 137.9, Whisper jfk 1246 / 1245 ms. Kept.
- 2026-10-10, all models, GPU work that fails or that a driver cancels is an error: i915 cancels a
  request 20 s after it was queued and wgpu reports success, so every submission ends with a
  `mark` dispatch that writes the call's number to its own slot, read back with the output, and
  wgpu's uncaptured and device-lost errors are kept. The portable path submits each Conformer
  block on its own, each waiting for the one before (i915 starts the clock when a request is
  queued). N100: `make test` in parallel 0 → 6 of 6, `wavo run parakeet-v2` on 5 minutes 0 → 808
  words. Per block on every GPU cost the M2 ru 43.6 → 44.1, dots 453 → 456 ms, so the fast path
  stays one submission (then within noise). Kept.

### Build

- 2026-10-10, all models, own FFT (`src/fft.rs`, transcribe.cpp's radix-2 over odd DFT leaves in
  Stockham order, FMA on x86_64 when present) for all three frontends instead of `rustfft`.
  Windows 7945HX: clean `cargo build --release` 40.0 → 29.5 s, library rebuild 11.9 → 2.1 s,
  library LLVM IR 353k → 93k lines. Mel bitwise unchanged for Whisper and Parakeet, GigaAM within
  4e-4, fixtures exact; mel per call Whisper jfk 7.3 → 3.4 ms (M1), Parakeet jfk 2.7 → 3.2 ms.
  Kept.

### Long audio

- 2026-10-10, GigaAM e2e-rnnt and Parakeet V3, `wavo run` split at the quietest 300 ms (≤ 25 /
  60 s segments): 10 minutes 15.6 → 5.2 s and 22.9 → 8.9 s, peak 856 → 349 and 1513 → 865 MiB,
  GigaAM 177 → 905 of ~923 words. Kept. Time is linear in length (~0.52 / ~0.88 s per minute of
  audio) and peak flat apart from the decoded audio (~230 MB per hour). Parakeet `--segment` 30 /
  60 / 120 on 10 minutes: 8.6 / 8.9 / 9.8 s, 830 / 863 / 955 MiB; 120 s skips a sentence, and over
  60 minutes 30 s twice inserts a phrase nobody says. 60 s is the default.
- 2026-10-10, GigaAM e2e-rnnt and Parakeet V3, 1 to 10 minutes in one pass: text identical to the
  reference's. GigaAM (trained on ≤ 25 s) drops words past ~1 minute (177 of 923 at 10 minutes);
  Parakeet stays coherent but skips sentences. Time grows faster than length (attention is
  quadratic: ×10 length is ×21 for GigaAM, ×20 for Parakeet).

### Benchmarks and profiles

Numbers are in the plans' "## Result" sections.

- 2026-10-11, Parakeet V3 and Whisper turbo against Neural Engine engines: FluidAudio (ANE) is
  2.2–2.3× faster than wavo on 11–35 s at a third of the energy; WhisperKit (ANE) is 1.4–1.8×
  slower at ~15% more energy; parakeet-mlx 1.7–1.8× slower on 11–35 s. [Other
  engines](plans/20261010-other-engines.md).
- 2026-10-11, Moonshine tiny and base, baseline: fixtures exact on the first run with F16 caches;
  warm 1.7–3.4× faster than the reference's faster backend on a busy M2, faster on the RTX 4060
  but for base `dots` (228 vs 227 ms). [Moonshine plan](plans/20261011-moonshine.md)
- 2026-10-11, Windows RTX 4060, GPU timestamps around the 16×16 GEMM: the portable GEMMs were 50%
  of the kernels on turbo `jfk` and 75% on Parakeet V3 `dots`; after it attention is 74% on turbo
  (471 of 633 ms). [16×16 GEMM plan](plans/completed/20261011-nvidia-fast.md).
- 2026-10-11, Whisper variants, baseline: on the M2 8–30% faster than the reference; on Windows
  the portable encoder is 1.5–3.4× behind the reference's cooperative matrices from base on.
  [Whisper plan](plans/completed/20261010-whisper.md).
- 2026-10-10, Whisper large-v3-turbo, end of phase 12: faster on all four clips (jfk 1239–1343 vs
  1375–1415 ms); one pass over all 32 samples gives the reference's text. [Whisper
  plan](plans/completed/20261010-whisper.md). Against whisper.cpp: a tie on one or two windows, 18% faster
  on 306 s (not investigated). [Other engines](plans/20261010-other-engines.md).
- 2026-10-10, Windows RTX 4060 and Linux N100, portable kernels against the reference's CPU and
  Vulkan builds. [Windows plan](plans/completed/20261010-windows.md), [Linux plan](plans/20261010-linux.md).
- 2026-10-10, GigaAM e2e-rnnt and Parakeet V3, 2,923 dictation recordings (`make compare`): text
  the same in 99.86% / 98.9%, every difference from a near-tie; Parakeet V3 36% less compute,
  GigaAM on par (slower under 2 s, faster from 5 s). [Real recordings
  plan](plans/completed/20261010-real-recordings.md).
- 2026-10-10, Parakeet V2 and V3, end of phase 6: warm 14–35% faster than the reference, load
  ~40% below. [Parakeet V3 plan](plans/completed/20261010-parakeet-tdt-v3.md).
- 2026-10-10, Parakeet V3, first run on the V2 code path: fixtures exact with no code change;
  warm 175 / 198 / 196 ms (reference ~216 / 251 / 240), peak 804 vs 884 MiB.
- 2026-10-10, GigaAM and Parakeet V2, end of phase 5, and the same binaries on an M1.
  [Parakeet speed plan](plans/completed/20261010-parakeet-fast.md).
- 2026-10-10, Parakeet V2, profile at the start of phase 5 (GPU timestamps per dispatch kind,
  temporary tool): attention 23% of GPU time on jfk and 45% on dots (~300 ms: scalar 16×16 tiles
  through memory, position scores over all 2T − 1 positions); GEMMs 1.5–1.9 TFLOP/s; the stacked
  GEMM computed q twice; `first_depthwise` 5.7 / 18.9 ms; the F32 LSTM step 0.6 ms per token,
  ~47 GB/s, memory-bound. Passes of independent dispatches overlap, so kernels sum to more than
  the span.
- 2026-10-10, Parakeet V2, phase 4 baseline: fixtures exact on the first run, no reference
  roundings needed. [Parakeet V2 plan](plans/completed/20261009-parakeet-tdt-v2.md).
- 2026-10-09, GigaAM variants, e2e-ctc, rnnt and ctc as heads of the shared encoder: CTC logits
  as the last GEMM (rows padded to 64 at load), greedy collapse on the CPU, the charwise RNN-T on
  the same decoder. [GigaAM variants plan](plans/completed/20261009-gigaam-variants.md).
- 2026-10-09, GigaAM, state after the memory and first-call task: warm on par (ru 43.7 vs 43.7,
  ru-long 317.9 vs 310.4 ms), the first call 25–37 ms over warm (arena allocation, bind groups,
  buffer zero-init) against the reference's 10–15.
- 2026-10-09, GigaAM, profile (GPU timestamps per dispatch, each in its own pass, ~5% inflated):
  GEMM 88% of GPU time on ru, 72% on ru-long, attention 24% on ru-long. FFN GEMMs ~1.8 TFLOP/s on
  ru-long but ~0.7 on ru (FFN down has 48 workgroups at M = 113). The same 1536←768 shape takes
  26.8 ms with Q8_0 and 16.8 with F16 weights. Recording and submitting 261 dispatches < 0.5 ms.
  The first call's extra ~40 ms was wgpu running the weight uploads with the first submit.
- 2026-10-09, GigaAM, phase 1 baseline: warm on par, peak 878 / 880 / 925 vs 322 / 324 / 333 MiB.
  Portable kernels only: 85.1 / 203.1 / 808.4 vs 46.4 / 99.6 / 311.2 ms (ru-long attention 370 ms
  with the scores in memory, GEMM 425 ms). [GigaAM plan](plans/completed/20261009-gigaam-e2e-rnnt.md).

### Research branch (all weights expanded to F32)

The `research` branch before this repository: GigaAM naive 621 / 1389 / 4311 → 56.5 / 118 / 374.5
ms (1.18–1.29× slower than the reference), Parakeet V2 jfk 200.5, dots 644 ms; load 0.65 / 1.8 s
and peak 5.0 / 7.8 GB, all from expanding the 0.27 / 0.73 GB files to F32. What worked there:

- Cooperative-matrix GEMM (Metal simdgroup 8×8 via `EXPERIMENTAL_COOPERATIVE_MATRIX`, F32): 9–15×
  per GEMM over a scalar tiled GEMM. 128 threads = 4 subgroups × 32, output tile 32×32, K step 16;
  larger tiles and larger K were slower. Peak ~1.6 TFLOP/s.
- `zero_initialize_workgroup_memory: false` (see Quirks).
- `gemm_fast`: output tile M32×N64, K16, vec4 loads, 8 accumulators per subgroup: 1.17–1.96× per
  GEMM, most at small M.
- Prepacking weights at load into 8×8 fragment order: 1.02–1.26× per GEMM, ~8% end to end, +0.21 s
  load.
- Subgroup reductions: LayerNorm 2.5–2.9×, softmax 1.4–14× (microbenchmarks).
- One command buffer per Conformer block (17 submits per inference) instead of one per op.
- CPU RNN-T decoder with NEON/FMA matvec; predictor output cached across blanks.

Tried and rejected there. The first three ran on a busy machine (load average up to 30, identical
controls varying 2–3×), so their verdicts are not reliable and they may be retried:

1. CPU weight packing in local 8×8 order to speed up load: GigaAM −26%, Parakeet unclear. Moot
   once weights stay Q8 on the GPU.
2. Caching 128 small parameter uniform buffers: no visible gain. A full activation arena was not
   tried there (wavo has one).
3. Decoder: batched encoder projection (NEON 4×4) plus a 32-frame joint lookahead (what the
   reference does): inconclusive.
4. Fused online-softmax attention on cooperative matrices, 16 queries × 32 keys per workgroup:
   −28% peak memory, +12.6% time on ru-long. Rejected. A larger query tile was not tried.
5. Whisper: compensated (Neumaier) summation in GEMM/LayerNorm to bit-match the CPU reference.
   Correct but several times slower. Never do this; see agents.md → Correctness.
