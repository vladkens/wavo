# Performance notes

Apple M2 (8 CPU / 24 GiB), macOS 27, wgpu 30.0.1 Metal. Times are warm medians without model
load unless stated. GigaAM clips: `ru` 4.5 s, `ru-short` 11 s, `ru-long` 33.8 s. Parakeet clips:
`jfk` 11 s, `dots` 35.3 s.

## Targets: transcribe.cpp Metal (commit 5bb2deb, same Q8_0 GGUF)

| Model | Warm | Load | First call | Peak footprint |
|---|---|---|---|---|
| GigaAM v3 e2e-rnnt | 44 / 103 / 312–326 ms | 0.13–0.16 s | 59–70 / 103–118 / 319–331 ms | 322 / 324 / 333 MiB |
| Parakeet TDT V2 | jfk 203 ms, dots 820 ms | 0.26 s | — | — |

GigaAM measured 2026-10-09 with `transcribe-bench` (warm: `--warmup 1 --iters 10`, median of
`wall_ms`; first call: fresh process, `--warmup 0 --iters 1`; footprint: `/usr/bin/time -l` of
that process). Ranges are repeated runs on a machine with load average 3–5.

The same C++ build on F32-expanded weights: GigaAM 45 / 110 / 368 ms. So Q8 storage itself buys
0–14% warm speed. Its big wins are memory and load time.

Requirement: GigaAM e2e-rnnt warm must beat these numbers; other models at least match them. Load,
first call and memory should beat them too.

The reference is not an F32 pipeline. Its Metal matmul (`kernel_mul_mm_q8_0_f32`) dequantizes
Q8_0 weights to half, casts activations to half, multiplies with `simdgroup_half8x8` and
accumulates in F32.

## Current state: GigaAM v3 e2e-rnnt (phase 1 baseline)

| Clip | Warm median / min | First call | Load | Peak footprint |
|---|---|---|---|---|
| ru | 45.4 / 45.1 ms | 71–84 ms | 69–76 ms | 878 MiB |
| ru-short | 102.1–104.0 / 101.6 ms | 131–135 ms | 68–72 ms | 880 MiB |
| ru-long | 339.5–349.5 / 333.5 ms | 372–388 ms | 67–83 ms | 925 MiB |

`wavo bench` and `/usr/bin/time -l wavo MODEL AUDIO`, back to back with the reference runs above.
Warm is on par for ru and ru-short and 4–10% slower on ru-long; load is ~1.8× faster; the first
call and peak memory are worse.

How it runs: CPU log-mel (rustfft), then the whole encoder as one compute pass over an arena
sized for the longest input so far, with bind groups cached per dispatch; weights stay Q8_0/F16
on the GPU. Fast kernels (cooperative-matrix GEMM with Q8_0/F16 dequantized while staging,
barrier-free flash attention, one subgroup per row for LayerNorm and the conv module) each turn
on after a probe dispatch; portable WGSL kernels run otherwise. The joint's encoder projection is
the last GEMM; the RNN-T loop (LSTM, joint, argmax) runs on one CPU thread.

Profile on ru-long (by skipping dispatches, so approximate): GEMM ~200 ms (~1.8 TFLOP/s),
attention ~85 ms, LayerNorm/conv/im2col ~8 ms, decoder 38.5 ms (reference 19 ms), mel 8 ms
(reference 13 ms). Peak memory: the whole GGUF read into memory, the Q8_0 repack copies and
wgpu's upload staging buffers all exist next to the GPU weights during load.

## Research history: `research` branch (all weights expanded to F32)

| Model | Warm | vs C++ | Load | RSS after load | Peak footprint |
|---|---|---|---|---|---|
| GigaAM | 56.5 / 118 / 374.5 ms | 1.18–1.29× slower | 0.65 s | ~0.9 GB | ~5.0 GB |
| Parakeet V2 | jfk 200.5, dots 644 ms | on par / 21% faster | 1.8 s | ~2.4 GB | ~7.8 GB |

GigaAM load breakdown (ms): GGUF read 25, dequant to F32 79, GPU upload 227, weight packing 209,
other 86. Parakeet: upload 633, packing 590, dequant 217 of 1755. All of this comes from expanding
to F32; the GGUF files are only 0.27 / 0.73 GB.

### What worked (research: naive 621 / 1389 / 4311 ms → 56.5 / 118 / 374.5 ms)

1. Cooperative-matrix GEMM (Metal simdgroup 8×8 via wgpu `EXPERIMENTAL_COOPERATIVE_MATRIX`,
   F32): 9–15× per GEMM over a scalar tiled GEMM. 128 threads = 4 subgroups × 32; output tile
   32×32, K step 16. Larger tiles and larger K were slower. Peak measured ~1.6 TFLOP/s.
2. `PipelineCompilationOptions { zero_initialize_workgroup_memory: false }`: Naga otherwise clears
   shared memory from a single lane.
3. `gemm_fast`: output tile M32×N64, K16, vec4 loads, 8 accumulators per subgroup: another
   1.17–1.96× per GEMM (largest gain at small M).
4. Prepacking weights at load into 8×8 fragment order (no B transpose per tile): 1.02–1.26× per
   GEMM, ~8% end to end, +0.21 s load on CPU.
5. Subgroup reductions: LayerNorm 2.5–2.9×, softmax 1.4–14× (microbenchmarks).
6. One command buffer per Conformer block (17 submits per inference) instead of one per op.
7. CPU RNN-T decoder with NEON/FMA matvec; predictor output cached across blanks.

### Profile of the research state (GigaAM; GPU timestamps, ms per inference, ru / ru-short / ru-long)

- GEMM 48 / 102 / 348 = 68–79% of GPU time. FFN up+down alone 29 / 59 / 215.
- Attention (scores + softmax + values) 4 / 13 / 104: grows ~T², 20% at 34 s.
- Elementwise passes (SiLU, residual adds, GLU, RoPE) ~5 / 12 / 43: can be fused into GEMM
  epilogues.
- GPU idle gaps inside the encoder: 12 / 23 / 77 ms of a 69 / 147 / 562 ms span (~15%).
- CPU enqueue 21–40 ms per inference: 470 fresh output buffers, 470 uniform buffers and bind
  groups, 17 submits. 0.25 / 0.7 / 3.2 GB of buffers allocated per inference: this is the 5 GB
  peak footprint.
- CPU decoder 7 / 19 / 52 ms (10–14% of wall). Frontend 1–11 ms (ignore).

### Tried and rejected in research

Experiments 1–3 ran on a busy machine (load average up to 30, identical controls varying 2–3×).
Their verdicts are not reliable, so they may be retried.

1. CPU weight packing in local 8×8 order to speed up load: GigaAM −26%, Parakeet unclear.
   Moot once weights stay Q8 on the GPU.
2. Caching 128 small parameter uniform buffers: no visible gain. A full activation arena was never
   tried.
3. Decoder: batched encoder projection (NEON 4×4) plus 32-frame joint lookahead (what C++ does):
   inconclusive.
4. Fused online-softmax attention, coop matrices, 16 queries × 32 keys per workgroup (clean
   measurement): −28% peak memory, +12.6% time on ru-long. Rejected. A larger query tile was not
   tried.
5. Whisper: compensated (Neumaier) summation in GEMM/LayerNorm to bit-match the CPU reference.
   Correct but several times slower. Never do this; see agents.md → Correctness.

## Generic GPU lessons (from the Whisper work)

- Bound in-flight work: queuing all 32 encoder blocks at once kept every intermediate alive
  (22.7 GB footprint). Waiting every 4 blocks cut it to 9 GB and was faster.
- Dynamic indexing of private arrays in WGSL is slow on Metal. Explicitly unrolled scalar
  accumulators made a GEMM 2× faster.
- M=1 (decoder step, output head) needs a dedicated matvec kernel. A 32-row GEMM tile with one
  useful row was 2× slower (16.6 → 8.6 ms on a 51866×1280 head).

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

## Where to look for kernel ideas

- `3rd/transcribe.cpp/ggml/src/ggml-metal/` is what the reference runs on Apple Silicon:
  `kernels/mul_mm.metal` (matmul), `kernels/mul_mv.metal` (matrix-vector), `kernels/fa.metal`
  (flash attention), `ggml-metal-fusion.cpp` (which ops it fuses), `ggml-metal-tuning.cpp`
  (per-device parameters).
- `3rd/transcribe.cpp/ggml/src/ggml-webgpu/wgsl-shaders/` is ggml's WebGPU backend, written in
  WGSL like ours, with Q8_0 `mul_mat`, `mul_mat_vec` and flash attention. transcribe.cpp doesn't
  use this backend, but its shaders can be adapted directly (research did this for the Whisper
  output head).

## Not tried yet (GigaAM, from the phase 1 baseline)

The research list (Q8_0 on the GPU, arena, epilogue fusion, flash attention, decoder lookahead,
F16 operands) is done or rejected; see the Log.

1. Peak memory (0.9 GB vs 0.33 GB): read tensors from the file per block instead of the whole
   GGUF, repack Q8_0 straight into the mapped upload buffer, and skip wgpu's staging copy
   (`MAPPABLE_PRIMARY_BUFFERS` on unified memory). Should also cut load.
2. First call (25–40 ms over warm, reference 10–15 ms): find the split between arena allocation,
   bind group creation and wgpu's lazy zero-fill of new buffers; consider sizing the arena for
   ~30 s at load.
3. GEMM (~60% of GPU time): tile shape by M (more workgroups for short clips, 64×64 for long
   ones), Q8_0 prepacked in fragment order at load, double-buffered staging.
4. Attention (~85 ms on ru-long): 64-key blocks, q fragments in registers via a kernel
   specialized for head_dim (override constant), skipping the rescale when no row max moved.
5. Decoder (38.5 vs 19 ms): a 4-row dot written with NEON intrinsics; LLVM vectorizes the
   portable form across rows and loses (see Log).

## Log

Add entries here, newest first: date, model, idea, before → after (median, A/B/A), verdict, why.

- 2026-10-09, GigaAM, phase 1 baseline recorded in "Current state" and "Targets" above. Peak
  footprint wavo 878 / 880 / 925 MiB vs reference 322 / 324 / 333 MiB (max RSS 609 / 602 / 620
  vs 356 / 357 / 366 MiB).
- 2026-10-09, GigaAM, decoder: removed the helper thread (a library must not busy-wait a core,
  and a panic in the main loop would hang `thread::scope` on it). Decoder 26 → 38.5 ms on
  ru-long; warm ru-long ~+12 ms. Single-threaded replacements, all slower than the plain matvec
  (38.7 ms, direct timer, A/B back to back):
  - 4 joint rows per pass over z, so each z chunk loads once for 4 rows (exact per row):
    generic `dots<const R>` 94 ms (accumulators spilled), index loops 43.5 ms (bounds checks per
    row, z reloaded per row), zipped row iterators 61.6 ms (LLVM's SLP vectorizer transposes
    across the 4 rows with shuffles). Rejected; needs NEON intrinsics to win.
  - Spans of frames sharing one predictor output, rows outer and frames inner (the reference's
    lookahead): the earlier 8-frame try was 20 ms slower. Why: the matvec is bound by L1 loads
    and FMAs, not by L2 traffic, so keeping a weight row in L1 across frames saves nothing unless
    the loaded registers are reused across frames (same codegen problem as above), while the
    frames after each emission are rescored (estimated ~20% more dots on ru-long: 143 tokens
    over 845 frames). Not retried.
- 2026-10-09, GigaAM, state after Task 2 (back to back, warm median / min, first call, load):
  wavo ru 43.7 / 43.5, 80.7, 75 ms; ru-short 97.6 / 97.3, 122.2, 69 ms; ru-long 317.9 / 316.6,
  346.1, 71 ms. Reference ru 43.7 / 43.3, 58.9, 123 ms; ru-short 99.6 / 98.6, 103.0, 124 ms;
  ru-long 310.4 / 308.3, 318.9, 128 ms. The first call is 25–37 ms over warm (arena allocation,
  bind groups, buffer zero-init); the reference's only 10–15 ms.
- 2026-10-09, GigaAM, decoder: a helper thread scores the upper half of the joint vocabulary,
  spinning on an atomic step counter for the duration of `decode`. Decoder 38 → 26 ms on
  ru-long (direct timer). Warm B/A/B: ru-long 330.5 / 343.1 / 334.5, ru-short A/B/A 107.9 /
  102.7 / 106.8 ms. Kept; bit-exact with one thread.
- 2026-10-09, GigaAM, GEMM F16 operands (a and dequantized W rounded to f16 in shared memory,
  f16×f16→f32 cooperative matrices, `SHADER_F16`): fixtures unchanged but slower, ru-long 353.1
  vs 329.3, ru 45.9 vs 45.4 ms. Rejected.
- 2026-10-09, GigaAM, GEMM K step 32 instead of 16 (one Q8_0 block per step, half the barriers):
  ru-long B/A/B 347.8 / 330.0 / 347.2 ms. Rejected, as in research.
- 2026-10-09, GigaAM, flash attention without workgroup barriers: each subgroup walks all keys
  on its own, loads k and v fragments straight from memory and syncs only with
  `subgroupBarrier` around its softmax. ru-long B/A/B 331.3 / 357.9 / 329.7 ms (attention ~111
  → ~85 ms). Kept. Reading rows of qk/v past t (masked) needs them finite: the arena's are.
- 2026-10-09, GigaAM, flash attention variants on the barrier version, ru-long: q tile staged
  once in shared memory (arrays sized for head_dim ≤ 48) 329.0 vs 330.4, no gain, rejected;
  q fragments in an array of cooperative matrices 394.2 vs 375.9 (spills), rejected; shared
  memory 32 → 26 KB 374.4 vs 376.5, no gain, rejected; k/v staging loops without division by
  head_dim 370.4 vs 377.5 / 375.9, kept until the barrier-free kernel removed staging; softmax
  with lane = key and subgroupMax/Add instead of 2 lanes per row (16-way bank conflicts) 368.9
  / 370.1 / 368.4, no change alone, kept in the barrier-free kernel. An ablation without the
  softmax/rescale phase saved 47 of 111 ms: the serial phase between barriers was the cost.
- 2026-10-09, GigaAM, mel filterbank with the vectorized dot: mel 25 → 8 ms on ru-long. Kept.
- 2026-10-09, GigaAM, decoder dot: 32 accumulators 43 → 48 ms (the 32 scalar adds of the final
  sum dominate a 320-long dot); 16 FMA accumulators summed pairwise 43 → 38.5 ms. Kept.
- 2026-10-09, GigaAM, decoder: joint scored for spans of 8 frames per weight pass while the
  predictor output is unchanged. ru-long A/B/A 397.5 / 417.2 / 397.9 ms. Rejected: rescoring
  the frames after each token outweighs the cache reuse.
- 2026-10-09, GigaAM, GEMM weight type as a pipeline-overridable constant (one pipeline per
  type) instead of a runtime switch: fast ru A/B/A 56.9 / 50.0 / 56.8, ru-long 431.4 / 396.2 /
  429.7 ms. Kept. Portable path neutral (119.2 / 116.9 / 120.0). The one-entry portable GEMM is
  slower than Task 1's three-entry one (ru 85 → 118 ms); cause not found, fallback only.
- 2026-10-09, GigaAM, load: Q8_0 repack by per-block memcpy instead of a byte iterator (encoder
  upload 181 → 77 ms), the 16 blocks repacked and uploaded from 16 threads (77 → 28 ms), GGUF
  read in 8 parallel chunks (36 → 20 ms). Load 233 → 60–90 ms. Kept. Measured with temporary
  timers, 2–3 runs each, machine load average 3–5.
- 2026-10-09, GigaAM, fast paths: cooperative-matrix GEMM (research `gemm_fast` layout: 32×64
  tile, K16, F32 8×8, Q8_0/F16 dequantized while staging W), flash attention (64 queries × 32
  keys per workgroup, online softmax, output accumulator in shared memory), one subgroup per row
  for LayerNorm(+RoPE) and conv_glu. Warm portable → fast, A/B/A: ru 118.0 / 56.4 / 118.4,
  ru-short 280.2 / 128.7 / 283.1, ru-long 1046 / 427 / 1008 ms. Kept. Profile after it
  (ru-long, by skipping dispatches): GEMM ~245 ms (~1.5 TFLOP/s), attention ~109 ms, other GPU
  ~8 ms; CPU mel 25 ms, decoder 43 ms.
- 2026-10-09, GigaAM, Task 1 baseline (portable kernels, one compute pass per call): warm 85.1 /
  203.1 / 808.4 ms, load 0.23 s. Reference warm 46.4 / 99.6 / 311.2 ms. Profile (ru-long):
  attention 370 ms (scores matrix in memory), GEMM 425 ms, LayerNorm/conv 55 ms.
