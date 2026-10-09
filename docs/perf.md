# Performance notes

Apple M2 (8 CPU / 24 GiB), macOS 27, wgpu 30.0.1 Metal. Times are warm medians without model
load unless stated. GigaAM clips: `ru` 4.5 s, `ru-short` 11 s, `ru-long` 33.8 s. Parakeet clips:
`jfk` 11 s, `dots` 35.3 s.

## Targets: transcribe.cpp Metal (commit 5bb2deb, same Q8_0 GGUF)

| Model | Warm | Load | First call | Peak footprint |
|---|---|---|---|---|
| GigaAM v3 e2e-rnnt | 44–45 / 102–103 / 312–326 ms | 0.13–0.17 s | 48–73 / 103–124 / 319–353 ms | 322 / 324 / 334 MiB |
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

## Current state: GigaAM v3 e2e-rnnt (phase 2 done)

| Clip | Warm median / min | First call | Load | Peak footprint |
|---|---|---|---|---|
| ru | 43.4–43.5 / 43.1 ms | 47.9–48.7 ms | 78–109 ms | 295 MiB |
| ru-short | 96.0–96.1 / 94.8 ms | 100.7 ms | 82–100 ms | 299 MiB |
| ru-long | 297.3–297.7 / 295.3 ms | 304.7–305.2 ms | 81–99 ms | 320 MiB |

`wavo bench -n 20` and `/usr/bin/time -l wavo MODEL AUDIO`, each clip as wavo / reference / wavo
back to back (load average 3–4). The reference in the same runs: warm 44.7 / 102.1 / 321.5 ms
(min 43.7 / 100.2 / 315.9), first call 69–73 / 111–119 / 335–353 ms, load 127–167 ms, peak
footprint 322 / 324 / 334 MiB. So warm is 3% / 6% / 7.5% faster with the two wavo runs within
0.2%, and first call, load and footprint are below the reference on all clips. The first run
after a shader change loads in 0.2–0.4 s while Metal compiles and caches the new shaders.

How it runs: the GGUF header is read at load and each tensor is streamed from the file straight
into weight buffers that are mapped on unified memory (no staging copy); weights stay Q8_0/F16
on the GPU. Load ends with an encoder run on 8 silent frames to pay the GPU's first use of
pipelines and weights. Per call: CPU log-mel (rustfft, one frame at a time), then the whole
encoder as one compute pass over an arena sized for the longest input so far, with bind groups
cached per dispatch. Fast kernels (cooperative-matrix GEMM with Q8_0/F16 dequantized while
staging and rounded to f16 like the reference's, barrier-free flash attention, one subgroup per
row for LayerNorm and the conv module) each turn on after a probe dispatch; portable WGSL
kernels run otherwise. The head's linear layer is the last GEMM: the joint's encoder projection
for RNN-T, the logits (rows padded to 64) for CTC. The RNN-T loop (LSTM, joint, argmax) runs on
one CPU thread with NEON tiles that score two frames per pass over the joint weights; CTC is an
argmax and collapse per frame.

### Profile (2026-10-09, GPU timestamps per dispatch, warm, ms per call)

Measured with a temporary tool (removed): each dispatch in its own compute pass with begin/end
timestamps, which inflates the GPU span ~5%; CPU timers in the normal single-pass mode.

| | ru (T=113) | ru-long (T=845) |
|---|---|---|
| GPU kernels total | 42.2 | 306 |
| FFN down, 768←3072 Q8_0 + residual (32×) | 12.3 | 71.6 |
| FFN up, 3072←768 Q8_0 + SiLU (32×) | 11.2 | 71.0 |
| q·k projection, 1536←768 Q8_0 (16×) | 4.2 | 26.8 |
| conv pointwise1, 1536←768 F16 (16×) | 2.9 | 16.8 |
| v projection, 768←768 Q8_0 (16×) | 2.4 | 13.6 |
| attention out, 768←768 Q8_0 + residual (16×) | 1.7 | 9.5 |
| conv pointwise2, 768←768 F16 + residual (16×) | 1.6 | 8.8 |
| pre-encode convs + joint projection | 0.6 | 3.2 |
| attention (16×) | 2.3 | 73.7 |
| LayerNorm 64×, conv_glu 16×, LayerNorm+RoPE 16×, im2col 2× | 2.6 | 11.2 |
| CPU: mel / encode + submit / GPU wait / readback | 1.0 / 0.3 / 40 / 0.01 | 7.7 / 0.4 / 285 / 0.05 |
| CPU: decoder | 5.0 (ref 2.6) | 37.6 (ref 19) |
| First call: GPU wait | 83 (+43) | 324 (+40) |

- GEMM is 88% of GPU time on ru and 72% on ru-long; attention is 24% on ru-long.
- FFN GEMMs run at ~1.8 TFLOP/s on ru-long but ~0.7 TFLOP/s on ru: at M=113 FFN down has only
  48 workgroups (N/64 × ⌈M/32⌉) for a 3072-long K loop.
- The same 1536←768 shape takes 26.8 ms with Q8_0 weights and 16.8 ms with F16: dequantizing
  while staging W costs ~60% on top.
- Recording and submitting the 261 dispatches costs < 0.5 ms; bind groups are reused.
- At that point the first call's extra ~40 ms was GPU time: wgpu ran the weight uploads (staging
  → GPU, ~285 MB) with the first submit, not at load. Peak memory held the whole GGUF read into
  memory, the Q8_0 repack copies and wgpu's staging buffers next to the GPU weights. Both are
  fixed; see the Log.

## Current state: other GigaAM v3 variants (phase 3)

Warm median, first call and peak footprint per clip (ru / ru-short / ru-long), wavo vs
`transcribe-bench` on the same GGUF, measured back to back as above (load average 4–7, a
Spotlight indexer on one core):

| Variant | Warm wavo | Warm reference | First call wavo | First call reference | Peak wavo | Peak reference |
|---|---|---|---|---|---|---|
| e2e-ctc | 42.0 / 85.1 / 273.4 ms | 50.9 / 95.3 / 318.5 ms | 48 / 89 / 274 ms | 54 / 101 / 323 ms | 289 / 293 / 314 MiB | 314 / 315 / 323 MiB |
| rnnt | 42.3 / 95.5 / 295.4 ms | 43.3 / 98.2 / 311.3 ms | 48 / 99–102 / 301 ms | 64 / 100–113 / 305 ms | 292 / 298 / 319 MiB | 319 / 321 / 329 MiB |
| ctc | 39.4 / 84.7 / 268.3 ms | 41.6 / 92.2 / 295.9 ms | 43 / 88 / 273 ms | 60 / 95 / 316 ms | 288 / 292 / 314 MiB | 313 / 315 / 323 MiB |

Load with the file in the page cache: wavo 76–126 ms, reference 120–186 ms (a cold first read of
a new GGUF took wavo 0.19–0.47 s). The CTC variants have no decoder loop; their reference reads
the encoder output back and runs the head and a log-softmax on the CPU.

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

## Not tried yet (GigaAM, after phase 2)

The phase 1 list (GEMM tile shape, split-K, prefetched staging, attention blocks and rescale
skip, NEON decoder) and the research list are done or rejected; see the Log.

1. Attention: q fragments in registers in a kernel specialized for head_dim 48 (only constant
   loop bounds were tried). Its softmax (~17 of 50 ms on ru-long) still runs 32 subgroup
   reductions per 64-key block.
2. q·k and v projections as one GEMM (concatenated weights, one A buffer holding `yr | y`): one
   dispatch less per block and more workgroups at small M.
3. Q8_0 prepacked in fragment order: low priority, W traffic is not the GEMM's limit (see Log).
4. Decoder: `Wx · embed[token]` depends only on the token, so a per-call cache would skip half
   of the LSTM step (~68 µs) for repeated tokens.

## Log

Add entries here, newest first: date, model, idea, before → after (median, A/B/A), verdict, why.

- 2026-10-09, GigaAM variants, GEMM rounds dequantized W to f16, as the reference's
  `kernel_mul_mm` does. Without it e2e-ctc ru-short emits one token a frame late: at frame 116
  the reference has 13.4252 for the token and 13.4183 for blank, wavo 13.4093 and 13.4172 (mean
  |Δ logits| 0.0026). With a and W both rounded: 13.4250 vs 13.4171, mean 0.0016, all argmaxes
  equal. W only: all four models' fixtures exact (e2e-rnnt unchanged), warm e2e-rnnt ru 43.7 /
  43.5 vs 43.8 / 43.5 ms without, ru-long B/A/B 296.0 / 295.2 / 296.4 ms. Kept. Rounding a as
  well: +1.0 ms on ru (44.9 / 44.5 ms), no fixture needs it, rejected. Rounding with integer ops
  instead of `pack2x16float`: +2.4 ms (W) / +4 ms (both) on ru, rejected.
- 2026-10-09, GigaAM variants, e2e-ctc, rnnt and ctc as heads of the shared encoder: CTC logits
  as the last GEMM (257 or 34 rows padded to 64 at load: `Gpu::linear` pads any layer with zero
  rows, the encoder drops the padding columns at readback), greedy collapse on the CPU; the
  charwise RNN-T reuses the decoder (34 classes). Bench in "Current state" above.
- 2026-10-09, GigaAM, small encoder experiments on ru (encoder wall time with a temporary timer;
  the machine drifts by ±1 ms over minutes, so only back-to-back runs count). All rejected:
  - 48 extra one-workgroup im2col dispatches: 39.9–40.4 vs 39.8–40.5 ms, so a dispatch boundary
    costs little. (Earlier runs, 48 extra one-row LayerNorms +1.4 ms and 48 LayerNorms fewer
    −1.2 ms, can't be told apart from the drift.)
  - Each block's output norm fused with the next block's first norm (one kernel, the first
    output recomputed per element, −15 dispatches): A/B/A 38.8–38.9 / 38.9–39.3 / 38.9–39.1 ms.
  - LayerNorm statistics with eight loads issued before their sums: warm A/B/A 44.4 / 44.2 /
    44.1 ms. LayerNorm rows held in a 32-entry register array: encoder +1.5 ms (likely spilled).
  - GEMM skipping the MMAs of a subgroup's lower 8 rows when they are all past m (6% of the
    MMAs at M = 113): the fast GEMM no longer passes its probe (a branch on the subgroup id
    around cooperative-matrix calls).
- 2026-10-09, GigaAM, attention softmax with two lanes per row (each lane loops over 32 keys of
  one parity, one `subgroupShuffleXor` per reduction instead of 16 × 2 subgroup reductions):
  rows 66 floats apart (no bank conflicts, but rows not 16-byte aligned) 98.5 ms, 72 apart
  (aligned, 4-way conflicts) 55.5 ms; scores and accumulator column-major with 16-float columns
  (aligned and conflict-free, lanes r and r + 16 on one row) 56 ms; vs 50.5 ms (GPU timestamps,
  ru-long). Rejected.
  Cooperative-matrix loads and stores on rows that are not 16-byte aligned cost ~2×. Ablations
  of the kept kernel: without the softmax 33 ms, without P·V 37.6 ms, so scores ~20, softmax
  ~17 and P·V ~13 ms of the 50.5.
- 2026-10-09, GigaAM, frontend: one frame at a time in a reused buffer (no 8.6 MB spectrum
  allocation on ru-long, no div/mod per sample) and each mel filter's `dot` only over its
  nonzero bins widened to 16-bin chunks (same lanes, the skipped terms are exact zeros). Mel
  bit-identical on all fixture clips (temporary bitwise check). Frontend (direct timer) ru
  1.0 → 0.57, ru-short 2.5 → 1.43, ru-long 7.6 → 4.4 ms (the rest is the FFT ~3 ms and `ln`).
  Warm ru-long A/B/A 307.8 / 303.2 / 307.5 ms. Kept.
- 2026-10-09, GigaAM, GEMM limits (GPU timestamps, ru-long, timing-only ablations with wrong
  outputs; FFN down / FFN up / q·k, normally 72–74 / 70–72 / 26–28 ms): W loads all from one
  2 KB region 71.3 / 67.6 / 27.8; a loads all from one region 73.8 / 73.1 / 25.8; staging (loads
  and shared-memory stores) only on the first K step 60.9 / 61.2 / 22.8; MMAs only on the first
  K step 37.3 / 30.6 / 11.3. So device memory traffic costs nothing, staging ~15%, and the rest is
  the on-chip loop; FFN up runs at ~1.8 TFLOP/s, about half the M2's FP32 peak. A 64×64 tile
  (each subgroup 32×32, 16 accumulators, 0.5 fragment loads per MMA instead of 0.75): 73.3 /
  70.3 / 26.7 ms, no gain. Rejected. Prepacking W in fragment order (plan item) not tried: W
  traffic is not the limit.
- 2026-10-09, GigaAM, GEMM split-K for short clips (timing-only ablation, outputs wrong: FFN down
  dispatched with 4 K-slices as `wg.z`, no reduction): FFN down at M = 113 (ru) 11.7–12.4 vs
  12.3–12.6 ms (GPU timestamps); only at M = 2 (load warm-up) it drops, 13.8 → 9.6 ms. At
  M = 113 its 48 workgroups already run at the throughput of FFN up's 192 (~1.55 TFLOP/s on the
  128 padded rows), so it is not short of workgroups. Rejected without the reduction pass.
- 2026-10-09, GigaAM, GEMM register prefetch (the next K step's a and raw W words load into
  registers before this step's multiplies): FFN down on ru 12.4–12.7 vs 12.3–13.4 ms; ru-long
  A/B/A noise (A 308.9 / 297.5 / 311.0, B 313.2 / 298.6 / 306.4 ms). Rejected.
- 2026-10-09, GigaAM, decoder with NEON tiles (`std::arch::aarch64`; plain `dot` elsewhere): a
  tile computes 2 rows × 2 frames or 4 rows × 1 frame, each dot with `dot`'s 16 lanes and
  pairwise sum, so every value is bit-identical. The joint scores frames in pairs against one
  predictor output (the frame after a token is scored again, ~8% extra). Decoder (direct timer)
  ru 5.0 → 3.6, ru-short 11.7 (4 × 1 tiles alone) → 10.3, ru-long 38.5 → 27.2 ms; on ru-long
  4 × 1 tiles alone gave 31.3 ms and spans of 4 frames with 1 × 4 tiles 29.8 ms (18% rescored).
  Warm A/B/A: ru 45.7 / 44.1 / 45.8, ru-short 101.6 / 97.5 / 101.9 (another B 102.2, A 104.8),
  ru-long 315.2 / 306.2 / 315.8 ms. Fixtures exact. Kept. ru-long after: pair spans 13.9 ms
  (458), LSTM + predictor 9.8 ms (143 steps, 3.2 MB of f32 weights each, ~57 GB/s from L2),
  single-frame joints 3.2 ms. The reference spends 19 ms (Accelerate sgemm on 32-frame spans).
  The GGUF stores these weights as Q8_0 (expanded to f32 at load; not tried in-register).
- 2026-10-09, GigaAM, attention specialized for head_dim 48 (constant instead of the param, so
  the d and c loops are fixed): attention 49.8 vs 48.9 ms (GPU timestamps, ru-long). No gain;
  rejected.
- 2026-10-09, GigaAM, attention in 64-key blocks (scores in two 32-key halves, softmax with two
  keys per lane, one output load/store and one rescale check per 64 keys): attention 56.4 → 48.9
  ms; warm ru-long A/B/A 322.3 / 316.6 / 321.9 ms, ru-short B/A/B 102.0 / 102.5 / 102.3 ms.
  Kept.
- 2026-10-09, GigaAM, attention skips the output rescale when a row's running max didn't move
  (exact: the factor would be 1): attention 71.4 → 56.4 ms; warm ru-long A/B/A 337.1 / 322.1 /
  337.4 ms. Fixtures exact. Kept.
- 2026-10-09, GigaAM, Q8_0 GEMM cost (GPU timestamps, ru-long, timing ablations whose outputs
  were wrong). q·k 1536←768 Q8_0 takes 27.2 ms vs 16.7 for the F16 pointwise1 of the same
  shape, but: without the per-row scale lookup q·k 25.6 ms (FFN 72.2 / 70.7 → 69.5 / 69.0);
  without the int8 unpack 25.4 ms; with the q·k weights stored as F16 ~25 ms; reading `y`
  instead of the RoPE output 27.1 ms. So dequantizing Q8_0 costs ~8% (5% the scale lookup), and
  the remaining gap depends on where the GEMM runs in the block: attention-out (Q8_0) takes 9.0
  ms like pointwise2 (F16, 8.8 ms) while v (Q8_0, same shape as attention-out) takes 10–12.5 ms.
  Per-dispatch timestamps with one pass per dispatch likely charge part of the previous kernel's
  write-back to the next one. Nothing to keep; scales next to the data would add ~12 MB, which
  the ru-long footprint margin doesn't allow, for ≤ 5%. The load-time warm-up profile (M = 2)
  shows FFN down at 0.37 ms per dispatch: only 12 workgroups each walk 192 K steps.

- 2026-10-09, GigaAM, first call: warm-up at load, an encoder run on 8 silent frames. Phase 1
  build (A) vs this (B), back to back: first call ru 92.1 / 50.8 / 71.9 / 51.1 ms, ru-short
  146.6 / 110.0 / 131.2 / 108.5 ms, ru-long 369.9 / 348.1 / 369.4 / 348.9 ms; load 65–68 →
  77–81 ms. Kept. How it was found: with all weights flushed at load (temporary
  `queue.submit([])` + wait) the first call fell to 53 ms and load rose to 96 ms, so the cost
  was wgpu's deferred staging copies; with mapped weight buffers those are gone, but the first
  GPU use of pipelines and weights still costs ~41 ms of GPU time even on 8 frames, of which
  20–30 ms showed in the first call. Encode + submit were < 1 ms either way.
- 2026-10-09, GigaAM, first call: arena allocated for 30 s at load (warm-up on it) instead of on
  demand: ru 51.5 / 50.1 ms, ru-short 120.6 / 108.4 ms, no better than the on-demand arena and
  +26 MB on short clips. Rejected.
- 2026-10-09, GigaAM, peak memory: the GGUF header is read alone and every tensor is streamed
  from the file in 278 KB pieces straight into weight buffers created mapped with
  `MAPPABLE_PRIMARY_BUFFERS` (unified memory, no staging copy); Q8_0 scales stay f16 (exact,
  −12 MB); the arena drops its im2col and v buffers (`h`, `qk` and `y` take those roles, −12 MB
  on ru-long). Peak footprint 920 / 922 / 961 → 307 / 318 / 346 MB (reference 338 / 340 / 349
  MB); max RSS 312 / 317 / 326 MB. Warm unchanged: shared weight buffers vs private ones
  334.6 vs 334.0 / 335.6 ms on ru-long. Kept. Skipping the unused portable pipelines saved only
  0.5 MB: rejected.
- 2026-10-09, GigaAM, profile with GPU timestamps per dispatch kind (temporary tool, removed):
  see "Profile" above.

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
