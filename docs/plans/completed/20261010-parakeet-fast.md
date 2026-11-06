# Parakeet TDT V2, fast

Done: 2026-10-10, commit 5c15289

Roadmap phase 5. Follow `agents.md`; read `docs/perf.md` first and log every attempt there.

## Goal

On the M2, against `transcribe-bench` on the same GGUF and WAV, measured A/B/A back to back:

- Warm median at least on par with the reference on `jfk`, `dots` and `jobs-silence`, and faster
  where possible; on `dots` also below the reference's warm minimum.
- First call (fresh process) and peak footprint (`/usr/bin/time -l`) no worse than the reference
  on any clip.
- Load stays below the reference.
- Fixtures stay exact, Parakeet and GigaAM. If a change alters them, stop and show the user the
  diff.
- GigaAM e2e-rnnt gets no slower: a change to shared core (`src/gpu/`, `src/conformer.rs`,
  `src/cpu.rs`) is also measured A/B/A on `ru` and `ru-long`.
- Core + `src/parakeet/` stays within ~3k lines of Rust and WGSL, tests excluded.

Baseline (phase 4): warm 181 / 756–763 / 94 ms vs 209–213 / 799–821 / 95–97 ms (reference
minimum 192 / 702–709 / 94 ms), first call 186–189 / 764–765 / 99 ms vs 262–284 / 720–926 /
110 ms, peak 810 / 877 / 800 MiB vs 820 / 834 / 815 MiB, load 0.16–0.19 s vs 0.26 s.

## Tasks

### Task 1: measure precisely

- [x] GPU timestamps per dispatch kind and CPU timers for mel, encode, GPU wait, readback, LSTM
      steps and joint, as a temporary local tool that is not kept. Record the jfk and dots
      profiles in `docs/perf.md`; the order of the tasks below follows from them.

### Task 2: relative attention for head_dim 128 (~300 of 638 ms GPU on dots)

- [x] Flash kernel with relative positions: a subgroup's 16 queries walk 64-key blocks; per block
      the position term `(q + v)·P` is computed only for the 79 positions the block needs (the
      second 8 queries use the first's fragments shifted by one), added to the scores in shared
      memory, then online softmax and P·V as in the head_dim ≤ 64 kernel. Fit head_dim 128 in
      32 KB of shared memory (subgroups per workgroup, accumulator layout). Probe, unit test on
      small odd shapes; the portable path stays as is.
- [x] `ps` and `s` leave the arena on the fast path (−18 MiB on dots, and no [8][T][2T − 1] at
      5 min).

### Task 3: memory

- [x] q computed once: the qkv GEMM computes `[q | k | v]` and its epilogue also writes q + v
      next to q + u, exact as now (−25.5 MiB of weights, −1/4 of that GEMM).
- [x] Subsampling in time chunks: conv0 + conv2, conv3 and conv5 per chunk of output frames
      through two small scratch buffers, so `h` and `qk` shrink to the blocks' size (−38 MiB on
      dots, −14 MiB on jfk). `first_depthwise` (5.7 / 18.9 ms) with weights as [tap][ch] and
      several pixels per thread.
- [x] Peak footprint ≤ the reference on all three clips; if not, the decoder weights stay Q8_0
      and are dequantized in registers (bit-exact, −25 MiB).

### Task 4: decoder (LSTM 21 / 116 ms, 0.6 ms per token, memory-bound)

- [x] Try in order, keep what wins and stays bit-exact: layer 0's `Wx·embed[token]` cached per
      token, the LSTM step split over threads spawned per step (no busy waiting), Q8_0 weights
      dequantized in registers so a step's weights fit in L2.

### Task 5: rest of the encoder

- [x] Simplify core + `src/parakeet/` back to ~3.1k lines (near-duplicate kernels and helpers
      merged), speed and fixtures unchanged.
- [x] Re-profile. conv_glu's depthwise weights as [tap][ch] (shared with GigaAM). GEMM tile
      shape at M = 138 (1.5 vs 1.8 TFLOP/s at M = 442) if the encoder is still behind the
      reference's.
- [x] Check: `make check`; `make test`; the bench tables in `docs/perf.md` and `readme.md`
      updated; roadmap phase 5 ticked if the goal is met.

Decision point: if warm, first call or peak footprint is still behind the reference on any clip
after these, stop and report to the user with the profile.

## Result

M2, end of phase 5: warm 149 / 503 / 83 ms (jfk / dots / jobs-silence), peak 765 / 790 / 757 MiB,
from 181 / 756–763 / 94 ms and 810 / 877 / 800 MiB at the end of phase 4. The same session's
GigaAM rows are in the GigaAM plans; the end-of-phase-6 table is in the Parakeet V3 plan.

The same binaries on an Apple M1 (8 GPU cores, 16 GiB, macOS 15.8.1; its fan is broken, so it
throttles under load: a sanity check, not a speed reference), two rounds, wavo / reference, ms.
All 15 texts equal the fixtures.

- Warm median: GigaAM e2e-rnnt 75.2–76.5 / 168.5–169.7 / 452.3–452.8 vs 74.9–75.4 / 175.3–176.1 /
  461.8–463.7; Parakeet V2 254.5 / 719.5 / 129.7–130.6 vs 269.6–273.9 / 882.9–883.0 /
  158.9–160.1. The CTC models are 6–21% faster, Parakeet 6–19%, the two RNN-T GigaAM models on
  par (−1% to +4%).
- Warm calls get slower than the first call for both engines, from the CPU side: on e2e-rnnt ru
  the reference's `decode_ms` grows from 4.9 to ~16 over 20 calls and wavo's mel + decoder from
  ~7 to ~24 ms while both GPU times stay flat. So the RNN-T warm medians mostly measure each CPU
  decoder throttled.
- First call below the reference except rnnt ru-short and ru-long (within 0.5%); load 35–45% and
  peak footprint 14–40 MiB below.
