# Parakeet TDT V2

Roadmap phase 4: `parakeet-tdt-0.6b-v2` as a second family on the shared core, matched exactly
against the reference, plus a first benchmark. Follow `agents.md` → "Adding a model".

## Goal

`make test` reproduces the reference fixtures `jfk`, `dots` and `jobs-silence` exactly (text,
token pieces, start frames; `jobs-silence` is empty). Weights stay Q8_0/F16 on the GPU.
`docs/perf.md` gets the first wavo vs `transcribe-bench` numbers (warm, first call, load, peak).
Core + `src/parakeet/` stays within ~3k lines of Rust and WGSL, tests excluded (core + GigaAM is
~2.5k).

## Model facts

Sources in `3rd/transcribe.cpp`: `scripts/convert-parakeet.py`,
`src/arch/parakeet/{model,encoder,decoder,weights}.cpp`, `src/conformer/conformer.cpp`,
`src/transcribe-mel.cpp`, `src/transcribe-tokenizer.cpp`, `docs/models/parakeet*.md`,
`docs/porting/families/parakeet.md`, `catalog/parakeet-tdt-0.6b-v2.json`
(`handy-computer/parakeet-tdt-0.6b-v2-gguf`, `parakeet-tdt-0.6b-v2-Q8_0.gguf`, 697 tensors).

- Metadata to check at load: `general.architecture` `parakeet`, `stt.parakeet.head_kind` `tdt`;
  `encoder.{n_layers 24, d_model 1024, n_heads 8, d_ff 4096, conv_kernel 9,
  subsampling_factor 8, subsampling_channels 256}`, `use_bias` and `xscaling` false (bools),
  `att_context_left/right` −1 (INT32), `att_context_style` `regular`, `conv_context_left/right`
  4, `conv_norm_type` `batch_norm`; `predictor.{hidden 640, n_layers 2, vocab 1025}`;
  `joint.{hidden 640, num_extra_outputs 5, activation relu}`; `tdt.durations` INT32 array
  [0, 1, 2, 3, 4], `tdt.max_symbols` 10; `stt.frontend.{n_fft 512, win_length 400,
  hop_length 160, num_mels 128, window hann, normalize per_feature}`.
- Tensors (ggml dims): `enc.pre_encode.conv.{0,2,5}.weight` [3,3,1,256] F32 (conv0 dense
  1→256, conv2/5 depthwise; freq tap fastest, then time), `conv.{3,6}.weight` [1,1,256,256] F32,
  all with F32 biases; `enc.pre_encode.out` [4096,1024] Q8_0 + bias. Per block
  `enc.blocks.{i}.`: `norm_{ff1,attn,conv,ff2,out}`, `ff{1,2}.linear{1,2}` Q8_0,
  `attn.linear_{q,k,v,out,pos}` [1024,1024] Q8_0, `attn.pos_bias_{u,v}` [128,8] F32,
  `conv.pointwise1` [1,1024,2048] F16, `conv.depthwise` [9,1,1024] F32,
  `conv.bn.{weight,bias,running_mean,running_var}`, `conv.pointwise2` [1,1024,1024] F16; no
  linear or conv biases. Decoder: `pred.embed` [640,1025], `pred.lstm.{0,1}.{Wx,Wh}` [640,2560]
  + `bias` (= bias_ih + bias_hh), `joint.enc` [1024,640], `joint.pred` [640,640], `joint.out`
  [640,1030], all Q8_0 with F32 biases. No frontend tensors.
- Frontend (`transcribe-mel.cpp` with the loader's `MelConfig`, `model.cpp:482`): frames =
  n / 160 + 1. Pre-emphasis in f64 before padding (`y[0] = x[0]`, `y[i] = x[i] − 0.97f·x[i−1]`),
  256 zeros on each side (`pad_mode` `constant`; the file header says reflect), symmetric
  Hann(400) in f64 centred in 512, FFT 512 in f64, power `re² + im²` cast to f32. Slaney
  filterbank [128][257] built in f64 (fmin 0, fmax 8000, librosa `norm='slaney'`), cast to f32;
  mel is an f32 dot (Accelerate sgemm), then `ln(x + 2⁻²⁴)` in f32. Per mel bin over the first
  frames − 1 frames: f64 mean and unbiased variance, `(x − mean) / (std + 1e-5)` to f32; the
  last frame is 0. The GGUF's dither (1e-5) is never applied.
- Subsampling (`build_pre_encode`, `conformer.cpp:992`): image [T_mel][128], one channel. conv0
  (3×3, stride 2, pad 1 on both axes) + bias + ReLU; conv2 depthwise 3×3 s2 p1 + bias, no ReLU;
  conv3 pointwise + bias + ReLU; conv5 depthwise + bias; conv6 pointwise + bias + ReLU. Each axis
  L → (L − 1) / 2 + 1: freq 128 → 64 → 32 → 16, time T_mel → T (dots 3534 → 442; 80 ms frames).
  Flatten to [T][c·16 + f], then `out` + bias. No √d scaling, no length masks (those are
  `parakeet-ultra` only). conv0's output is the largest tensor of the whole encoder
  ([T/2][64][256], 115 MB on `dots`); the next two ([2T][32][256], 29 MB each) still outsize
  every block buffer, so they set the arena size.
- Block (`build_conformer_block`, `conformer.cpp:819`): GigaAM's order, `x += ½·ff1(LN x)`,
  `x += attn(LN x)`, `x += conv(LN x)`, `x += ½·ff2(LN x)`, `x = norm_out(x)`; LN eps 1e-5.
  Conv module: pointwise1 → GLU (first half · σ(second)) → depthwise k9, pad 4/4, no bias →
  BatchNorm folded at load like `fuse_batch_norm` (`model.cpp:208`: `s = w / sqrt(var + 1e-5)`,
  `b = bias − mean·s` in f32, applied as `x·s + b`) → SiLU → pointwise2.
- Relative attention (`rel_pos_mhsa`, `conformer.cpp:561`): 8 heads × 128. `q_u = q + pos_bias_u`,
  `q_v = q + pos_bias_v`. `pos_emb` [2T−1][1024], row r is position T−1−r: `[2k] = sin(pos·d_k)`,
  `[2k+1] = cos(pos·d_k)`, `d_k = exp(2k·(−ln 10000 / 1024))`, all f32 on the CPU
  (`model.cpp:1263`; Rust's f32 `sin/cos/exp/ln` call the same libm). Rows depend only on the
  position, so a table for the longest T serves every shorter call as a contiguous slice.
  `P = linear_pos(pos_emb)` per block. Score(i, j) = (q_u,i·k_j + q_v,i·P[j − i + T − 1]) / √128,
  softmax over j, then `linear_out`. Materialized position scores are [heads][T][2T−1]
  (12.5 MB on `dots`, ~0.9 GB at 5 min).
- TDT decoder (`decode_tdt_greedy`, `decoder.cpp:1018`; CPU, F32, Q8_0 dequantized):
  predictor = 2 LSTM layers of 640, gates i, f, g, o, `(Wx·x + Wh·h) + b`, output the top h.
  The predictor output for a decision is the LSTM of (embedding of the last emitted token, or
  zeros at the start; committed states, zero at the start). Joint `out(relu(enc_proj[t] +
  pred(h)))` → 1030 logits: tokens [0, 1025) with blank 1024, then 5 durations; both argmaxes
  take the first max; duration = `durations[argmax]`. From t = 0 while t < T: a non-blank emits
  (token, t), commits that LSTM state and becomes the next input; a blank keeps both. Then
  `t += duration`, `symbols += 1`; duration ≠ 0 → symbols = 0; else if symbols ≥ 10 or blank →
  `t += 1`, symbols = 0. Frames after a blank are skipped by its duration, so GigaAM's
  two-frame joint span does not carry over.
- Text and fixtures (`build_result_from_raw_tokens`, `model.cpp:868`): after decoding, drop
  UNKNOWN/CONTROL tokens and `<unk>` (v2: ids 0 and 1024); text = pieces with `▁` → space, runs
  of spaces collapsed, trimmed (`model.cpp:702`). No byte-fallback pieces in v2; id 819 is a
  lone `▁`. The CLI prints t0 = round(80·frame) ms and each token decoded (` And`, where the
  GigaAM fixtures show `▁В`); a silent clip prints `text: (empty)` and no `tokens:` line.
- Reference roundings that NeMo/F32 lack (only matter if a fixture mismatches): pre-encode
  pointwise convs go through `ggml_conv_2d` with an F16 im2col (`ggml.c:4763`), so both operands
  are halves; flash attention casts K, V and the scaled position bias to F16
  (`conformer.cpp:696, 766`); the Q8_0 Metal matmul halves both operands; `joint.enc` and the
  decoder run on the CPU in F32 (wavo's GEMM rounds W to f16). Research's all-F32 encoder still
  matched the Metal CLI's tokens, frames and durations on `jfk` and `dots`.

## Port from `research` (read only: `git show research:<path>`)

- `src/parakeet/frontend.rs`: window, filterbank, f64 pre-emphasis and padding, rustfft f64,
  normalize. Its per-filter f32 `mul_add` in bin order was bitwise equal to the reference mel on
  the M2; keep the order (zero weights add exact zeros, so loop over nonzero bins only). Drop
  rayon (not an allowed dependency).
- `src/parakeet/encoder.rs`: `batch_norm`, `relative_positions`, subsampling order (no ReLU
  between depthwise and pointwise, flatten `channel·16 + freq`).
- `src/shaders/spatial.wgsl`: `pixel()` addressing for 3×3 stride-2 pad-1 convs on [t][f][c],
  `depthwise2d`, flatten and shift indexing (`k + T − 1 − q`).
- `src/parakeet/decoder.rs`: `greedy()` and its scripted-logits tests; `special_tokens` and
  `join_tokens` rules.
- Pitfall it hit: an elementwise pass over conv0's output on `dots` ([1767][64][256]) needed
  113088 workgroups in x; keep every grid dimension ≤ 65535.
- Avoid: weights expanded to F32 (1.8 s load, 7.8 GB peak), a buffer and submit per op, the 60 s
  cap, `Variant` tables, geometry and finiteness checks, trace hooks, tensor-level oracle tests.

## Tasks

### Task 1: model and fixtures

- [x] `parakeet-tdt-0.6b-v2` is already in `MODELS`; `make models`.
- [x] `make fixtures MODEL=parakeet-tdt-0.6b-v2 SAMPLES="jfk dots jobs-silence"`; keep trailing
      spaces in token lines (a lone `▁` prints as a space).
- [x] One fixture test for all models (`tests/gigaam.rs` → `tests/models.rs`): samples per
      model, `text: (empty)` = empty text and no tokens, pieces compared with `▁` as a space;
      fails on a missing model or sample. Lands with Task 3.

### Task 2: shared core (GigaAM output and speed unchanged)

- [x] Own commit first: `Token::start_ms` replaces `frame` (frame × 40 / 80 ms); `--tokens`
      prints ms; the test compares ms with the fixture t0; `agents.md` and `readme.md` updated.
- [x] `gguf.rs`: bools, negative INT32, INT32 arrays; unit test.
- [x] `Gpu::linear`: layers without bias (zeros) and an F32 bias vector from the caller.
- [x] Move the Conformer block (weights + `record`) from `src/gigaam/encoder.rs` to
      `src/conformer.rs`, switched only by attention (rotary / relative) and conv norm
      (LayerNorm / per-channel affine); move `dot`, `matmul`, `tile` out of `src/gigaam/mod.rs`
      for both decoders. Frontend, subsampling, arena and head stay per family.
- [x] `conv_glu`: kernel size as a param (5, 9) and the affine mode, portable and fast; test.
- [x] Check: GigaAM fixtures exact, warm `ru` A/B/A unchanged, `make check`, `make test`.

### Task 3: vertical slice `src/parakeet/`

- [x] `mod.rs`: metadata and shapes checked once at load, `transcribe`, token filter and text as
      above; `Model` picks the family from `general.architecture`.
- [x] `frontend.rs` per the facts; input under 3 mel frames gives an empty transcript, not NaN.
- [x] Kernels, each tested against a CPU loop on small odd shapes: 3×3 stride-2 conv on
      [t][f][c] with conv0 + ReLU fused into the first depthwise conv (conv0's output never
      exists) and plain depthwise for conv5; flatten to [t][c·16 + f]; pos scores
      `PS[h][i][r] = q_v,i·P_r`; portable `scores` adds `PS[h][i][j − i + T − 1]` before the
      scale and takes a row stride. The flash kernel stays head_dim ≤ 64, so Parakeet runs the
      portable attention in this phase.
- [x] `encoder.rs`: subsampling (pointwise convs and `out` via `Gpu::linear`); one stacked GEMM
      `[q + u | k | v | q + v]` (weights q, k, v, q; biases `pos_bias_u`, 0, 0, `pos_bias_v`;
      exact, the reference also adds after the matmul); `pos_emb` table cached on the CPU for
      the longest T so far, its 2T−1 rows written per call; `P` per block; the shared block;
      `joint.enc` as the last GEMM; one compute pass over an arena whose block buffers also
      hold the subsampling intermediates; warm-up at load.
- [x] `decoder.rs`: TDT loop and 2-layer LSTM on the shared NEON `matmul`, decoder weights
      expanded to F32 at load (~33 MB, as the reference does). Scripted-logits test: blank + 0
      advances one frame, non-blank + 0 stays, the cap of 10 counts blanks, duration is the
      metadata value.
- [x] Code adapted from `3rd/` (mel filterbank, TDT loop) names its source file; add the
      transcribe.cpp copyright line to `LICENSE` once.
- [x] Fixtures exact; `make check`; `make test`. On a mismatch compare with the reference dumps
      (`TRANSCRIBE_DUMP_DIR`: `enc.mel.in`, `enc.pre_encode.out`, `enc.block.0.*`,
      `enc.final`, `dec.*`), then try its roundings in this order: scaled position bias to f16,
      K/V to f16, pre-encode pointwise operands to f16.

### Task 4: bench and baseline

- [x] Measure `jfk`, `dots`, `jobs-silence` with `wavo bench -n 20` against
      `transcribe-bench --warmup 1 --iters 10` (warm = median of `per_iter[].wall_ms`), first
      call in a fresh process with `--warmup 0 --iters 1`, peak from `/usr/bin/time -l`.
- [x] Log the numbers in `docs/perf.md` (replace the research-era Parakeet row in "Targets",
      add "Current state: Parakeet TDT V2"); add the model to `readme.md`; tick roadmap phase 4.
- [x] Check: `make check`; `make test`.

After this, write the phase-5 speed plan from a profile of the working V2.
