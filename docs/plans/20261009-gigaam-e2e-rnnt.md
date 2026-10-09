# GigaAM v3 e2e-rnnt

Roadmap phase 1: the first working model, matched exactly against the reference, plus a baseline
benchmark. Follow `agents.md`. All three tasks land as one commit.

## Goal

`wavo <gigaam-v3-e2e-rnnt-Q8_0.gguf> ru.wav` prints `Важно различать глаголы и дополнения.`
`make test` reproduces the reference output exactly (text, token pieces, start frames) on `ru`,
`ru-short` and `ru-long`. Weights stay Q8_0/F16 on the GPU. `docs/perf.md` gets the first wavo vs
transcribe.cpp numbers.

## Model facts

Sources: `3rd/transcribe.cpp/scripts/convert-gigaam.py` (tensor names and layouts),
`src/arch/gigaam/`, `docs/porting/families/gigaam.md`.

- Frontend: periodic Hann 320, hop 160, FFT 320, full windows only (`center=false`). The mel
  filterbank `[64, 161]` and the window come from GGUF tensors `frontend.mel_filterbank` /
  `frontend.window`. Scaling is natural `log(clamp(x, 1e-9, 1e9))`.
- Subsampling: `enc.pre_encode.conv.{0,2}`, conv1d kernel 5, stride 2, each followed by ReLU
  (×4, 40 ms frames).
- 16 Conformer blocks `enc.blocks.{i}`, width 768, 16 heads of 48, LayerNorm eps 1e-5:
  - `x += 0.5 · ff1(norm_ff1(x))`, where the FFN is `linear1` (768→3072) → SiLU → `linear2`.
  - Attention: `y = norm_attn(x)`. Apply RoPE (theta 5000, split-half rotation inside each head)
    to `y` before the Q and K projections; V uses the unrotated `y`. Then `x += attn.linear_out(…)`.
  - Conv: `norm_conv` → `conv.pointwise1` (768→1536) → GLU → `conv.depthwise` (kernel 5) →
    `conv.ln` (a LayerNorm) → SiLU → `conv.pointwise2`, added to `x`.
  - `x += 0.5 · ff2(norm_ff2(x))`, then `norm_out`.
- RNN-T decoder (CPU): embedding `pred.embed`, LSTM 320 (`pred.lstm.{i}`, gates i, f, g, o, one
  summed bias). The initial predictor input is zero, not the blank embedding. The LSTM state
  commits only after a non-blank token and stays cached across blank frames. Joint:
  `joint.enc` + `joint.pred` → ReLU → `joint.out`. Vocab 1025, blank 1024; `tokenizer.ggml.tokens`
  holds 1024 pieces followed by `<blank>`. At most 10 symbols per frame; argmax ties go to the
  lowest id; `▁` becomes a space.
- The joint's encoder projection `joint.enc` runs on the GPU as the last GEMM (Q8_0). The CPU
  decoder's matrices (`pred.embed`, `pred.lstm.0.{Wx,Wh}`, `joint.pred`, `joint.out`, ~6 MB)
  expand to F32 at load, like biases, norms and frontend tensors.

## Tasks

### Task 1: first transcript

- [x] Dependencies via `cargo add`: `thiserror`, `anyhow`, `bytemuck`, `half`, `pollster`,
      `rustfft`, `hound`,
      `wgpu@=30.0.1 --no-default-features --features std,wgsl,metal`.
- [x] `src/gguf.rs`: GGUF v3 reader (metadata, tensor views for F32/F16/Q8_0).
- [x] `src/gpu/`: device, buffers, weight upload (F16 as is; Q8_0 repacked into aligned int8 data
      + f32 block scales), readback, one submit per encoder block. Simple portable kernels:
      linear with an F32/F16/Q8_0 weight, layer norm, softmax, SiLU/ReLU/GLU/add, conv1d via
      im2col, depthwise conv, RoPE, attention (scores → softmax → values). Each kernel has its
      own entry point.
- [x] `src/gigaam/`: frontend, encoder, RNN-T decoder, detokenizer, per the facts above.
- [x] `Model::load` / `transcribe` in `src/lib.rs`; CLI `wavo MODEL.gguf AUDIO.wav [--tokens]`
      reads mono 16 kHz PCM16/F32 WAV.
- [x] `make fixtures MODEL=gigaam-v3-e2e-rnnt SAMPLES="ru ru-short ru-long"`, plus `tests/gigaam.rs`
      matching all three exactly.
- [x] Check: the CLI prints the `ru` transcript; `make check`; `make test`.

### Task 2: fast paths

- [x] Cooperative-matrix GEMM and attention, subgroup layer norm and softmax. Each turns on only
      after a probe dispatch on the real pipeline; otherwise the portable kernel runs. See
      `docs/perf.md` for kernel layouts and Naga quirks.
- [x] Inline tests: each kernel against a CPU loop on small and odd shapes, on both paths,
      including the Q8_0/F16 GEMM.
- [x] Check: `make check`; `make test` with fixtures still exact.

### Task 3: bench and baseline

- [x] `wavo bench MODEL.gguf AUDIO.wav [-n N]`: load, first call, warm median and min.
- [x] Measure `ru`, `ru-short`, `ru-long` against `transcribe-bench --warmup 1 --iters 10`
      (warm = median of `per_iter[].wall_ms`). For the first call, run a fresh process with
      `--warmup 0 --iters 1`. Peak memory comes from `/usr/bin/time -l`.
- [x] Log the numbers in `docs/perf.md`, add a speed table to `readme.md`, tick roadmap phase 1.
- [x] Check: `make check`; `make test`.

After this, write the phase-2 speed plan from the measurements.
