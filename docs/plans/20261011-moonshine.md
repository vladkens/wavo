# Moonshine

A fourth family, on the person's request, after the Whisper variants and model downloads (roadmap
13–14): `moonshine-tiny` and `moonshine-base` first, then the twelve single-language fine-tunes as
config. `moonshine-streaming-*` is a separate architecture and an optional later phase. Follow
`agents.md` → "Adding a model"; read `docs/perf.md` before speed work. Benchmark numbers go in
this plan's final `## Result` and the PRs; `docs/perf.md` gets only log entries of attempts.

## Goal

- `make test` reproduces the reference fixtures `jfk`, `dots` and `jobs-silence` of
  `moonshine-tiny` and `moonshine-base` exactly (text only, see Fixtures). All other fixtures stay
  exact.
- Against `transcribe-bench` on the same Q8_0 GGUF and WAV on the M2: warm, first call, load and
  peak footprint at least on par. The reference publishes M4 Max numbers (catalog): Metal `jfk`
  56 ms (encode 16, decode 40), `dots` 422 ms (encode 90, decode 331). Its CPU backend is as fast
  for a model this small (54 / 373 ms), so we measure against both. Task 5 records the baseline in
  `## Result`.
- Weights stay Q8_0/F32 on the GPU as stored. Core plus `src/moonshine/` stays within ~3k lines
  (see Line budget).

## Model facts

Sources in `3rd/transcribe.cpp`: `catalog/moonshine-*.json`, `scripts/convert-moonshine.py`,
`src/arch/moonshine/{model,encoder,decoder,weights}.cpp`, `docs/models/moonshine*.md`,
`docs/porting/families/moonshine.md`, `docs/input-limits.md`. No Moonshine file is downloaded
yet: the metadata below comes from the converter, and Task 2 checks it against the published
header.

- Files (`handy-computer/<variant>-gguf`, `<variant>-Q8_0.gguf`): `moonshine-tiny` 35 466 912 B
  (27.1 M params), `moonshine-base` 77 476 480 B (61.5 M), `moonshine-tiny-{ar,ja,ko,uk,vi,zh}`
  35 466 944 B, `moonshine-base-{ar,ja,ko,zh}` 77 476 480 B, `moonshine-base-{uk,vi}`
  77 476 512 B. Streaming: `moonshine-streaming-tiny` 50 462 816 B, `-small` 198 506 848 B,
  `-medium` 295 793 568 B. All MIT.
- Metadata: `general.architecture` `moonshine`; `general.languages` one code (`en` for the
  English two); `stt.moonshine.encoder.{n_layers, d_model, n_heads, n_kv_heads, ffn_dim,
  activation gelu}`; `stt.moonshine.decoder.{n_layers, d_model, n_heads, n_kv_heads, ffn_dim,
  activation silu, vocab_size 32768, max_position_embeddings 194, tie_word_embeddings true}`;
  `stt.moonshine.{partial_rotary_factor, rope_theta 10000, attention_bias false,
  pad_head_dim_to_multiple_of 8, decoder_start_token_id 1, bos_token_id 1, eos_token_id 2,
  pad_token_id 2}`; `stt.moonshine.conv_stem.{channels [d, 2d, d], kernel_sizes [127, 7, 3],
  strides [64, 3, 2], groupnorm_num_groups 1, groupnorm_eps 1e-5}` (u32 arrays);
  `stt.frontend.{type raw, sample_rate 16000, num_mels 1}`; `stt.capability.*` all false;
  `tokenizer.ggml.model` `bpe` with `byte_fallback`.
- Shapes. Tiny: d 288, 6 + 6 layers, 8 heads × 36, ffn 1152, rotary factor 0.9. Base: d 416,
  8 + 8 layers, 8 heads × 52, ffn 1664, factor 0.62. Rotated dims per head `int(hd · factor) & ~1`
  = 32 in both (`weights.h:151-161`); heads padded to 40 and 56 inside attention.
- Tensors (ggml dims, fastest first): `enc.conv.0.weight` [127, 1, d] (no bias),
  `enc.conv.1.weight` [7, d, 2d] + bias, `enc.conv.2.weight` [3, 2d, d] + bias,
  `enc.conv.norm.{weight,bias}` [d]; per block `enc.blocks.{i}.` `norm_attn`,
  `attn.{q,k,v,out}` [d, d] (no biases), `norm_ffn`, `ffn.fc1` [d, ffn] + bias, `ffn.fc2`
  [ffn, d] + bias; `enc.final_norm`; `dec.token_embd.weight` [d, 32768] (embedding and tied
  head); per block `dec.blocks.{i}.` `norm_self`, `self_attn.{q,k,v,out}`, `norm_cross`,
  `cross_attn.{q,k,v,out}`, `norm_ffn`, `ffn.fc1` [d, 2·ffn] + bias, `ffn.fc2` [ffn, d] + bias;
  `dec.final_norm`. Every LayerNorm has a weight and no bias. In the Q8_0 file the linears and
  the embedding are Q8_0 (every inner dim divides 32); the conv weights are F32 (the Q8_0
  preset's conv column, `tools/transcribe-quantize/policy.cpp:271`), norms and biases F32.
- Frontend: none. Raw PCM goes into the conv stem as is, any length, no normalization
  (`model.cpp:398`). Frames: T1 = (n − 127)/64 + 1, T2 = (T1 − 7)/3 + 1, T = (T2 − 3)/2 + 1,
  no padding (`encoder.cpp:191-216`): ~41.7 frames/s (`jfk` 457, 48.5 s 2019). Under 895
  samples there is no frame and the reference fails the run.
- Encoder (`encoder.cpp:218-356`): conv0 (k127 s64) → tanh → GroupNorm with one group over all
  [T1][d] values (`ggml_group_norm`, then `· w`, then `+ b`; `:277-293`) → conv1 (k7 s3) + bias →
  GELU → conv2 (k3 s2) + bias → GELU → [T][d] → pre-LN blocks `x += out(attn(LN x))`,
  `x += fc2(GELU(fc1(LN x)))` → final LN. LayerNorm eps 1e-5, no bias (`conformer.cpp:34-41`).
  GELU is exact-erf (`ggml_gelu_erf`), on Metal ggml's `erf_approx`, as in Whisper. Attention
  (`:67-148`): bidirectional, no mask, partial RoPE on q and k at positions 0..T−1, scale
  1/√hd (unpadded), the pad dims zero and sliced off before `out`.
- RoPE: GPT-J interleaved pairs (x[2i], x[2i+1]) = `GGML_ROPE_TYPE_NORMAL`, not the NEOX
  split-half that our `layer_norm` rotary does; only the first 32 dims of a head rotate;
  θ = pos · 10000^(−2i/32) (`encoder.cpp:43-60`, `ggml-metal/kernels/rope.metal:66-81`).
- Decoder (`decoder.cpp:454-614`, step graph `:617-699`): x = `token_embd[id]` (no positions,
  no scale); pre-LN layers: causal self-attention with partial RoPE on q and k at the token's
  position (the cache holds rotated K, `:241-337`), cross-attention without RoPE over K/V
  computed once per call (`Wk·enc`, `Wv·enc`, no biases; `:402-451`), SwiGLU FFN: fc1 → [x | gate],
  `silu(gate) · x`, fc2 (`:82-109`); final LN; logits = `token_embd · x`, no bias.
- Reference precision (only if a fixture mismatches): K/V caches are F32 by default
  (`model.cpp:444-450`; the CLI's `--kv-type` defaults to auto). On Metal the decoder runs
  without flash attention (`model.cpp:294`): explicit K·Q, softmax, V in F32. The encoder runs
  flash attention, but Metal's flash kernel has head size 40 and not 56
  (`ggml-metal-device.m:1720-1736`), so base's encoder attention falls back to the CPU backend.
  GEMMs as for Whisper: M > 8 `kernel_mul_mm` (f16 operands), M = 1 `mul_mv` (F32).
- Decoding (`model.cpp:485-765`): greedy only, no suppression, no language or task tokens. Prompt
  [1] at position 0; the prompt's argmax runs on the CPU over log-softmax with a strict `>`
  (`:580-598`); later steps run `ggml_argmax` over raw logits (Metal picks the highest index
  among equal maxima, `misc.metal:25-27`; such a tie has not come up). Stop at EOS 2 or at 194
  positions, so at most 194 tokens. A repetition guard runs after every token
  (`transcribe-repetition-guard.h`: `k_stop_bar` {128, 8, 64, 192, 4}, a 1-token block needs
  64 copies, an 8-token block 8). On a stop at the cap, the guard trims a repeating tail
  (`k_budget_trim_bar`, `:114-128`). `TRANSCRIBE_NO_REPETITION_GUARD=1` turns it off.
- Text (`model.cpp:767-790`): `decode_sentencepiece` (`transcribe-tokenizer.cpp:174-227`): `▁` →
  space, `<0xHH>` → that raw byte, out-of-range ids skipped, control tokens (`<s>`, `</s>`,
  `<<ST_N>>`) printed literally. Then one leading space is removed. Vocab: 0 `<unk>`, 1 `<s>`,
  2 `</s>`, 3–258 the 256 byte tokens, 259–31999 pieces, 32000–32767 `<<ST_N>>`.
- Timestamps: none (`capabilities.cpp`): one untimed segment. `--timestamps token|segment` fails
  with `UNSUPPORTED_TIMESTAMPS` (`transcribe-asr.cpp:121-126`), so `make fixtures` as is writes
  empty files. `--timestamps none` prints `run: <status>` and `text: …` only
  (`examples/cli/asr.cpp:865-886`). A truncated or looping run still prints its text, followed by
  a `note:` line that the fixture awk drops.
- `max_audio_ms`: advisory 194 tokens / 4 tokens/s = 48 500 ms (`model.cpp:72-84`). This is a
  soft window: any length runs, and a clip that reaches the cap is flagged as truncated
  (`docs/input-limits.md:104-121`).
- Variants: base is config. The language fine-tunes have the same shapes as their size and a
  different vocabulary content; their text goes through byte fallback (Cyrillic, Vietnamese
  diacritics, CJK, Arabic). On other languages and silence they tend to loop until the cap
  (`samples/README.md`). The FLEURS `<lang>-short` / `-long` clips exist so they can be measured
  without looping.

## Design

### Shared and new

- Core, extended:
  - GEMM and gemv: output widths that are multiples of 32 but not of 64 (d 288 and 416,
    `[q | k | v]` 864 and 1248, conv 576 and 832). Stores past n are masked, and c's row stride is
    n. The weight rows stay padded to 64 for the tile reads. Today c takes the padded width, which
    breaks the `[t][d]` layout every other kernel reads.
  - `Epilogue::Tanh` for conv0.
  - `im2col`: a padded column count. conv0's K = 127 becomes 128 with a zero column, and its F32
    weight gets a zero column at load, which is exact.
  - `attend` over more than 2048 keys, in 2048-key blocks with an online softmax: T passes 2048
    at ~49 s, and the library takes any length.
  - The pre-LN transformer pieces move from `src/whisper/` into a shared module now that a second
    family needs them (the Whisper plan's rule). That covers the encoder block and the decoder
    step, with options for biases, RoPE or learned positions, and GELU or SwiGLU. Whisper output
    and speed must stay unchanged.
- Core, new (portable kernels only unless a fast one measures faster):
  - `rope`: in place over rows `[t][q | k | …]`. In each head, the first `rot` dims rotate as
    adjacent pairs, using a table of `rot`/2 cosines and sines per position computed on the CPU.
    The first position is an immediate, for the decoder step.
  - `group_norm`: mean and variance (of x − mean) over all [T1][c] values, then
    `(x − mean)/√(var + ε) · w[c] + b[c]`. One workgroup.
  - `swiglu`: `h[..f] · silu(h[f..2f])` for the decoder FFN.
  - `argmax` over the logits into one u32, lowest id on a tie, so a step reads back 4 bytes
    instead of 128 KB.
- `src/moonshine/`:
  - `mod.rs`: metadata and shapes checked once at load, special ids, `transcribe` (encoder,
    decode loop, repetition guard, text and tokens), `max_audio_ms`.
  - `encoder.rs`: conv stem, then the shared blocks and final LN.
  - The decoder step and caches are the shared ones (`src/transformer.rs`), with options for
    RoPE, no biases and SwiGLU.
  - `repeat.rs`: the repetition guard.
  - `tokenizer.rs`: SentencePiece decode with byte fallback and pieces. It shares Whisper's
    "complete UTF-8 goes to the token with the last byte" logic where that fits.
- Not used by Moonshine: `conformer.rs` (except `linear`), `spatial.wgsl`, `attention_fast.wgsl`,
  `cpu.rs`, `fft.rs`.

### Encoder: one compute pass per call

pcm [n] → `im2col` (k127, s64, 128 columns) → conv0 GEMM + tanh → `group_norm` → `im2col` (k7 s3)
→ conv1 GEMM + GELU [T2][2d] → `im2col` (k3 s2) → conv2 GEMM + GELU [T][d] → per block through
`Pass::layers`: LN, `[q | k | v]` GEMM (no bias), `rope` on q and k, attention, out GEMM +
residual, LN, fc1 GEMM + GELU, fc2 GEMM + residual → final LN → per decoder layer a `[K | V]`
GEMM and `pack_f16` into the cross cache. The BOS step is recorded into the same pass, as in
Whisper. Buffers are sized by T, grown and kept. The heads of 36 and 52 dims are no multiple of
8, so attention runs the portable kernels in the correct slice. Their scores take 8·T² floats:
7 MB at 11 s, 130 MB at 48 s. The fast path comes in Task 6.

### Decoder: one GPU pass per token

The step reuses Whisper's kernels: `embed` (a zero buffer as positions); per layer LN, gemv
`[q | k | v]`, `rope` at `pos`, `pack` of k and v into self-cache row `pos`, `attend` over
pos + 1 keys, gemv out + residual, LN, gemv cross q, `attend` over T keys, gemv out + residual,
LN, gemv fc1 + bias, `swiglu`, gemv fc2 + residual; final LN, gemv head (32 768 Q8_0 rows),
`argmax`. That is ~80 dispatches for tiny. A token reads ~20 MB of weights for tiny and ~41 MB
for base: 0.2–0.4 ms at the M2's ~100 GB/s. So submit and readback latency per token dominate,
as the reference's decode share shows (`dots`: 331 of 422 ms). `attend` handles head_dim 36 and
52 as they are, and the caches are F16 like Whisper's. If a fixture needs it, an F32 cache
variant comes next (see Risks).

Decision point after the correct slice, with per-token GPU time and submit overhead measured:
record several steps per submit, where `argmax` writes the id that the next step's `embed` reads.
The CPU reads the ids back per batch, applies EOS, the 194 cap and the repetition guard token by
token, and drops the steps past the stop. The output is identical; at most a batch of steps is
wasted.

### Decoding, long audio and `max_audio_ms`

- Greedy as the reference: prompt [1] at position 0, stop at EOS or 194 positions. Repetition
  guard per decision 2, ported with comments that name the header.
- `transcribe` runs one pass over any length, as the reference does; `max_audio_ms()` is
  `Some(30_000)` (decision 3). Input under 895 samples gives an empty transcript (the reference fails it). No
  new public API.

### Tokens, text and fixtures (decision 1)

- `Transcript.text` is the reference's text.
- `tokens` are the generated ids before EOS. A piece is the vocabulary piece with `▁`. A
  byte-fallback token gets the text it completes: a character split over byte tokens goes to the
  token with its last byte, so pieces are valid UTF-8, some empty. A control token gets its
  literal piece, as the reference prints it. `start_ms` is 0, the start of the one untimed
  segment, as Whisper does for its segments. `Token`'s doc and `readme.md` say so.
- `make fixtures MODEL=moonshine-tiny TIMESTAMPS=none SAMPLES="jfk dots jobs-silence"`: fixtures
  hold the `text:` line only. The recipe fails when the reference's `run:` line is not `OK`, so a
  truncated or looping clip never becomes a fixture silently.
- `tests/models.rs`: for a text-only fixture, the text must match, and the pieces joined (`▁` as a
  space, one leading space trimmed) must equal it.
- CLI: `--srt` for Moonshine uses `srt_segments`, as for Whisper: one cue per split piece of the
  audio.

### First slice and variant order

1. `moonshine-tiny` (English): `jfk` (11 s), `dots` (35.3 s, the reference's bench clip; it must
   end at EOS under the cap), `jobs-silence` (5.3 s, empty for Parakeet: EOS first, or a loop
   that the guard must cut exactly).
2. `moonshine-base`, in the same PR: the same clips, config only (32 of 52 dims rotate).
3. Language fine-tunes, tiny then base: `uk` and `vi` first (byte fallback on diacritics), then
   `zh`, `ja` and `ko` (mostly byte tokens), then `ar`. Clips `<lang>-short` and `<lang>-long`.
4. Streaming (optional, below).

### Streaming variants

`moonshine-streaming-{tiny,small,medium}` is a separate architecture, `moonshine_streaming`
(`docs/porting/families/moonshine_streaming.md`, `scripts/convert-moonshine_streaming.py`,
`src/arch/moonshine_streaming/`):

- Frontend: 5 ms frames of 80 samples, normalized per frame (`ggml_norm`, eps 1e-6), then
  `asinh(exp(log_k) · x)`, a Linear 80 → d (F16, K = 80) + SiLU, and two causal conv1d k5 s2
  (+ SiLU after the first) → 50 Hz.
- Encoder: no positional encoding, and sliding-window attention per layer ([16, 4], [16, 4],
  [16, 0], [16, 0], [16, 4], [16, 4] for left, right). The LayerNorms have a unit offset folded
  in by the converter. small and medium have hidden 620 and 768 with a 512- and 640-wide attention
  (non-square q/k/v/o).
- Adapter: a learned position embedding is added, plus a projection when the encoder and decoder
  widths differ.
- Decoder: as Moonshine's, with partial RoPE 0.8, an untied `dec.lm_head`, pad id 0 and 4096
  positions. The output budget is 6.5 tokens/s of audio + 24.

It can share `src/moonshine/`'s decoder and tokenizer, with a second encoder. The one-shot path
fits today's batch API, which is how the reference's `transcribe_run` runs it. New pieces: masks
in both attention paths, a K tail in the GEMM (620, and 80 for the frontend Linear), self-attention
over up to 4096 keys, the time-domain frontend and the adapter. Real incremental streaming (feed,
partial results, finalize) needs a new public API, which `agents.md` reserves for the person's
approval; the roadmap lists streaming as "later, only on request". It is not in this plan.

## Line budget

Non-test lines of the core Moonshine uses, on main at the start: ~2.83k (`gguf.rs` 273,
`gpu/mod.rs` 903, `lib.rs` + `error.rs` 369; kernels `gemm` 217, `gemm_fast` 216, `ops` 198,
`ops_fast` 147, `attention` 122, `flash` 275, `decode` 111), already near the ~3k of
`agents.md`. Moonshine adds ~0.2k to the core (GEMM widths, tanh, `rope`, `group_norm`, `swiglu`,
`argmax`, `attend` blocks), the transformer pieces shared with Whisper, and ~0.4k in
`src/moonshine/`. Above ~3.2k for core + Moonshine, simplify before speed work.

## Risks

- Conv stem exactness: a 127-tap dot product over raw PCM per output, then tanh, then one
  GroupNorm reduction over ~0.8 M values for 11 s (3.5 M at 48 s), whose order differs from
  ggml's. Our GEMM rounds conv weights to f16 as `kernel_mul_mm` does. The F32 subsampling convs
  of GigaAM and Parakeet already match that way.
- Precision gaps in the reference itself: F32 caches and explicit softmax in the decoder on
  Metal, against our F16 caches with q rounded to f16; base's encoder attention on the CPU
  backend. Text fixtures tolerated Whisper's F16 gap, but greedy argmax over 32 768 logits can
  flip on a near-tie. Fallback: F32 caches without q rounding (2× the cache size, still small).
- Head dims 36 and 52 are no multiple of 8, so the flash kernel can't take them, and portable
  attention is O(T²) in memory on long single-pass input (as Parakeet's is today). The fast path
  zero-pads q/k/v to 40/56 per head with zero weight rows at load, which is exact and makes
  `[q | k | v]` 960 or 1344 wide (64-aligned). The out projection then needs either an unpad copy
  or `W_out` as F16 with zero columns. Q8_0 can't hold the zero columns, but F16 is exactly what
  our GEMM computes from Q8_0. That second option breaks "keep GGUF types"; decide on
  measurements.
- d not a multiple of 64 meets every kernel that assumed d ∈ {768, 1024, 1280}: GEMM outputs,
  gemv dispatch, LayerNorm's fast four-row path, the flash kernel's 64-row reads. Each needs a
  test at 288.
- Loops and the 194-token cap: out-of-distribution audio loops (silence, noise, the wrong
  language for a fine-tune). Matching it needs the guard ported exactly; fixture runs must end
  `run: OK`; a dense clip near 48 s can hit the cap.
- RoPE: interleaved, partial pairs, our CPU table against Metal's f32 `pow`/`cos`: drift at the
  last bit. Only a text change would matter.
- Tokenizer: byte fallback, literal control tokens, and only one leading space trimmed.

## Decisions

Decided with the person:

1. Tokens: every token gets `start_ms` 0 (one untimed segment, Whisper's rule), and fixtures hold
   text only, with no API change.
2. Repetition guard: ported. It is ~40 lines, deterministic, and matches the reference on
   looping audio, which the language fine-tunes hit on silence and other languages.
3. `max_audio_ms()`: `Some(30_000)`. It keeps a split piece well under the 194-token cap even at
   6 tokens/s, and the CLI then splits like Whisper. The library takes any length in one pass.
4. Registry: `moonshine-tiny` and `moonshine-base` ("English") first. The language fine-tunes
   come later as config, in a PR of their own (Task 7).
5. Streaming: a later, optional phase (Task 8), one-shot within the batch API. Real incremental
   streaming changes the public API, needs the person's approval and is out of scope.
6. CI: `moonshine-tiny` (35 MB) becomes CI's smallest model test (`CI_MODELS`, `CI_TESTS`).

## Tasks

### Task 1: plan and rules

- [x] This plan agreed, with decisions 1–6.
- [x] Roadmap phases 15 (tiny and base, correct), 16 (fast), 17 (language variants), 18
      (streaming one-shot, optional). `agents.md` needs nothing: `scripts/fixtures.sh` documents
      text-only fixtures and the rustdoc a Moonshine token's start time.

### Task 2: model and fixtures (phase 15)

- [x] `moonshine-tiny` (35 466 912 B) and `moonshine-base` (77 476 480 B), "English", in
      `hub::MODELS` and the lists it checks; `moonshine-tiny` in `CI_MODELS` and `CI_TESTS`;
      `make models`.
- [x] The published header checked against Model facts: keys, shapes, tensor types.
- [x] `make fixtures`: `TIMESTAMPS=none` gives a text-only file; the recipe fails when `run:` is
      not `ok`.
- [x] `make fixtures MODEL=<model> TIMESTAMPS=none SAMPLES="jfk dots jobs-silence"` for both,
      each generated twice; a clip that truncates or loops is replaced.
- [x] `tests/models.rs`: text-only fixtures; `moonshine_tiny` and `moonshine_base` over the three
      clips. Lands with Task 4.

### Task 3: shared core (GigaAM, Parakeet and Whisper output and speed unchanged)

- [x] GEMM and gemv with output widths that are multiples of 32 (masked stores, row stride n);
      tests at n 288 and 864, both paths.
- [x] `Epilogue::Tanh`, `im2col` padded columns; tests.
- [x] `rope`, `group_norm`, `swiglu`, `argmax`, each against a CPU loop on small odd shapes.
- [x] `attend` over more than 2048 keys; tests at 1, 37, 2049 and 4100 keys.
- [x] Pre-LN transformer pieces shared by Whisper and Moonshine; Whisper fixtures exact, warm
      `jfk` A/B/A unchanged.
- [x] All fixtures exact; `make check`; `make test`.

### Task 4: vertical slice `src/moonshine/` (correct)

- [x] `tokenizer.rs`: decode with byte fallback and pieces; unit tests with a character split over
      byte tokens and a control token.
- [x] `encoder.rs`, `mod.rs` (the decoder is the shared one); `Model` picks the family from
      `general.architecture`; `max_audio_ms()` per decision 3; no warm-up at load (no family keeps
      one since the 2026-10-11 load work).
- [x] Decode loop: prompt, EOS, the 194-position cap; repetition guard per decision 2 with
      scripted-id unit tests (stop bar, budget trim, emphatic repeats kept).
- [x] CLI: `--srt` per segment for Moonshine.
- [x] Adapted code names its source (`model.cpp` loop, `transcribe-repetition-guard.h`,
      `transcribe-tokenizer.cpp` decode, `rope.metal` angles).
- [x] Fixtures exact; `make check`; `make test`. On a mismatch compare with the reference dumps
      (`enc.conv1.out`, `enc.groupnorm.out`, `enc.conv2.out`, `enc.conv3.out`,
      `enc.block.N.out`, `enc.final`, `dec.block.N.out`, `dec.logits_raw`), then try the
      reference's precision: F32 caches without q rounding, then the encoder GEMMs' `a` to f16.
- [ ] Line count of core + Moonshine; above ~3.2k, simplify first.

### Task 5: first benchmark

- [x] `wavo bench -n 20` against `transcribe-bench --warmup 1 --iters 10` (Metal and CPU) on
      `jfk`, `dots` and `jobs-silence`, first call in a fresh process, peak from
      `/usr/bin/time -l`; the numbers in `## Result` and the PR.
- [x] `docs/perf.md`: how Moonshine runs; `readme.md`: model table, Token start times; tick
      roadmap phase 15. `make check`; `make test`.

### Task 6: fast (phase 16, next PR; detailed from the profile)

- [ ] Profile with GPU timestamps (temporary tool): conv stem, `group_norm`, GEMMs by kind,
      attention, LayerNorm; per step the kernels, submit and readback.
- [ ] Decoder: several steps per submit with `argmax` feeding `embed`; output identical.
- [ ] Encoder attention on the flash kernel: heads zero-padded to 40/56 at load; unpad copy
      against F16 `W_out`, measured.
- [ ] `swiglu` folded into the fc1 gemv and LayerNorm into the next gemv, if the profile asks.
- [ ] Load, first call, peak footprint.
- [ ] Decision point: if warm is still slower than the reference (Metal or CPU) once this list is
      exhausted, stop and report to the user before the variants.

### Task 7: language variants (phase 17)

- [ ] Language fine-tunes, tiny and base × `uk vi zh ja ko ar`: registry under their variant
      names (`moonshine-tiny-uk`, …), fixtures `<lang>-short <lang>-long`; config only.
- [ ] `readme.md`, roadmap phase 17.

### Task 8: streaming variants, one-shot (phase 18, optional, decision 5)

- [ ] `moonshine-streaming-tiny` through the batch API: time-domain frontend, sliding-window
      masks, adapter, untied head, 6.5 tokens/s budget; fixtures `jfk dots jobs-silence`.
- [ ] `moonshine-streaming-small` and `-medium`: GEMM K tails (620, 80), non-square attention;
      fixtures.

## Result

Phase 15: fixtures exact on the M2 (fast kernels), the RTX 4060 (portable kernels, Vulkan), the
N100 (portable, Vulkan) and the M1. Core + Moonshine is 3.71k non-test lines (core 3.02k, shared
transformer 0.28k, `src/moonshine/` 0.41k): the Task 4 simplification is still open.

Baseline, warm median of 20 calls (reference: 10), ms. The M2 was busy with another job (load
average 2–3), so its numbers are rough: the reference ran 2.8× slower there than its M4 Max
catalog numbers.

| Model, clip | M2 wavo | M2 Metal | M2 CPU | RTX 4060 wavo | RTX 4060 Vulkan |
|---|---|---|---|---|---|
| tiny `jfk` | **60** | 158 | 140 | **19** | 31 |
| tiny `dots` | **406** | 1475 | 997 | **142** | 144 |
| tiny `jobs-silence` | **30** | 76 | 51 | **12** | 19 |
| base `jfk` | **88** | 263 | 256 | **33** | 40 |
| base `dots` | **659** | 2853 | 2212 | 241 | **227** |
| base `jobs-silence` | **18** | 38 | 59 | **9** | 11 |

M2 load 24–38 ms against 47–68; peak footprint lower on short clips (tiny `jfk` 83 vs 118 MiB)
and higher on `dots` (tiny 187 vs 138, base 259 vs 236 MiB): portable attention keeps all
scores. Whisper unchanged: whisper-tiny `jfk` on the RTX 4060 main 50.0 / 51.2 / 51.5, branch
51.4 / 51.3 ms.
