# wavo

Speech-to-text library and small CLI in pure Rust. It runs the GGUF ASR models that
transcribe.cpp publishes as `handy-computer/*-gguf` on Hugging Face: encoder on the GPU through
`wgpu` with our own WGSL kernels, frontend and decoder on the CPU. Plan:
[docs/roadmap.md](docs/roadmap.md).

Coding agents operate this repository. A person asks for a change (for example "add model X" or
"make GigaAM faster") and the agent carries it out end to end by following this file.

Why it exists next to transcribe.cpp (C/C++ on ggml): a plain `cargo build` with no cmake, C++
toolchain or Metal shader compilation, and a small engine for a few models that agents fully
understand (transcribe.cpp has ~13k lines for GigaAM + Parakeet on top of ggml). It must be at
least as fast as transcribe.cpp on the same files, and faster for GigaAM v3 e2e-rnnt, the
primary model. Typical use is dictation, where a model is loaded per recording, so load time and
the first call matter as much as warm speed.

## Scope

- Keep the crate pure Rust: no C/C++, ONNX, BLAS or Python in the crate or its build. Expected
  dependencies: `wgpu`, `bytemuck`, `half`, `thiserror`, `anyhow`, `pollster`, `rustfft`, `hound`.
- Add, remove or upgrade dependencies only with `cargo add` / `cargo remove` / `cargo upgrade`
  (e.g. `cargo add wgpu@=30.0.1 --no-default-features --features std,wgsl,metal`). Never edit
  `[dependencies]` or `Cargo.lock` by hand.
- Target Apple Silicon / Metal. The fast path may use subgroups and
  `EXPERIMENTAL_COOPERATIVE_MATRIX`. Keep one simple portable fallback per kernel and don't
  optimize it.
- Keep the public API to `Model::load(path)` and `model.transcribe(&pcm)` → `Transcript { text,
  tokens }`, where pcm is 16 kHz mono `f32` and tokens carry id, piece and start time in ms. The
  CLI is `wavo MODEL.gguf AUDIO.wav [--tokens]` and `wavo bench MODEL.gguf AUDIO.wav [-n N]`.
  Change either only with the user's approval.
- Leave resampling, VAD, chunking and streaming to the caller.
- Whisper only on explicit request.

## Code

- Keep weights on the GPU in their GGUF type (Q8_0/F16/F32) and dequantize inside kernels. Expand
  to F32 at load only tiny tensors such as norms and biases. Accumulate in F32.
- Give each model family one module under `src/<family>/`. A variant of a family is config + head,
  not a copy. No generic graph runtime, plugin layers or disabled experiment code.
- Validate GGUF metadata and tensor shapes once at load. Keep hot paths free of defensive checks.
  Model files are trusted input: give clear errors for missing or mismatched tensors, and don't
  guard against crafted or corrupted files.
- Keep core plus one family (GGUF, GPU, kernels, frontend, encoder, decoder, CLI) within ~3k lines
  of Rust and WGSL, tests excluded. Simplify before growing past that.
- The library returns its own `Error` enum built with `thiserror`; only the CLI uses `anyhow`.
  Comment only non-obvious data layouts, numeric tricks and model quirks that differ from what the
  reference docs say.
- Start every `.rs` and `.wgsl` file with the line
  `// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo`.

## External sources

- Download models only into the shared Hugging Face cache with
  `hf download handy-computer/<name>-gguf <name>-Q8_0.gguf` (install the CLI with
  `brew install hf`); `make models` does this for every supported model. Never copy model files
  into the repository. Code finds a model at
  `${HF_HOME:-~/.cache/huggingface}/hub/models--handy-computer--<name>-gguf/snapshots/*/<name>-Q8_0.gguf`.
- Keep third-party checkouts in `3rd/` (gitignored) and never modify them. The reference is
  transcribe.cpp at commit `5bb2deb2a4afb1fd50534ecb51cfcb521ef94944`; `make reference` clones and
  builds it in `3rd/transcribe.cpp`. Use it to read source, generate fixtures and benchmark, never
  as a dependency.
- The GGUF container is standard (written with llama.cpp's `gguf-py`; `general.*` and
  `tokenizer.ggml.*` keys), but the `stt.*` keys, the tensor names and the bundled frontend
  tensors (window, mel filterbank) are transcribe.cpp's own schema. Its
  `scripts/convert-<family>.py` is the spec. The published Q8_0 files are the converter's F32
  output passed through `tools/transcribe-quantize`.
- When adapting code from `3rd/` (ggml and transcribe.cpp are MIT), name the source file in a
  comment next to the adapted code and add the upstream copyright line to `LICENSE` once.
- Take test audio from `3rd/transcribe.cpp/samples/`. Don't copy audio into the repository.
- If a `research` branch exists, treat it as read-only reference from an earlier, bloated attempt:
  read code with `git show research:<path>` and port ideas and kernels. Never merge it.

## Adding a model

1. Find the variant in `3rd/transcribe.cpp/catalog/<variant>.json` (published repo and files).
   Add its name to `MODELS` in the `Makefile` and run `make models`.
2. Read `scripts/convert-<family>.py`, `src/arch/<family>/` and `docs/models/<variant>.md` in the
   reference.
3. Generate fixtures with `make fixtures MODEL=<name> SAMPLES="..."`: 2–4 clips from
   `3rd/transcribe.cpp/samples/` in the model's language (short, ~11 s, long, plus silence where
   relevant).
4. Implement it: config + head inside an existing family, or a new `src/<family>/` that reuses
   the GGUF reader, GPU core and kernels.
5. Pass the fixtures exactly, then benchmark against the reference. If wavo is slower, work
   through `docs/perf.md`, or report the gap to the user.
6. Log the baseline in `docs/perf.md`, add the model to `readme.md`, and update
   `docs/roadmap.md`.

## Correctness

- Give every supported model fixtures in `tests/fixtures/<model>/`, generated by `make fixtures`
  from the reference CLI (text plus every token with its start time). `make test` must reproduce
  text, tokens and start times exactly, and must fail (not skip) when a model or sample is missing.
- Treat any fixture change as a bug. Changing arithmetic (e.g. F16 operands) is fine while the
  fixtures stay exact. If a speed or memory gain changes a fixture, show the user the fixture diff
  and the gain before committing.
- Debug mismatches by comparing intermediate tensors with the reference's own dumps
  (`TRANSCRIBE_DUMP_DIR=<dir>`; per-block dumps need a `-DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON`
  build in `3rd/`). Don't commit dumps or tensor-level oracle tests, and don't chase bit-exact
  intermediates.
- Unit-test only logic that can break silently: GGUF parsing, tokenizer, decoder loops, kernels
  against a CPU loop on small shapes.

## Performance

- Read [docs/perf.md](docs/perf.md) before speed work. Log every attempt there, kept or rejected.
- Compare `wavo bench` with the reference `transcribe-bench` on the same GGUF and WAV: at least 10
  warm calls, median and min. Measure a change A/B/A back to back. When the two A runs differ by
  more than 3%, rerun later instead of concluding. Run one GPU job at a time.
- Keep a speed change only when fixtures still match and the gain is clear. Delete rejected code.

## Workflow

- Before committing, run `make check` and `make test`. `make prepare` rewrites files, so don't use
  it as a check.
- Make each commit one meaningful change a person can check. The message is one short lowercase
  line, no body. Update the roadmap/docs in the same commit. Push only when asked.
- When fixing your own unpushed commit (wrong rule, typo, forgotten file), amend or fix up that
  commit. Never add a separate "update agents.md" / "fix plan" commit. After a push, make a normal
  commit with a real reason.
- Build in vertical slices: the first commit for a model already produces a transcript; speed
  and polish come after.
- Develop a feature on its own branch and commit there every step that passes `make check` and
  `make test`. The person reviews the branch; it lands on main as one squashed commit per roadmap
  phase, and then the branch is deleted.
- For multi-step work, keep a checklist plan in `docs/plans/yyyymmdd-<name>.md`. In it, only tick
  checkboxes: no evidence, progress or status prose.
- Ask the user only when output must change, the public API must break, a heavy dependency is
  needed, or the request is ambiguous. Otherwise decide and say so in your report.
- Keep docs to `readme.md` (for users), `docs/perf.md`, `docs/roadmap.md` and `docs/plans/`. No
  progress journals, hash receipts, run logs or disclaimer paragraphs.
