# wavo

Speech-to-text library in pure Rust with an ollama-like CLI. It runs the GGUF ASR models that
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

- Keep the library pure Rust: no C/C++, ONNX, BLAS or Python in it or its build. Its
  dependencies: `wgpu`, `bytemuck`, `half`, `thiserror`, `pollster`, `rustfft`; tests may also
  use `hound` (dev-dependency).
- The CLI sits behind the default `cli` feature (binary `wavo` in `src/cli/`, `required-features
  = ["cli"]`), so `cargo add wavo --no-default-features` gets only the library. The CLI's own
  dependencies are optional and enabled by `cli`: `anyhow`, `ureq` (rustls with ring and webpki
  roots), `symphonia`, `rubato`. ring compiles some C and assembly: the one accepted exception to
  pure Rust, and only in the CLI.
- Add, remove or upgrade dependencies only with `cargo add` / `cargo remove` / `cargo upgrade`
  (e.g. `cargo add wgpu@=30.0.1 --no-default-features --features std,wgsl,metal`, or
  `cargo add ureq --optional` for the CLI). Never edit `[dependencies]` or `Cargo.lock` by hand.
- Target Apple Silicon / Metal. The fast path may use subgroups and
  `EXPERIMENTAL_COOPERATIVE_MATRIX`. Keep one simple portable fallback per kernel and don't
  optimize it.
- Keep the public API to `Model::load(path)` and `model.transcribe(&pcm)` → `Transcript { text,
  tokens }`, where pcm is 16 kHz mono `f32` and tokens carry id, piece and start time in ms. The
  CLI is `wavo pull MODEL`, `wavo list`, `wavo rm MODEL`, `wavo run [MODEL] AUDIO [--json |
  --srt]` (default model `parakeet-v3`) and `wavo bench MODEL AUDIO [-n N]`, where MODEL is a
  short name, a full published name or a path to a `.gguf`. Change either only with the user's
  approval.
- Only `wavo pull` uses the network. `run` and `bench` never download; a missing model fails with
  the `wavo pull` command to run.
- The library leaves resampling, VAD, chunking and streaming to the caller. The CLI decodes
  common formats (wav, mp3, m4a/aac, flac, ogg/vorbis), downmixes to mono and resamples to 16 kHz;
  a 16 kHz mono WAV passes through unchanged. Neither does VAD, chunking or streaming.
- Whisper only on explicit request.

## Code

- Keep weights on the GPU in their GGUF type (Q8_0/F16/F32) and dequantize inside kernels. Expand
  to F32 at load only tiny tensors such as norms and biases. Accumulate in F32.
- Give each model family one module under `src/<family>/`. A variant of a family is config + head,
  not a copy. No generic graph runtime, plugin layers or disabled experiment code.
- Validate GGUF metadata and tensor shapes once at load. Keep hot paths free of defensive checks.
  Model files are trusted input: give clear errors for missing or mismatched tensors, and don't
  guard against crafted or corrupted files.
- Keep core plus one family (GGUF, GPU, kernels, frontend, encoder, decoder) within ~3k lines of
  Rust and WGSL, tests excluded. The CLI (`src/cli/`) is counted separately and stays lean,
  roughly ≤ 700 lines: hand-rolled argument parsing, JSON and SRT, no clap or serde. Simplify
  before growing past either budget.
- The library returns its own `Error` enum built with `thiserror`; only the CLI uses `anyhow`.
  Comment only non-obvious data layouts, numeric tricks and model quirks that differ from what the
  reference docs say.
- Start every `.rs` and `.wgsl` file with the line
  `// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo`.

## External sources

- Download models only into the shared Hugging Face cache, with `wavo pull <name>` or
  `hf download handy-computer/<name>-gguf <name>-Q8_0.gguf` (install with `brew install hf`);
  `make models` runs `hf` for every supported model. Never copy model files into the repository.
  The cache follows the Hub local cache spec (https://huggingface.co/docs/hub/local-cache), with
  the root resolved as Python huggingface_hub does: `HF_HUB_CACHE`, else `$HF_HOME/hub`, else
  `${XDG_CACHE_HOME:-~/.cache}/huggingface/hub`; a model is at
  `models--handy-computer--<name>-gguf/snapshots/<refs/main>/<name>-Q8_0.gguf`, a symlink into
  `blobs/`. `wavo pull` writes exactly that layout (lock in `.locks/`, `blobs/<etag>.incomplete`
  while downloading), so `hf` and `wavo` each find what the other downloaded. `wavo rm` deletes
  only inside the repo folder and leaves `hf`'s shared blob store to `hf cache rm`.
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

1. Find the variant in `3rd/transcribe.cpp/catalog/<variant>.json` (published repo, files and
   sizes). Add its name to `MODELS` in the `Makefile` and to the CLI registry in `src/cli/` (short
   name, Q8_0 size), and run `make models`.
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
  against a CPU loop on small shapes; in the CLI, model names, the cache layout, audio decoding (a
  16 kHz mono WAV gives exactly the samples the fixtures use) and SRT cues. Unit tests run in CI
  (`make test-unit`) without models or `3rd/`: synthesize their inputs.

## Performance

- Read [docs/perf.md](docs/perf.md) before speed work. Log every attempt there, kept or rejected.
- Compare `wavo bench` with the reference `transcribe-bench` on the same GGUF and WAV: at least 10
  warm calls, median and min. Measure a change A/B/A back to back. When the two A runs differ by
  more than 3%, rerun later instead of concluding. Run one GPU job at a time.
- Keep a speed change only when fixtures still match and the gain is clear. Delete rejected code.

## Workflow

- Before committing, run `make check` (it also checks the library alone, with
  `--no-default-features`) and `make test`. `make prepare` rewrites files, so don't use it as a
  check.
- Every change reaches `main` through a pull request, one per feature or roadmap phase:
  1. Agree the plan with the person. For multi-step work keep a checklist in
     `docs/plans/yyyymmdd-<name>.md`; in it, only tick checkboxes: no evidence, progress or status
     prose.
  2. Branch from `main`. Each agent works in its own git worktree, so parallel tasks don't touch
     each other's files.
  3. Commit on the branch every step that passes `make check` and `make test`: one short lowercase
     line, no body, roadmap/docs updated in the same commit. Any agent on the task may commit;
     fix-ups are fine, the branch is squashed.
  4. Push the branch and open the PR with `gh pr create`. The title is the squash commit: one short
     lowercase line. The body is short: what changed, the checks run, numbers for speed work.
  5. Whoever delegated the work (the orchestrating agent) reviews the diff, reruns the checks and
     sends findings back to the implementing agent until the PR is clean, then hands it to the
     person. CI runs only `make check` and `make test-unit` (no models there), so `make test` and
     benchmarks stay local.
  6. The person does the final review and merges: squash, one commit on `main`, the branch is
     deleted. Never commit or push to `main` directly.
  7. On conflicts, the orchestrator (or an agent it asks) rebases the branch onto `main`, reruns the
     checks and pushes with `--force-with-lease`.
- An orchestrating agent writes each task, verifies every report itself (`make check`, `make test`,
  the diff, benchmarks) instead of trusting it, and keeps the person informed.
- Build in vertical slices: the first commit for a model already produces a transcript; speed
  and polish come after.
- Ask the user only when output must change, the public API must break, a heavy dependency is
  needed, or the request is ambiguous. Otherwise decide and say so in your report.
- Keep docs to `readme.md` (for users), `docs/perf.md`, `docs/roadmap.md` and `docs/plans/`. No
  progress journals, hash receipts, run logs or disclaimer paragraphs.
