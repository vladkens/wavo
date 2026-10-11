# wavo

Speech-to-text in pure Rust, a library and an ollama-like CLI that run transcribe.cpp's GGUF
models with the encoder on the GPU. It must be at least as fast as transcribe.cpp, and faster on
GigaAM v3 e2e-rnnt, the primary model; dictation loads a model per recording, so load time and
the first call matter as much as warm speed. Agents carry out each change end to end by following
this file; plan in [docs/roadmap.md](docs/roadmap.md).

Each fact has one source, which this file points to: the public API is the rustdoc, the CLI
`wavo --help`, the models `hub::MODELS`, features and dependencies `Cargo.toml`, checks and CI
the `Makefile` and `.github/workflows/main.yml`.

## Scope

- Keep the library pure Rust: no C/C++, ONNX, BLAS or Python in it or its build. The one
  exception is `ring` (under `ureq`'s rustls), only in the CLI and the `hub` feature.
- Keep the build fast, also for crates that depend on wavo: prefer a small own routine to a
  generic dependency that compiles far more code than wavo uses (`cargo build --timings`). Change
  dependencies only with `cargo add` / `cargo remove` / `cargo upgrade`, never by hand.
- Target Apple Silicon / Metal; Linux and Windows run through Vulkan. Every kernel has one simple
  portable version; a fast tier per hardware class, chosen at load from adapter features, stays
  only with a measured gain on that class.
- The public API is what the rustdoc documents, and the CLI is what `wavo --help` shows: change
  either only with the person's approval. Every public struct and enum is `#[non_exhaustive]`.
- Only `wavo pull` and `Hub::pull` use the network; `run` and `bench` never download. `pull` blocks:
  the library starts no download threads.
- The library leaves resampling and splitting long audio at pauses to the caller (Whisper's own
  30 s windows aside); the CLI does both. VAD and streaming are in neither.

## Code

- Keep weights on the GPU in their GGUF type (Q8_0/F16/F32) and dequantize inside kernels. Expand
  to F32 at load only tiny tensors such as norms and biases. Accumulate in F32.
- One module per model family, `src/<family>/`; a variant is config + head, not a copy. No
  generic graph runtime, plugin layers or disabled experiment code.
- Validate GGUF metadata and tensor shapes once at load, with clear errors. Model files are
  trusted: no guards against crafted files, no defensive checks in hot paths.
- Keep core plus one family (GGUF, GPU, kernels, frontend, encoder, decoder) within ~3k lines of
  Rust and WGSL, tests excluded. The CLI (`src/cli/`) stays roughly ≤ 900 lines: hand-rolled
  argument parsing, JSON and SRT, no clap or serde. `src/hub/` stays roughly ≤ 600 lines with its
  docs. Simplify before growing past any budget.
- Every public item has a doc comment with its units and failure cases; examples are doctests.
- The library returns its own `thiserror` `Error`; only the CLI uses `anyhow`. Comment only
  non-obvious data layouts, numeric tricks and model quirks that differ from the reference docs.
- An encoder records its layers through `Pass::layers`, which submits each layer on its own on the
  portable path. Some drivers (i915 on Linux) cancel GPU work that runs over ~20 s and still report
  success, and a slow GPU takes that long for one long call. With the fast kernels (Apple) the
  helper keeps one submission, so it costs nothing there; a new model's encoder must loop through
  it.
- Start every `.rs` and `.wgsl` file with the line
  `// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo`.
- Put helper scripts in `scripts/`, named in kebab-case (`compare.sh`, `make-fixtures.py`): `sh`,
  or Python run with `uv run`. A Makefile target longer than a line or two calls a script.

## External sources

- Keep models only in the shared Hugging Face cache (`make models`), never in the repository.
  Follow the Hub cache spec as `src/hub/cache.rs` documents it; `rm` never touches `hf`'s shared
  blob store.
- Keep third-party checkouts in `3rd/` (gitignored) and never modify them. The reference is
  transcribe.cpp at the Makefile's `REFERENCE_REV`, built by `make reference`; use it to read
  source, generate fixtures and benchmark, never as a dependency.
- The `stt.*` GGUF keys, tensor names and bundled frontend tensors are transcribe.cpp's own
  schema: its `scripts/convert-<family>.py` is the spec.
- When adapting code from `3rd/` (MIT), name the source file in a comment next to it and add the
  upstream copyright line to `LICENSE` once.
- Take test audio from `3rd/transcribe.cpp/samples/`; never copy audio into the repository.
- The `research` branch is read-only reference from an earlier, bloated attempt: read it with
  `git show research:<path>` and port ideas. Never merge it.

## Adding a model

1. Find the variant in `3rd/transcribe.cpp/catalog/<variant>.json` and add it to `hub::MODELS`;
   `make test-unit` then names every other list to update. Run `make models`.
2. Read `scripts/convert-<family>.py`, `src/arch/<family>/` and `docs/models/<variant>.md` in the
   reference.
3. Generate fixtures with `make fixtures MODEL=<name> SAMPLES="..."`: 2–4 clips in the model's
   language (short, ~11 s, long, plus silence where relevant).
4. Implement it: config + head in an existing family, or a new `src/<family>/`.
5. Pass the fixtures exactly, then benchmark against the reference. If wavo is slower, work
   through `docs/perf.md`, or report the gap to the person.
6. Put the baseline in the plan's Result and the PR body, and update `docs/roadmap.md`.

## Correctness

- Every model has fixtures in `tests/fixtures/<model>.txt`, one block per clip, written from the
  reference CLI by `make fixtures` (`scripts/fixtures.sh` is the format's spec). `make test` must
  reproduce text, tokens and start times exactly, and must fail (not skip) when a model, sample
  or block is missing.
- Treat any fixture change as a bug. Changing arithmetic (e.g. F16 operands) is fine while the
  fixtures stay exact. If a speed or memory gain changes a fixture, show the person the fixture
  diff and the gain before committing. A PR may add blocks but must not change an existing one
  without the person's approval.
- Debug mismatches by comparing intermediate tensors with the reference's own dumps
  (`TRANSCRIBE_DUMP_DIR=<dir>`; per-block dumps need a `-DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON`
  build). Don't commit dumps or tensor-level oracle tests, and don't chase bit-exact
  intermediates.
- Unit-test only logic that can break silently (parsing, tokenizer, decoder loops, kernels against
  a CPU loop on small shapes, model names, the cache layout, audio decoding, the long-audio split,
  SRT cues). Unit tests run in CI without models or `3rd/`: synthesize their inputs.

## Performance

- Read [docs/perf.md](docs/perf.md) before speed work and log every attempt there: what was
  tried, the result, kept or reverted and why. No benchmark tables: a task's numbers go in its
  plan's Result and the PR body, and the readme's Speed table holds the current ones for users.
- Profile with GPU timestamp queries in a temporary local tool. Never run Instruments or `xcrun
  xctrace` (Metal System Trace): on this Mac it froze the window server twice and needed a hard
  reset.
- Compare `wavo bench` with the reference `transcribe-bench` on the same GGUF and WAV: at least 10
  warm calls, median and min. Measure a change A/B/A back to back. When the two A runs differ by
  more than 3%, rerun later instead of concluding. Run one GPU job at a time.
- Keep a speed change only when fixtures still match and the gain is clear. Delete rejected code.
- For text agreement and speed on many recordings, use `make compare`.

## Workflow

- Before committing, run `make check` and `make test`. `make prepare` rewrites files: not a check.
- Every change reaches `main` through a pull request, one per feature or roadmap phase:
  1. Agree the plan with the person. For multi-step work keep a checklist in
     `docs/plans/yyyymmdd-<name>.md` and during the work only tick its checkboxes, with no
     progress prose. When finished, add a first line `Done: yyyy-mm-dd, PR #N` and a short final
     `## Result` section with the key numbers. Plans stay; never delete them.
  2. Branch from `main` as `feat/<name>`, in the agent's own git worktree. A small change that
     fits an open PR's topic goes into that PR instead of a new one: docs-only edits join the open
     docs PR.
  3. Commit every step that passes `make check` and `make test`: one short lowercase line, no
     body, docs updated in the same commit. The branch is squashed.
  4. Push and open the PR with `gh pr create`. The title is the squash commit, one short
     lowercase line; the body says what changed, the checks run and numbers for speed work.
  5. The orchestrating agent reviews the diff, reruns the checks and sends findings back until
     the PR is clean, then hands it to the person. CI runs only the smallest model of each family,
     so `make test` and benchmarks stay local. Copilot reviews every PR: fix what is right, answer
     the rest in the thread. Once the orchestrator accepts the solution, the implementing agent
     runs `codex -C <worktree> review --base origin/main -c model_reasoning_effort=xhigh` if
     available and fixes or rejects each finding with a reason; the orchestrator checks that last
     diff before handing the PR over.
  6. The person does the final review and squash-merges. Never commit or push to `main` directly.
  7. On conflicts, rebase onto `main`, rerun the checks and push with `--force-with-lease`.
- A release goes through a PR too: a branch with one `release: vX.Y.Z` commit (changelog
  written by the person's skill, version bumped), squash-merged; CI then tags it, publishes to
  crates.io and creates the GitHub release.
- An orchestrating agent writes each task, verifies every report itself (`make check`, `make test`,
  the diff, benchmarks) instead of trusting it, and keeps the person informed.
- Report a blocker as soon as you hit it: an unreachable machine, a missing permission or tool, a
  failing service, or anything else that stops the task. Tell whoever delegated the task right
  away (a subagent messages the orchestrator), say what can still be done without it, and keep
  doing that part. Never wait or poll silently for more than a few minutes.
- Build in vertical slices: a model's first commit already produces a transcript.
- Ask the person only when output must change, the public API must break, a heavy dependency is
  needed, or the request is ambiguous. Otherwise decide and say so in your report.
- Keep docs to `readme.md` (users), the rustdoc (the library's user guide), `docs/perf.md`,
  `docs/roadmap.md` and `docs/plans/`. No progress journals, hash receipts, run logs or
  disclaimer paragraphs. `changelog.md` is written only in a release commit, by the person's
  changelog skill from the git history: don't edit it in feature work.
