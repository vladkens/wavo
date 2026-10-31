# wavo

Speech-to-text in pure Rust: a library and an ollama-like CLI that run transcribe.cpp's GGUF models with the encoder on the GPU, or on a Mac's Neural Engine for Parakeet and GigaAM. It must be at least as fast as transcribe.cpp, and faster on GigaAM v3 e2e-rnnt, the primary model; dictation loads a model per recording, so load time and the first call matter as much as warm speed.

Each fact has one source: the public API is the rustdoc, the CLI `wavo --help`, the models `hub::MODELS`, features and dependencies `Cargo.toml`, checks and CI the `Makefile` and `.github/workflows/main.yml`, speed work [docs/perf.md](docs/perf.md), the plan [docs/roadmap.md](docs/roadmap.md) and `docs/plans/`.

## Boundaries

- Keep the library pure Rust: no C/C++, ONNX, BLAS or Python in it or its build. The one exception is `ring` (under `ureq`'s rustls), only in the CLI and the `hub` feature.
- Keep the build fast, also for crates that depend on wavo: prefer a small own routine to a generic dependency that compiles far more code than wavo uses (`cargo build --timings`), and add a heavy dependency only with the person's approval. Change dependencies only with `cargo add` / `cargo remove` / `cargo upgrade`, never by hand.
- Change the public API (what the rustdoc documents) or the CLI (what `wavo --help` shows and its output formats: JSON and SRT fields, `serve` responses) only with the person's approval. Every public struct and enum is `#[non_exhaustive]`.
- Only `wavo pull`, `wavo run`, `wavo serve` and `Hub::pull` use the network: `run` downloads a missing model only after a confirmation on a terminal or with `-y`, and `bench` never does; `serve` downloads only its startup models, as `run` does, never for a request; `pull` blocks, and the library starts no download threads.
- The library leaves resampling to the caller (the CLI does it) and has no VAD or streaming.

## Code

- Target Metal on Apple Silicon and Vulkan on Linux and Windows; Windows without a Vulkan adapter falls back to DX12, which runs only the portable kernels, so they must compile with FXC. Every kernel has one simple portable version; a fast tier per hardware class, chosen at load from adapter features, stays only with a measured gain on that class.
- Keep weights on the GPU in their GGUF type (Q8_0/F16/F32) and dequantize inside kernels; expand to F32 at load only tiny tensors such as norms and biases, and accumulate in F32.
- One module per model family, `src/<family>/`; a variant is config + head, not a copy. No generic graph runtime, plugin layers or disabled experiment code.
- Validate GGUF metadata and tensor shapes once at load, with clear errors. Model files are trusted: no guards against crafted files, no defensive checks in hot paths.
- A new model's encoder records its layers through `Pass::layers` (drivers like i915 cancel GPU work over ~20 s).
- Each part has a code budget, tests, docs and comments excluded (`BUDGETS` in `scripts/size.py`: the core, the kernels, each family, `src/ane/`, `src/hub/`, and `src/cli/` in three: its core, `debug.rs`, and its server (`serve.rs`, `http.rs`)), checked by `make size ARGS=--check` (and `make check`); simplify before growing past one, and raising one is a decision recorded in the PR. The CLI hand-rolls argument parsing, JSON, SRT and HTTP (on `std::net` with `httparse`): no clap, serde, HTTP framework or async runtime.
- Public items document their units and failure cases, with examples as doctests.
- Start every `.rs`, `.wgsl` and `.metal` file with `// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo`.
- Helper scripts: `scripts/`, kebab-case, `sh` or Python with `uv run`.

## External sources

- Keep models only in the shared Hugging Face cache (`make models`), never in the repository.
- Keep third-party checkouts in `3rd/` (gitignored) and never modify them. The reference is transcribe.cpp at the Makefile's `REFERENCE_REV`, built by `make reference`: use it to read source, generate fixtures and benchmark, never as a dependency. Its `scripts/convert-<family>.py` is the spec of the `stt.*` GGUF keys, tensor names and bundled frontend tensors.
- When adapting code from `3rd/` (MIT), name the source file in a comment next to it and add the upstream copyright line to `LICENSE` once.
- Take test audio from `3rd/transcribe.cpp/samples/`, or `3rd/samples/` for clips it lacks (`make samples-he`); never copy audio into the repository.
- The `research` branch is a read-only reference from an earlier, bloated attempt: read it with `git show research:<path>` and port ideas; never merge it.

## Adding a model

1. Add the variant from `3rd/transcribe.cpp/catalog/<variant>.json` to `hub::MODELS`; `make test-unit` then names every other list to update. Run `make models MODELS=<name>`.
2. Read the reference's `scripts/convert-<family>.py`, `src/arch/<family>/` and `docs/models/<variant>.md`.
3. Generate fixtures with `make fixtures MODEL=<name> SAMPLES="..."`: 2–4 clips in the model's language (short, ~11 s, long, plus silence where relevant). The model's first commit already produces a transcript.
4. Pass the fixtures of every model in the family exactly, then benchmark against the reference; if wavo is slower, work through `docs/perf.md` or report the gap to the person.
5. A new family gets its own budget in `BUDGETS` in `scripts/size.py`: its code (`make size`) rounded up to the next hundred.
6. Update `docs/roadmap.md`.

## Correctness

- Every model has fixtures in `tests/fixtures/<model>.txt`, one block per clip, written from the reference CLI by `make fixtures` (`scripts/fixtures.sh` is the format's spec). `make test-all` must reproduce text, tokens and start times exactly, and fail (not skip) when a model, sample or block is missing; `make test` does the same for the smoke models (the Makefile's `MODELS`).
- Treat any fixture change as a bug: arithmetic may change (e.g. F16 operands) while the fixtures stay exact. A PR may add blocks, but changes an existing one only with the person's approval, asked before committing, with the fixture diff and the gain it buys.
- The Neural Engine is close to the GPU, not exact: its `*_ane` tests (macOS, `make test-all`) hold each clip within `check_ane`'s tolerances in `tests/models.rs`, while the GPU's fixtures stay exact. When ANE code or a model changes, run `make compare BACKEND=ane` and `BACKEND=gpu` on an hour of speech cut at pauses; `target/release/examples/batch compare` of their two `wavo.jsonl` must give a WER ≤ 0.2% and the same text on ≥ 95% of the 2–15 s pieces.
- Debug mismatches against the reference's own dumps (`TRANSCRIBE_DUMP_DIR`); don't commit dumps or tensor-level oracle tests, and don't chase bit-exact intermediates.
- Unit-test only logic that can break silently; unit tests run in CI without models or `3rd/`, so synthesize their inputs.

## Performance

- Read `docs/perf.md` before speed work: it says how to measure and where each attempt and number goes. Profile with GPU timestamp queries in a temporary local tool.
- Keep a speed change only when `make test-all` passes and the gain is clear; delete rejected speed code.

## Workflow

- `make check` and `make test` (the smoke run) pass before every commit. `make test-all` (after `make models-all`, `make samples` and `make samples-he`) also passes before a release and before merging speed work or a change to GPU kernels, `src/gpu/`, the GGUF reader, or shared frontend or decoder numerics.
- Agree a multi-step change's plan first; it keeps a checklist in `docs/plans/yyyymmdd-<name>.md`, and during the work only tick its checkboxes, with no progress prose. The PR that finishes its last item adds a `Done: yyyy-mm-dd, PR #N` line under the title and a short final `## Result` section with the key numbers, and moves it to `docs/plans/completed/`, fixing links to it. Never delete plans.
- Every change reaches `main` through a pull request from a `feat/<name>` branch, one per feature or roadmap phase; never commit or push to `main` directly. The maintainer squash-merges it.
- A commit, and the PR title as the squash commit, is one short lowercase line with no body; docs change in the same commit. A speed PR's body gives the numbers.
- A release is a PR with one `release: vX.Y.Z` commit that bumps the version and writes `changelog.md` from the git history; `changelog.md` changes in no other commit. After the squash-merge CI tags it, publishes to crates.io and creates the GitHub release.
- Keep docs to `readme.md` (users), the rustdoc (the library's user guide), `docs/perf.md`, `docs/roadmap.md` and `docs/plans/`; no progress journals, benchmark tables, hash receipts, run logs or disclaimer paragraphs.
