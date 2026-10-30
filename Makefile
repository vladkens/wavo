.PHONY: prepare check test test-all build update clean

CARGO_FLAGS := --release --locked

prepare:
	cargo +nightly fmt
	cargo clippy $(CARGO_FLAGS) --fix --all-targets --allow-dirty --allow-no-vcs -- -D warnings
	cargo check $(CARGO_FLAGS)

check:
	cargo +nightly fmt --check
	uv run --no-project --python '>=3.11' python -m doctest scripts/size.py
	uv run scripts/size.py --check
	cargo clippy $(CARGO_FLAGS) --all-targets -- -D warnings
	cargo clippy $(CARGO_FLAGS) --no-default-features --all-targets -- -D warnings
	cargo clippy $(CARGO_FLAGS) --no-default-features --features hub --all-targets -- -D warnings
	cargo check $(CARGO_FLAGS)
	RUSTDOCFLAGS="-D warnings" cargo doc $(CARGO_FLAGS) --no-deps --no-default-features
	RUSTDOCFLAGS="-D warnings" cargo doc $(CARGO_FLAGS) --no-deps --no-default-features --features hub

# The smoke run: unit tests, the smoke models below and `wavo serve` end to end.
test: test-unit
	cargo test $(CARGO_FLAGS) --test models -- --exact $(SMOKE_TESTS)
	cargo test $(CARGO_FLAGS) --test serve

test-all:
	cargo test $(CARGO_FLAGS)

build:
	cargo build $(CARGO_FLAGS)
	ls -lh target/release/wavo

update:
	cargo upgrade --incompatible
	cargo update

clean:
	cargo clean

# wavo
.PHONY: test-unit models models-all samples samples-he reference fixtures compare bench-compare size

ALL_MODELS := gigaam-v3-e2e-rnnt gigaam-v3-e2e-ctc gigaam-v3-rnnt gigaam-v3-ctc \
	parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v3 parakeet-tdt-0.6b-v3-he parakeet-ultra \
	whisper-large-v3-turbo ivrit-whisper-large-v3-turbo whisper-tiny whisper-base whisper-small whisper-medium whisper-large-v3 whisper-tiny.en whisper-base.en whisper-small.en whisper-medium.en whisper-large whisper-large-v2 Breeze-ASR-25 \
	moonshine-tiny moonshine-base
# `make models` and `make test` (and CI) take only the smoke models, by these names in
# tests/models.rs: one per family, plus English-only Parakeet v2 next to multilingual v3.
# `make test-all` covers them all. CI caches the downloads under a key made from this line, so
# keep it on one line.
MODELS := gigaam-v3-e2e-rnnt parakeet-tdt-0.6b-v3 parakeet-tdt-0.6b-v2 whisper-tiny moonshine-tiny
SMOKE_TESTS := gigaam_v3_e2e_rnnt parakeet_tdt_v3 parakeet_tdt_v2 whisper_tiny moonshine_tiny
REFERENCE_REV := 5bb2deb2a4afb1fd50534ecb51cfcb521ef94944
# Whisper has segment timestamps only: TIMESTAMPS=segment; Moonshine none: TIMESTAMPS=none.
TIMESTAMPS := token

# Unit tests and doc examples only: no models or 3rd/ needed.
test-unit:
	cargo test $(CARGO_FLAGS) --lib --bins
	cargo test $(CARGO_FLAGS) --doc --no-default-features
	cargo test $(CARGO_FLAGS) --doc --no-default-features --features hub

# All at once with our own `wavo pull`. A second run waits on the first one's locks.
models: build
	target/release/wavo pull $(MODELS)

models-all:
	$(MAKE) models MODELS="$(ALL_MODELS)"

# Only the reference's samples/, for `make test` without building it.
samples:
	test -d 3rd/transcribe.cpp || { git clone --filter=blob:none --no-checkout --sparse \
		https://github.com/handy-computer/transcribe.cpp 3rd/transcribe.cpp \
		&& git -C 3rd/transcribe.cpp sparse-checkout set samples \
		&& git -C 3rd/transcribe.cpp checkout -q --detach $(REFERENCE_REV); }

# The Hebrew clips transcribe.cpp lacks, from FLEURS into 3rd/samples/ (see the script).
samples-he:
	scripts/samples-he.sh

reference:
	test -d 3rd/transcribe.cpp || git clone https://github.com/handy-computer/transcribe.cpp 3rd/transcribe.cpp
	git -C 3rd/transcribe.cpp sparse-checkout disable
	git -C 3rd/transcribe.cpp checkout --detach $(REFERENCE_REV)
	cmake -S 3rd/transcribe.cpp -B 3rd/transcribe.cpp/build -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON \
		"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=$(CURDIR)/3rd/transcribe.cpp/build/bin"
	cmake --build 3rd/transcribe.cpp/build --config Release --target transcribe-cli transcribe-bench -j

# make fixtures MODEL=gigaam-v3-e2e-rnnt SAMPLES="ru ru-short ru-long" (see the script)
fixtures:
	scripts/fixtures.sh "$(MODEL)" "$(TIMESTAMPS)" $(SAMPLES)

# make compare MODEL=gigaam-v3-e2e-rnnt LIST=wavs.txt OUT=dir [BACKEND=gpu|ane] (see the script)
compare:
	scripts/compare.sh "$(MODEL)" "$(LIST)" "$(OUT)" $(BACKEND)

# Coarse speed check against main's dev build, for CI (see the script).
bench-compare:
	scripts/ci-bench.sh

# Code, docs and tests per module; ARGS="--files", "--map" (each file and what it does), "--check" (the budgets) or "--vs main".
size:
	uv run scripts/size.py $(ARGS)
