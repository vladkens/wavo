.PHONY: prepare check test build dist update clean

CARGO_FLAGS := --release --locked

prepare:
	cargo +nightly fmt
	cargo clippy $(CARGO_FLAGS) --fix --all-targets --allow-dirty --allow-no-vcs -- -D warnings
	cargo check $(CARGO_FLAGS)

check:
	cargo +nightly fmt --check
	cargo clippy $(CARGO_FLAGS) --all-targets -- -D warnings
	cargo clippy $(CARGO_FLAGS) --no-default-features --all-targets -- -D warnings
	cargo check $(CARGO_FLAGS)

test:
	cargo test $(CARGO_FLAGS)

build:
	cargo build $(CARGO_FLAGS)
	ls -lh target/release/wavo

# Prebuilt binaries as target/distrib/wavo-dev-<target>.tar.gz (.zip for Windows), from an Apple
# Silicon Mac: Linux (glibc 2.28) and Windows through cargo-cross, installed at a fixed version.
# macOS keeps its file and line info in wavo.dSYM, which has to stay next to the binary. Not
# target/dist: cargo keeps the dist profile's build scripts there.
dist: T := $(or $(CARGO_TARGET_DIR),target)
dist:
	cargo install --locked cargo-cross@1.6.0
	CARGO_PROFILE_DIST_SPLIT_DEBUGINFO=packed cargo build --profile dist --locked --target aarch64-apple-darwin
	cargo cross build --profile dist --locked --glibc-version 2.28 \
		--targets x86_64-unknown-linux-gnu,aarch64-unknown-linux-gnu,x86_64-pc-windows-gnu
	rm -rf $(T)/distrib && mkdir -p $(T)/distrib
	tar czhf $(T)/distrib/wavo-dev-aarch64-apple-darwin.tar.gz readme.md LICENSE \
		-C $(T)/aarch64-apple-darwin/dist wavo wavo.dSYM
	for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do \
		tar czf $(T)/distrib/wavo-dev-$$t.tar.gz readme.md LICENSE -C $(T)/$$t/dist wavo || exit 1; done
	zip -jq $(T)/distrib/wavo-dev-x86_64-pc-windows-gnu.zip readme.md LICENSE \
		$(T)/x86_64-pc-windows-gnu/dist/wavo.exe
	ls -lh $(T)/distrib

update:
	cargo upgrade -i

clean:
	cargo clean

# wavo
.PHONY: test-unit test-ci models models-ci samples reference fixtures compare bench-compare

MODELS := gigaam-v3-e2e-rnnt gigaam-v3-e2e-ctc gigaam-v3-rnnt gigaam-v3-ctc \
	parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v3 \
	whisper-large-v3-turbo whisper-tiny whisper-base whisper-small whisper-medium whisper-large-v3 whisper-tiny.en whisper-base.en whisper-small.en whisper-medium.en whisper-large
# CI downloads and tests (by these names in tests/models.rs) only the smallest model of each
# family; switch when a smaller one is added. `make test` covers them all.
CI_MODELS := gigaam-v3-e2e-rnnt parakeet-tdt-0.6b-v3 whisper-tiny
CI_TESTS := gigaam_v3_e2e_rnnt parakeet_tdt_v3 whisper_tiny
REFERENCE_REV := 5bb2deb2a4afb1fd50534ecb51cfcb521ef94944
CLI := 3rd/transcribe.cpp/build/bin/transcribe-cli
# Whisper has segment timestamps only: TIMESTAMPS=segment.
TIMESTAMPS := token

# Unit tests only: no models or 3rd/ needed.
test-unit:
	cargo test $(CARGO_FLAGS) --lib --bins

test-ci: test-unit
	cargo test $(CARGO_FLAGS) --test models -- --exact $(CI_TESTS)

# All at once with our own `wavo pull`. A second run waits on the first one's locks.
models: build
	target/release/wavo pull $(MODELS)

models-ci:
	$(MAKE) models MODELS="$(CI_MODELS)"

# Only the reference's samples/, for `make test` without building it.
samples:
	test -d 3rd/transcribe.cpp || { git clone --filter=blob:none --no-checkout --sparse \
		https://github.com/handy-computer/transcribe.cpp 3rd/transcribe.cpp \
		&& git -C 3rd/transcribe.cpp sparse-checkout set samples \
		&& git -C 3rd/transcribe.cpp checkout -q --detach $(REFERENCE_REV); }

reference:
	test -d 3rd/transcribe.cpp || git clone https://github.com/handy-computer/transcribe.cpp 3rd/transcribe.cpp
	git -C 3rd/transcribe.cpp sparse-checkout disable
	git -C 3rd/transcribe.cpp checkout --detach $(REFERENCE_REV)
	cmake -S 3rd/transcribe.cpp -B 3rd/transcribe.cpp/build -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON \
		"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=$(CURDIR)/3rd/transcribe.cpp/build/bin"
	cmake --build 3rd/transcribe.cpp/build --config Release --target transcribe-cli transcribe-bench -j

# make fixtures MODEL=gigaam-v3-e2e-rnnt SAMPLES="ru ru-short ru-long"
fixtures:
	@set -eu; model=$$(hf download handy-computer/$(MODEL)-gguf $(MODEL)-Q8_0.gguf | sed 's/^path=//'); \
	mkdir -p tests/fixtures/$(MODEL); \
	for s in $(SAMPLES); do \
		$(CLI) -m "$$model" --timestamps $(TIMESTAMPS) 3rd/transcribe.cpp/samples/$$s.wav 2>/dev/null \
			| awk '/^text: /{print} /^(tokens|segments): /{t=1; print; next} t && /^  \[/{print}' \
			> tests/fixtures/$(MODEL)/$$s.txt; \
	done

# make compare MODEL=gigaam-v3-e2e-rnnt LIST=wavs.txt OUT=dir (see the script)
compare:
	scripts/compare.sh "$(MODEL)" "$(LIST)" "$(OUT)"

# Coarse speed check against main's dev build, for CI (see the script).
bench-compare:
	scripts/ci-bench.sh
