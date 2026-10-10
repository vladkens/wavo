.PHONY: prepare check test test-unit build update clean models reference fixtures compare

CARGO_FLAGS := --release --locked
MODELS := gigaam-v3-e2e-rnnt gigaam-v3-e2e-ctc gigaam-v3-rnnt gigaam-v3-ctc \
	parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v3
REFERENCE_REV := 5bb2deb2a4afb1fd50534ecb51cfcb521ef94944
CLI := 3rd/transcribe.cpp/build/bin/transcribe-cli

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

# CI has no models: unit tests only.
test-unit:
	cargo test $(CARGO_FLAGS) --lib --bins

build:
	cargo build $(CARGO_FLAGS)
	ls -lh target/release/$(shell basename $(CURDIR))

update:
	cargo upgrade -i

clean:
	cargo clean

models:
	for m in $(MODELS); do hf download handy-computer/$$m-gguf $$m-Q8_0.gguf; done

reference:
	test -d 3rd/transcribe.cpp || git clone https://github.com/handy-computer/transcribe.cpp 3rd/transcribe.cpp
	git -C 3rd/transcribe.cpp checkout --detach $(REFERENCE_REV)
	cmake -S 3rd/transcribe.cpp -B 3rd/transcribe.cpp/build -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON
	cmake --build 3rd/transcribe.cpp/build --target transcribe-cli transcribe-bench -j

# make fixtures MODEL=gigaam-v3-e2e-rnnt SAMPLES="ru ru-short ru-long"
fixtures:
	@set -eu; model=$$(hf download handy-computer/$(MODEL)-gguf $(MODEL)-Q8_0.gguf | sed 's/^path=//'); \
	mkdir -p tests/fixtures/$(MODEL); \
	for s in $(SAMPLES); do \
		$(CLI) -m "$$model" --timestamps token 3rd/transcribe.cpp/samples/$$s.wav 2>/dev/null \
			| awk '/^text: /{print} /^tokens: /{t=1; print; next} t && /^  \[/{print}' \
			> tests/fixtures/$(MODEL)/$$s.txt; \
	done

# make compare MODEL=gigaam-v3-e2e-rnnt LIST=wavs.txt OUT=dir: every 16 kHz mono WAV in LIST (one
# path per line) through wavo (examples/batch.rs) and then the reference, one model load each, then
# speed and text agreement. When either engine fails on a file, the comparison still prints (the
# JSONL rows say which file) and make then fails.
compare:
	@set -eu; model=$$(hf download handy-computer/$(MODEL)-gguf $(MODEL)-Q8_0.gguf | sed 's/^path=//'); \
	cargo build $(CARGO_FLAGS) --example batch; mkdir -p $(OUT); \
	tr '\n' '\0' < $(LIST) | xargs -0 cat > /dev/null; \
	/usr/bin/time -l target/release/examples/batch "$$model" $(LIST) \
		> $(OUT)/wavo.jsonl 2> $(OUT)/wavo.time && wavo=0 || wavo=$$?; \
	/usr/bin/time -l $(CLI) -m "$$model" --batch $(LIST) --batch-jsonl \
		> $(OUT)/reference.jsonl 2> $(OUT)/reference.time && ref=0 || ref=$$?; \
	target/release/examples/batch compare $(OUT)/wavo.jsonl $(OUT)/reference.jsonl; \
	test $$wavo -eq 0 -a $$ref -eq 0
