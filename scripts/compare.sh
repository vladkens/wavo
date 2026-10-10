#!/bin/sh
# Usage: scripts/compare.sh MODEL LIST OUT (or make compare MODEL= LIST= OUT=)
#
# Every 16 kHz mono PCM16 WAV in LIST (one path per line) through wavo (examples/batch.rs) and then
# transcribe.cpp's batch mode, one model load each, then speed and text agreement. When either
# engine fails on a file, the comparison still prints (the JSONL rows say which file) and the script
# then fails. `/usr/bin/time -l` is macOS's.
set -eu
model_name=$1 list=$2 out=$3
model=$(hf download "handy-computer/$model_name-gguf" "$model_name-Q8_0.gguf" | sed 's/^path=//')
cargo build --release --locked --example batch
mkdir -p "$out"
tr '\n' '\0' < "$list" | xargs -0 cat > /dev/null # into the page cache, so neither run reads disk

/usr/bin/time -l target/release/examples/batch "$model" "$list" \
  > "$out/wavo.jsonl" 2> "$out/wavo.time" && wavo=0 || wavo=$?
/usr/bin/time -l 3rd/transcribe.cpp/build/bin/transcribe-cli -m "$model" --batch "$list" \
  --batch-jsonl > "$out/reference.jsonl" 2> "$out/reference.time" && ref=0 || ref=$?
target/release/examples/batch compare "$out/wavo.jsonl" "$out/reference.jsonl"
test $wavo -eq 0 -a $ref -eq 0
