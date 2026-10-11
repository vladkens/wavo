#!/bin/sh
# Usage: scripts/fixtures.sh MODEL TIMESTAMPS SAMPLE... (or make fixtures MODEL= SAMPLES= [TIMESTAMPS=])
#
# Runs the reference CLI on 3rd/transcribe.cpp/samples/SAMPLE.wav and writes its transcript as the
# block `## SAMPLE` of tests/fixtures/MODEL.txt: the other blocks stay byte for byte, all sorted by
# name and separated by one blank line. A body line starts with `text: `, `tokens: `, `segments: `
# or two spaces, so no transcript reads as a header. Writes nothing if the reference fails or
# prints anything else.
set -eu
[ $# -ge 3 ] || { echo "usage: $0 MODEL TIMESTAMPS SAMPLE..." >&2; exit 2; }
name=$1 timestamps=$2
shift 2
cli=3rd/transcribe.cpp/build/bin/transcribe-cli
model=$(hf download "handy-computer/$name-gguf" "$name-Q8_0.gguf" | sed 's/^path=//')
out=tests/fixtures/$name.txt
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/blocks"

# The current blocks, one file each, without the blank lines between them.
if [ -f "$out" ]; then
  awk -v dir="$tmp/blocks" '/^## /{f = dir "/" substr($0, 4); next} /./{print > f}' "$out"
fi

for s in "$@"; do
  "$cli" -m "$model" --timestamps "$timestamps" "3rd/transcribe.cpp/samples/$s.wav" \
    > "$tmp/raw" 2> /dev/null || { echo "$s: transcribe-cli failed" >&2; exit 1; }
  awk '/^text: /{print} /^(tokens|segments): /{t=1; print; next} t && /^  \[/{print}' "$tmp/raw" \
    > "$tmp/block"
  if ! head -n 1 "$tmp/block" | grep -q '^text: ' \
    || grep -v -e '^text: ' -e '^tokens: ' -e '^segments: ' -e '^  ' "$tmp/block" > /dev/null; then
    echo "$s: unexpected output from transcribe-cli" >&2
    exit 1
  fi
  mv "$tmp/block" "$tmp/blocks/$s"
done

first=1
for s in $(ls "$tmp/blocks" | LC_ALL=C sort); do
  [ $first = 1 ] || echo
  first=0
  echo "## $s"
  cat "$tmp/blocks/$s"
done > "$tmp/out"
mv "$tmp/out" "$out"
