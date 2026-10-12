#!/bin/sh
# Usage: scripts/fixtures.sh MODEL TIMESTAMPS SAMPLE... (or make fixtures MODEL= SAMPLES= [TIMESTAMPS=])
#
# Runs the reference CLI on 3rd/transcribe.cpp/samples/SAMPLE.wav and writes its transcript as the
# block `## SAMPLE` of tests/fixtures/MODEL.txt: the other blocks stay byte for byte, all sorted by
# name and separated by one blank line. A body line starts with `text: `, `tokens: `, `segments: `
# or two spaces, so no transcript reads as a header; with TIMESTAMPS=none (Moonshine) a block is
# the `text: ` line alone. Writes nothing if the reference fails, cuts a decode short (`run:` not
# `ok`: truncated or repeating) or prints anything else.
#
# The clips come from transcribe.cpp's samples/ and stay there; some are Google FLEURS recordings
# (CC-BY-4.0). The fixtures keep only the reference's transcripts, never audio.
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
  grep -qx 'run: ok' "$tmp/raw" || { echo "$s: $(grep '^run: ' "$tmp/raw")" >&2; exit 1; }
  # From `text:` to `realtime:` every line must be known, so the filter below drops no part of
  # the transcript, such as a second line of text.
  awk '/^text: /{r=1; next} /^  realtime: /{r=0}
    r && !/^(detected-language|words|tokens|segments): |^  \[/{print "unexpected: " $0; exit 1}' \
    "$tmp/raw" >&2 || { echo "$s: unexpected output from transcribe-cli" >&2; exit 1; }
  # The text and the one list TIMESTAMPS asks for: with `token` the reference also prints segments
  # and words for some models (Parakeet on jfk).
  awk -v list="${timestamps}s: " '/^text: /{print} /^[^ ]/{t=0} t && /^  \[/{print}
    index($0, list) == 1{t=1; print}' "$tmp/raw" > "$tmp/block"
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
