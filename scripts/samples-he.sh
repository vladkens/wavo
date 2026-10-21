#!/bin/sh
# Usage: scripts/samples-he.sh (or make samples-he)
#
# The Hebrew test clips, which transcribe.cpp's samples/ lacks, into 3rd/samples/ (gitignored):
# two utterances of the Google FLEURS he_il test split (CC-BY-4.0), taken from the start of its audio
# archive at a pinned revision and stored as 16 kHz mono PCM16 like transcribe.cpp's samples.
# Needs curl and ffmpeg; does nothing when the clips are there.
set -eu
rev=70bb2e84b976b7e960aa89f1c648e09c59f894dd
url=https://huggingface.co/datasets/google/fleurs/resolve/$rev/data/he_il/audio/test.tar.gz
dir=3rd/samples
# Name and FLEURS utterance id: 11.3 s, and 29.9 s with Latin names in it.
clips="he-short:10062352599179116465 he-long:10263266185176824960"

missing=0
for c in $clips; do
  [ -f "$dir/${c%%:*}.wav" ] || missing=1
done
[ $missing = 1 ] || exit 0

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Both clips are in the first 12 MB of the 387 MB archive. Unpacking the cut-off archive fails at
# its end, after the clips: the check below catches a clip that is still missing.
curl -fsSL -r 0-11999999 -o "$tmp/head.tar.gz" "$url"
gzip -dc < "$tmp/head.tar.gz" > "$tmp/head.tar" 2> /dev/null || true
mkdir -p "$dir"
for c in $clips; do
  name=${c%%:*} id=${c#*:}
  tar -xf "$tmp/head.tar" -C "$tmp" "test/$id.wav" 2> /dev/null || true
  [ -f "$tmp/test/$id.wav" ] || { echo "$id.wav is not in the archive's head" >&2; exit 1; }
  ffmpeg -v error -y -i "$tmp/test/$id.wav" -ar 16000 -ac 1 -c:a pcm_s16le -map_metadata -1 \
    -bitexact "$dir/$name.wav"
done
