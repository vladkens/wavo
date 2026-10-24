#!/bin/sh
# Usage: scripts/lid-data.sh (the language ID evaluation, examples/langid-eval.rs)
#
# Test clips of 13 languages for language identification, into 3rd/lid/<code>/ (gitignored): the
# first 12 distinct sentences in the head of each Google FLEURS test archive (CC-BY-4.0) at a
# pinned revision, as 16 kHz mono PCM16, and `list.tsv` with each file and its transcription.
# Needs curl and ffmpeg; skips a language whose list is there.
set -eu
rev=70bb2e84b976b7e960aa89f1c648e09c59f894dd
base=https://huggingface.co/datasets/google/fleurs/resolve/$rev/data
langs="en:en_us ru:ru_ru uk:uk_ua de:de_de fr:fr_fr es:es_419 it:it_it pl:pl_pl cs:cs_cz sk:sk_sk he:he_il ar:ar_eg zh:cmn_hans_cn"
per=12

for l in $langs; do
  code=${l%%:*} name=${l#*:} dir=3rd/lid/${l%%:*}
  [ -f "$dir/list.tsv" ] && continue
  tmp=$(mktemp -d)
  # Some 20 MB of a 250-650 MB archive; unpacking the cut-off archive fails at its end.
  curl -fsSL -r 0-19999999 -o "$tmp/head.tar.gz" "$base/$name/audio/test.tar.gz"
  curl -fsSL -o "$tmp/test.tsv" "$base/$name/test.tsv"
  mkdir -p "$tmp/x" "$dir"
  gzip -dc < "$tmp/head.tar.gz" > "$tmp/head.tar" 2> /dev/null || true
  # In archive order, without the last file: the cut may fall inside it.
  tar -tf "$tmp/head.tar" 2> /dev/null | grep '\.wav$' | sed '$d' > "$tmp/files" || true
  tar -xf "$tmp/head.tar" -C "$tmp/x" 2> /dev/null || true
  : > "$dir/.list"
  seen=" "
  while read -r path; do
    file=$(basename "$path")
    line=$(awk -F '\t' -v f="$file" '$2 == f { print $1 "\t" $3; exit }' "$tmp/test.tsv")
    [ -n "$line" ] || continue
    id=${line%%"	"*}
    case "$seen" in *" $id "*) continue ;; esac
    ffmpeg -nostdin -v error -y -i "$tmp/x/$path" -ar 16000 -ac 1 -c:a pcm_s16le -map_metadata -1 \
      -bitexact -f wav "$dir/$file"
    seen="$seen$id "
    printf '%s\t%s\n' "$file" "${line#*"	"}" >> "$dir/.list"
    [ "$(wc -l < "$dir/.list")" -ge $per ] && break
  done < "$tmp/files"
  mv "$dir/.list" "$dir/list.tsv"
  rm -rf "$tmp"
  echo "$code: $(wc -l < "$dir/list.tsv") clips"
done
