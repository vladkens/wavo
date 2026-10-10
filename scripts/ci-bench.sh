#!/bin/sh
# Usage: scripts/ci-bench.sh (or make bench-compare); needs gh
#
# Coarse speed check for CI: `wavo bench` of target/release/wavo against main's latest build from
# the `dev` pre-release, on the same runner, 4 runs of 5 warm calls per side in the order ref, PR,
# PR, ref, twice. Compares the medians of the runs' warm medians and warns, never fails, when the
# PR is over 25% slower: this protocol's A/A noise on CI runners reaches ~12%, as their GPUs
# switch speed within seconds. Skipped on a pull request that leaves src/, Cargo.toml and
# Cargo.lock alone, and while there is no dev build: both are checked before anything is built
# or downloaded.
set -eu
summary=${GITHUB_STEP_SUMMARY:-/dev/stdout}
if [ "${GITHUB_EVENT_NAME:-}" = pull_request ] \
  && git diff --quiet HEAD^1 HEAD -- src Cargo.toml Cargo.lock; then
  echo "Speed check skipped: no changes in src/, Cargo.toml or Cargo.lock." >> "$summary"
  exit 0
fi
tmp=$(mktemp -d)
asset=wavo-dev-aarch64-apple-darwin.tar.gz
if ! gh release download dev -p "$asset" -D "$tmp"; then
  echo "Speed check: no reference build, skipped." >> "$summary"
  exit 0
fi
tar xzf "$tmp/$asset" -C "$tmp" wavo
mv "$tmp/wavo" "$tmp/ref"
make build samples
make models MODELS="gigaam-v3-e2e-rnnt parakeet-tdt-0.6b-v3"
cp target/release/wavo "$tmp/pr"

median() {
  awk -v s="$1" '$1 == s {print $10}' "$tmp/runs" | sort -n \
    | awk '{v[NR] = $1} END {print (v[int((NR + 1) / 2)] + v[int(NR / 2) + 1]) / 2}'
}

: > "$tmp/rows"
: > "$tmp/warnings"
for case in gigaam-v3:ru-short parakeet-v3:jfk; do
  model=${case%:*} clip=${case#*:}
  : > "$tmp/runs"
  for side in ref pr pr ref ref pr pr ref; do
    out=$("$tmp/$side" bench "$model" "3rd/transcribe.cpp/samples/$clip.wav" -n 5)
    echo "$side $out" | tee -a "$tmp/runs"
  done
  a=$(median ref) b=$(median pr)
  change=$(awk -v a="$a" -v b="$b" 'BEGIN {printf "%+.1f", (b / a - 1) * 100}')
  echo "| $model | $clip | $a | $b | $change% |" >> "$tmp/rows"
  if awk -v c="$change" 'BEGIN {exit !(c > 25)}'; then
    echo "::warning::$model on $clip is $change% slower than main: run the benchmarks locally" \
      >> "$tmp/warnings"
  fi
done

{
  echo "Warm median of \`wavo bench\`, main's dev build against this PR; warns over 25% slower."
  echo
  echo "| model | clip | main, ms | PR, ms | change |"
  echo "| --- | --- | ---: | ---: | ---: |"
  cat "$tmp/rows"
} >> "$summary"
cat "$tmp/warnings"
