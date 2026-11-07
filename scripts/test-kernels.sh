#!/bin/sh
# Usage: scripts/test-kernels.sh [--smoke | TIER...] [-- TEST...] (or make test-all KERNELS= TESTS=)
#
# tests/models.rs without its `_ane` tests, once per kernel tier, forced with WAVO_KERNELS, which
# unit tests must not see. Without TIER: the tiers the `kernels` row of `wavo debug` allows on the
# first adapter, leaving out its default, which the plain `cargo test` covers. The list is fixed by
# that row, not by which tiers load, so a tier that fails its probe fails here. With --smoke: only
# `portable`, if that list has it. TEST... goes to the test binary. A failing tier is named and the
# others still run; the script then fails.
set -eu
smoke=
if [ "${1:-}" = --smoke ]; then
  smoke=1
  shift
fi

tiers=
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  tiers="$tiers $1"
  shift
done

if [ $# -gt 0 ]; then
  shift
fi

if [ -z "$tiers" ]; then
  cargo build --release --locked --bin wavo
  debug=$(target/release/wavo debug)
  adapter=$(echo "$debug" | sed -n 's/^\[0\] \(.*)\).*/\1/p')
  kernels=$(echo "$debug" | sed -n '/^\[0\] /,/^\[1\] /s/^  kernels  *//p')
  echo "test-kernels: default tier on ${adapter:-no adapter}: ${kernels:-no kernels row}"

  # A CPU adapter refuses 128-wide tiles.
  portable="portable portable-32 portable-cpu"
  case $adapter in
    *", software)") portable="portable-cpu" ;;
  esac

  case $kernels in
    "8x8 matrices, Metal 4 tensor GEMM") tiers="matrix8 $portable" ;;
    "8x8 matrices: "* | "16x16 matrices") tiers=$portable ;;
    "portable, CPU tiles") tiers= ;; # the default is `portable-cpu`
    portable | "portable: "*) tiers="portable-32 portable-cpu" ;;
    *)
      echo "test-kernels: can't tell the tiers from this \`wavo debug\`:" >&2
      echo "$debug" >&2
      exit 1
      ;;
  esac

  if [ -n "$smoke" ]; then
    case " $tiers " in
      *" portable "*) tiers=portable ;;
      *) tiers= ;;
    esac
  fi

  echo "test-kernels: tiers to run: ${tiers:-none}"
fi

failed=
for tier in $tiers; do
  echo "test-kernels: running tier $tier"
  if ! WAVO_KERNELS=$tier cargo test --release --locked --test models -- --skip _ane "$@"; then
    echo "test-kernels: tier $tier failed" >&2
    failed="$failed $tier"
  fi
done

if [ -n "$failed" ]; then
  echo "test-kernels: failed tiers:$failed" >&2
  exit 1
fi
