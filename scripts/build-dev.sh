#!/bin/sh
# Usage: scripts/build-dev.sh (on an Apple Silicon Mac with cargo-cross installed)
#
# Release builds for macOS, Linux (glibc 2.28 and up) and Windows (MinGW), packed with the readme
# and license as target/dist/wavo-dev-<target>.tar.gz (.zip for Windows). cargo-cross adds the Rust
# targets and downloads GCC toolchains into CROSS_COMPILER_DIR. Binaries keep their debuginfo for
# backtraces; on macOS it sits in wavo.dSYM, which has to stay next to the binary.
set -eu
cd "$(dirname "$0")/.."
target=${CARGO_TARGET_DIR:-target}
dist=$target/dist

CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed cargo build --release --locked --target aarch64-apple-darwin
cargo cross build --release --locked --glibc-version 2.28 \
  --targets x86_64-unknown-linux-gnu,aarch64-unknown-linux-gnu,x86_64-pc-windows-gnu

rm -rf "$dist"
mkdir -p "$dist"
tar czhf "$dist/wavo-dev-aarch64-apple-darwin.tar.gz" readme.md LICENSE \
  -C "$target/aarch64-apple-darwin/release" wavo wavo.dSYM
for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
  tar czf "$dist/wavo-dev-$t.tar.gz" readme.md LICENSE -C "$target/$t/release" wavo
done
zip -jq "$dist/wavo-dev-x86_64-pc-windows-gnu.zip" readme.md LICENSE \
  "$target/x86_64-pc-windows-gnu/release/wavo.exe"
ls -lh "$dist"
