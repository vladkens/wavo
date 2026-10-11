#!/bin/sh
# Usage: curl -fsSL https://raw.githubusercontent.com/vladkens/wavo/main/scripts/install.sh | sh
#        ... | sh -s dev    (the latest build of main instead of the latest release)
#
# Installs the prebuilt wavo binary for this macOS (Apple Silicon) or Linux machine into
# $WAVO_BIN, by default ~/.local/bin.
set -eu

repo=https://github.com/vladkens/wavo
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) target=aarch64-apple-darwin ;;
  Linux/x86_64) target=x86_64-unknown-linux-gnu ;;
  Linux/aarch64 | Linux/arm64) target=aarch64-unknown-linux-gnu ;;
  *) echo "no prebuilt wavo for $(uname -sm); build it with: cargo install wavo" >&2; exit 1 ;;
esac

# The latest release's tag, from where /releases/latest redirects.
tag=${1:-$(curl -fsSI "$repo/releases/latest" | sed -n 's|^[Ll]ocation: .*/tag/\([^[:space:]]*\).*|\1|p')}
[ -n "$tag" ] || { echo "can't find the latest wavo release" >&2; exit 1; }

dir=${WAVO_BIN:-$HOME/.local/bin}
mkdir -p "$dir"
curl -fL --progress-bar "$repo/releases/download/$tag/wavo-$tag-$target.tar.gz" | tar xz -C "$dir" wavo
echo "installed $("$dir/wavo" --version) to $dir/wavo"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add $dir to your PATH, e.g.: export PATH=\"$dir:\$PATH\"" >&2 ;;
esac
