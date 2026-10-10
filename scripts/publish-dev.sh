#!/bin/sh
# Usage: scripts/publish-dev.sh (in CI on main, after scripts/build-dev.sh; needs GH_TOKEN)
#
# Moves the `dev` tag to this commit and uploads the archives to the "Dev build" pre-release.
# Skips when main has moved on, so a slower older run never replaces a newer build.
set -eu
cd "$(dirname "$0")/.."
dist=${CARGO_TARGET_DIR:-target}/dist
sha=$(git rev-parse HEAD)
if [ "$(git ls-remote origin refs/heads/main | cut -f1)" != "$sha" ]; then
  echo "main has moved past $sha, skipping"
  exit 0
fi
notes="Latest build of \`main\` at $sha ($(date -u '+%Y-%m-%d %H:%M UTC'))."
git tag -f dev
git push -f origin refs/tags/dev
if gh release view dev > /dev/null 2>&1; then
  gh release upload dev "$dist"/* --clobber
  gh release edit dev --title "Dev build" --notes "$notes"
else
  gh release create dev "$dist"/* --prerelease --title "Dev build" --notes "$notes"
fi
