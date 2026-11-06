#!/bin/sh
# Usage: scripts/render-docs.sh [FILE.md ...] (readme.md and docs/*.md by default)
#
# Each Markdown file rendered by GitHub's own API and shot by a headless browser into .cache/render/,
# to look at the tables before pushing. Needs gh and a Chromium browser: BROWSER, or Helium.
set -eu
browser=${BROWSER:-/Applications/Helium.app/Contents/MacOS/Helium}
css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.8.1/github-markdown-light.min.css
out=.cache/render
mkdir -p "$out"
[ $# -gt 0 ] || set -- readme.md docs/models.md docs/benchmarks.md

for file in "$@"; do
  name=$(basename "$file" .md)
  {
    echo "<meta charset=utf-8><link rel=stylesheet href=$css>"
    echo '<style>body{max-width:900px;margin:30px auto;padding:0 20px}</style><body class=markdown-body>'
    gh api markdown/raw --input "$file" -H 'Content-Type: text/plain'
  } > "$out/$name.html"
  "$browser" --headless=new --user-data-dir="$out/profile" --disable-gpu --window-size=1000,3000 \
    --virtual-time-budget=3000 --screenshot="$PWD/$out/$name.png" "file://$PWD/$out/$name.html" 2> /dev/null
  echo "$out/$name.png"
done
