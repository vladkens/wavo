# Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
# /// script
# requires-python = ">=3.11"
# ///
"""Lines per module: code, docs, comments and inline tests, apart.

A module is a directory under src/ or a top-level file; the GPU kernels (src/gpu/*.wgsl and
*.metal) are one row apart from src/gpu/mod.rs, and so are the CLI's debug commands
(cli/debug.rs) and its server (cli/serve.rs and cli/http.rs); examples/ and tests/ are one row
each. Code is every line that is not blank, a comment, a doc comment or inside a `#[cfg(test)]`
item; a known limit: a multi-line raw string with unbalanced braces in such an item throws off
where it ends. With a git ref, a column shows how code moved since it.
`--map` lists the files with the first sentence of each one's module doc (`//!`, or the comment
after the header in a kernel): what it does.
`--check` sums the code per part against its budget (BUDGETS) and fails when one is over, when
a budget is not a whole number of hundreds, or when a directory under src/ has no budget; with
`SIZE_WARN` set (CI on `main`, where two PRs that each fit can merge into one over) it only warns.

uv run scripts/size.py [--files | --map | --check] [--vs <ref>]
"""

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ("src", "examples", "tests")
EXTS = (".rs", ".wgsl", ".metal")
KINDS = ("code", "docs", "comments", "tests", "blank")
# The code lines each part may have: the core (src/'s top-level files and src/gpu/mod.rs: the
# public API, errors, the GGUF reader, the GPU runtime, the CPU paths, the FFT, the shared conformer and transformer
# blocks, the tokenizer, the splitting of long audio), the GPU kernels (WGSL and Metal), each
# model family (src/<family>/), the Neural Engine backend (src/ane/), the hub (src/hub/), and the
# CLI in three: its core (src/cli/ without the two others), its debug commands (cli/debug.rs) and
# its server (cli/serve.rs and cli/http.rs). A budget is whole hundreds (STEP): a change that
# doesn't fit is written tighter, or the budget goes up by a hundred, a decision recorded in its
# PR; never by the few lines one change needs.
STEP = 100
# Directories under src/ whose code counts in the core.
CORE = ("gpu",)
BUDGETS = {
  "core": 2800,
  "kernels": 2400,
  "gigaam": 500,
  "parakeet": 600,
  "whisper": 600,
  "moonshine": 300,
  "ane": 1100,
  "hub": 600,
  "cli": 800,
  "cli/debug": 400,
  "cli/serve": 600,
}


def module(path: str, files: bool) -> str:
  """The module a file's lines count in.

  >>> [module(p, False) for p in ("src/gguf.rs", "src/gpu/mod.rs", "src/gpu/ops.wgsl", "src/gpu/gemm_tensor.metal")]
  ['gguf', 'gpu', 'kernels', 'kernels']
  >>> [module(p, False) for p in ("src/whisper/mod.rs", "src/cli/main.rs", "src/cli/http.rs", "src/cli/debug.rs", "tests/models.rs")]
  ['whisper', 'cli', 'cli/serve', 'cli/debug', 'tests']
  """
  if files:
    return path

  parts = path.split("/")
  if parts[0] != "src":
    return parts[0]

  if path.endswith((".wgsl", ".metal")):
    return "kernels"

  if len(parts) == 2:
    return parts[1].removesuffix(".rs")

  if parts[1] == "cli" and parts[2] in ("serve.rs", "http.rs"):
    return "cli/serve"

  if parts[1] == "cli" and parts[2] == "debug.rs":
    return "cli/debug"

  return parts[1]


def part(path: str) -> str | None:
  """The budget a file's code counts against: its module's, or the core's for src/'s top-level
  files and the GPU runtime; examples and tests have none, another directory is `src/<dir>`.

  >>> [part(p) for p in ("src/gguf.rs", "src/gpu/mod.rs", "src/gpu/ops.wgsl", "src/parakeet/mod.rs", "src/cli/http.rs")]
  ['core', 'core', 'kernels', 'parakeet', 'cli/serve']
  >>> [part(p) for p in ("src/hub/cache.rs", "examples/batch.rs", "src/vad/mod.rs")]
  ['hub', None, 'src/vad']
  """
  parts = path.split("/")
  if parts[0] != "src":
    return None

  if (m := module(path, False)) in BUDGETS:
    return m

  return "core" if len(parts) == 2 or parts[1] in CORE else f"src/{parts[1]}"


def check(files: dict[str, str]) -> list[str]:
  """Prints each part's code against its budget; what is wrong, if anything."""
  used = defaultdict(int)
  for path, text in files.items():
    if (p := part(path)) is not None:
      used[p] += count(text)["code"]

  problems = [f"no budget for {p}" for p in used if p not in BUDGETS]
  problems += [f"{p}: budget {limit} is not whole hundreds: raise it by {STEP}, not by a few lines" for p, limit in BUDGETS.items() if limit % STEP]
  for line in problems:
    print(line)

  for p, limit in BUDGETS.items():
    over = used[p] > limit
    print(f"{p:<10} {used[p]:>6} / {limit:<6} {'over' if over else 'ok'}")
    if over:
      problems.append(f"{p} is over its code budget")

  return problems


def braces(line: str, pair: str = "{}") -> int:
  """Opening minus closing brackets of a line, outside char literals, strings and its comment.

  >>> [braces(x) for x in ('"http://x"; if a {', "match c { '{' => 1", "'\\\\u{7b}'", 'f("}") // {')]
  [1, 1, 0, 0]
  """
  line = re.sub(r"'(\\[^']*|[^'\\])'", "", line)
  line = re.sub(r'"(\\.|[^"\\])*"', "", line)
  line = line.split("//")[0]
  return line.count(pair[0]) - line.count(pair[1])


def count(text: str) -> dict[str, int]:
  """Lines of each kind. A `#[cfg(test)]` item runs to its closing brace or semicolon at its own
  depth, to a comma that ends its first line (a field, variant or arm), or to just before a line
  that closes the item around it, which stays code.

  >>> count("//! Doc.\\nfn a() {}\\n\\n#[cfg(test)]\\nmod tests {\\n  // x\\n}\\nfn b() {}\\n")
  {'code': 2, 'docs': 1, 'comments': 0, 'tests': 4, 'blank': 1}
  >>> count("struct S {\\n  #[cfg(test)]\\n  x: u32,\\n  y: u32,\\n}\\n")
  {'code': 3, 'docs': 0, 'comments': 0, 'tests': 2, 'blank': 0}
  >>> count("struct S {\\n  #[cfg(test)]\\n  x: u32\\n}\\n")
  {'code': 2, 'docs': 0, 'comments': 0, 'tests': 2, 'blank': 0}
  """
  n = dict.fromkeys(KINDS, 0)
  test, depth, item = False, 0, False

  for raw in text.splitlines():
    line = raw.strip()
    if not test and line.startswith("#[cfg(test)]"):
      test, depth, item = True, 0, False

    if test:
      depth += braces(line)
      if depth < 0:
        test = False
      else:
        n["tests"] += 1
        first = not item and not (line.startswith("#[") and line.endswith("]"))
        item |= first
        if depth == 0 and ("}" in line or line.endswith(";") or first and line.endswith(",")):
          test = False

        continue

    if not line:
      n["blank"] += 1
    elif line.startswith(("///", "//!")):
      n["docs"] += 1
    elif line.startswith("//"):
      n["comments"] += 1
    else:
      n["code"] += 1

  return n


def summary(path: str, text: str) -> str:
  """The first sentence of a file's leading `//!` doc, or of a kernel's comment below its header.

  >>> summary("src/gpu/ops.wgsl", "// Copyright (c) vladkens\\n// Row-wise kernels. More.\\nfn a() {}\\n")
  'Row-wise kernels'
  >>> summary("src/lib.rs", "// Copyright\\n#![allow(x)]\\n#![cfg_attr(\\n  doc = \\"[a]: b\\"\\n)]\\n//! Speech. More.\\n")
  'Speech'
  """
  kernel = not path.endswith(".rs")
  doc, attr = [], 0
  for raw in text.splitlines()[1:] if kernel else text.splitlines():
    line = raw.strip()
    if attr or not doc and line.startswith("#!["):
      attr += braces(line, "[]")
      continue

    if line.startswith("//!") or kernel and line.startswith("//"):
      doc.append(line.lstrip("/!").strip())
    elif doc or (line and not line.startswith("//")):
      break

  first = re.split(r"(?<=[.])\s", " ".join(doc).strip(), maxsplit=1)[0]
  return first.rstrip(".") if len(first) <= 110 else first[:109].rstrip() + "…"


def files_now() -> dict[str, str]:
  out = {}
  for d in DIRS:
    for p in sorted((ROOT / d).rglob("*")):
      if p.suffix in EXTS:
        out[str(p.relative_to(ROOT))] = p.read_text()

  return out


def files_at(ref: str) -> dict[str, str]:
  git = ["git", "-C", str(ROOT)]
  names = subprocess.run([*git, "ls-tree", "-r", "--name-only", ref, *DIRS], capture_output=True, text=True, check=True).stdout
  out = {}
  for name in names.split():
    if name.endswith(EXTS):
      out[name] = subprocess.run([*git, "show", f"{ref}:{name}"], capture_output=True, text=True, check=True).stdout

  return out


def table(files: dict[str, str], by_file: bool) -> dict[str, dict[str, int]]:
  rows = defaultdict(lambda: dict.fromkeys(KINDS, 0))
  for path, text in files.items():
    row = rows[module(path, by_file)]
    for k, v in count(text).items():
      row[k] += v

  return rows


def main():
  ap = argparse.ArgumentParser(description="Lines per module.")
  ap.add_argument("--files", action="store_true", help="one row per file")
  ap.add_argument("--map", action="store_true", help="one row per file with what it does")
  ap.add_argument("--vs", metavar="REF", help="show the code change since this git ref")
  ap.add_argument("--check", action="store_true", help="fail when a part is over its budget")
  args = ap.parse_args()

  files = files_now()
  if args.check:
    problems = check(files)
    if problems and os.environ.get("SIZE_WARN"):
      print(f"::warning::{'; '.join(problems)}; fix it in the next PR")

    raise SystemExit(0 if not problems or os.environ.get("SIZE_WARN") else 1)

  if args.map:
    for path in sorted(files, key=lambda p: (p.count("/"), p)):
      print(f"{path:<30} {count(files[path])['code']:>5}  {summary(path, files[path])}")

    return

  rows = table(files, args.files)
  try:
    base = table(files_at(args.vs), args.files) if args.vs else None
  except subprocess.CalledProcessError:
    ap.error(f"unknown ref {args.vs}")

  names = sorted(set(rows) | set(base or {}), key=lambda m: -rows[m]["code"] if m in rows else 0)
  total = {k: sum(r[k] for r in rows.values()) for k in KINDS}

  w = max(len("module"), *(len(m) for m in names))
  head = f"{'module':<{w}} {'code':>7} {'%':>4} {'docs':>6} {'comm':>6} {'tests':>6}"
  print(head + (f" {'vs ' + args.vs:>10}" if base else ""))
  print("-" * (len(head) + (11 if base else 0)))

  for m in [*names, None]:
    r = total if m is None else rows.get(m, dict.fromkeys(KINDS, 0))
    if m is None:
      print("-" * (len(head) + (11 if base else 0)))

    pct = 100 * r["code"] / max(total["code"], 1)
    line = f"{m or 'total':<{w}} {r['code']:>7} {pct:>4.0f} {r['docs']:>6} {r['comments']:>6} {r['tests']:>6}"
    if base is not None:
      was = sum(b["code"] for b in base.values()) if m is None else base.get(m, {"code": 0})["code"]
      line += f" {r['code'] - was:>+10}"

    print(line)


if __name__ == "__main__":
  main()
