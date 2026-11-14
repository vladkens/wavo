#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The GPU fixtures on each kernel tier, or the ones named, until one fails."""

import os
import platform
import subprocess
import sys

CMD = ["cargo", "test", "--release", "--locked", "--test", "models", "--", "--skip", "_ane"]
# The tiers are GPU kernels; on x86_64 `portable` also swaps the CPU's VNNI for its f32 tile.
CPU = platform.machine().lower() in ("x86_64", "amd64")
TIERS = ["portable", "portable-32", "portable-cpu"]

if sys.platform == "darwin":
    TIERS.append("matrix8")

for tier in sys.argv[1:] or TIERS:
    print(f"kernels: {tier}", flush=True)
    cmd = CMD if CPU and tier == "portable" else [*CMD, "--skip", "_cpu"]
    code = subprocess.run(cmd, env={**os.environ, "WAVO_KERNELS": tier}).returncode

    if code:
        sys.exit(code)
