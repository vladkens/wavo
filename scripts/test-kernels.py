#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The fixtures on each kernel tier, or the ones named, until one fails."""

import os
import subprocess
import sys

CMD = ["cargo", "test", "--release", "--locked", "--test", "models", "--", "--skip", "_ane"]
TIERS = ["portable", "portable-32", "portable-cpu"]

if sys.platform == "darwin":
    TIERS.append("matrix8")

for tier in sys.argv[1:] or TIERS:
    print(f"kernels: {tier}", flush=True)
    code = subprocess.run(CMD, env={**os.environ, "WAVO_KERNELS": tier}).returncode

    if code:
        sys.exit(code)
