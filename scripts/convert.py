#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Convert a checkpoint into the Q8_0 GGUF that `wavo pull` downloads.

    [PROFILE=json] [CONVERTER=path] scripts/convert.py FAMILY SOURCE OUT [ARG...]

The reference's 3rd/transcribe.cpp/scripts/convert-FAMILY.py writes the GGUF in the source's
precision, in that converter's uv env (scripts/envs/FAMILY), and transcribe-quantize from `make
reference` turns it into OUT, named `<full name>-Q8_0.gguf` as `hub::ModelInfo::file` says. Prints
OUT's size and sha256 and the converter's commit, for the model card. SOURCE is what the converter
takes: a Hugging Face repo id or a local checkpoint; ARGs go to it as they are, such as `--revision
<commit>` to pin the source.

The converters know each model by a profile named by its slug, the full name (Parakeet takes it from
the output's folder, GigaAM from `--repo-id`). PROFILE adds one for a fine-tune the converter lacks:
a JSON object whose `base` names the profile it copies and whose other keys replace that one's.
CONVERTER runs a converter from another checkout of transcribe.cpp, such as an unmerged pull
request's (3rd/ is gitignored: check it out beside 3rd/transcribe.cpp).

Pin the source to the commit the model card names. GigaAM's converter takes the weights from its
own table by `--repo-id` (scripts/lib/gigaam_hf.py, the `gigaam` package's downloads): SOURCE only
names them, so name the commit that table pins. Parakeet's `--revision` pins only a Transformers
checkpoint (`"source_format": "hf_safetensors"`); NeMo loads a `.nemo` repository at its latest
commit, so pass a local `.nemo` downloaded at the commit instead:

    PROFILE='{"base": "parakeet-tdt-0.6b-v3", "source_format": "hf_safetensors", ...}' \\
      scripts/convert.py parakeet ihnatov-yuri/parakeet-tdt-0.6b-v3-uk \\
      out/parakeet-tdt-0.6b-v3-uk-Q8_0.gguf --revision <commit>

Publish a conversion under `vladkens` (its `owner` in hub::MODELS) as the repository `<full
name>-gguf`, with the GGUF and a README.md model card that names the source and its commit, its
license, the converter's commit and the sha256; a CC-BY source also needs its LICENSE text and a
NOTICE of the changes, an MIT one the license with its copyright line. Put them in one folder and
upload it, which creates the repository:

    hf upload vladkens/<full name>-gguf <folder> . --commit-message "<full name> Q8_0"

Needs uv and `make reference`. The converter downloads the source into the Hugging Face cache
(HF_HUB_CACHE moves it) and holds it in memory, a few GB for a 0.6B model.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUANTIZE = REPO / "3rd" / "transcribe.cpp" / "build" / "bin" / "transcribe-quantize"
# Runs in the converter's env: the converter as a module, the profile in its table before main().
RUN = """
import importlib.util, json, sys

path, name, profile, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
spec = importlib.util.spec_from_file_location("converter", path)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
if profile:
    profile = json.loads(profile)
    base = converter.VARIANT_PROFILES[profile.pop("base")]
    converter.VARIANT_PROFILES[name] = {**base, **profile}
sys.exit(converter.main([path, *args]))
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("family", help="parakeet, gigaam, whisper, ...")
    parser.add_argument("source", help="a Hugging Face repo id or a local checkpoint")
    parser.add_argument("out", type=Path, help="the GGUF to write, <full name>-Q8_0.gguf")
    args, rest = parser.parse_known_args()
    name = args.out.name.removesuffix("-Q8_0.gguf")
    if f"{name}-Q8_0.gguf" != args.out.name:
        sys.exit(f"{args.out}: not <full name>-Q8_0.gguf")

    if not QUANTIZE.exists():
        sys.exit(f"{QUANTIZE}: run make reference")

    default = REPO / "3rd" / "transcribe.cpp" / "scripts" / f"convert-{args.family}.py"
    converter = Path(os.environ.get("CONVERTER") or default).resolve()
    profile = os.environ.get("PROFILE", "")
    env = converter.parent / "envs" / args.family
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".convert.", dir=args.out.parent) as tmp:
        full = Path(tmp) / name / f"{name}.gguf"
        full.parent.mkdir()
        run = ["uv", "run", "--project", env, "python", "-I", "-c", RUN, converter, name, profile]
        subprocess.run([*run, args.source, full, *rest], check=True)
        quantized = Path(tmp) / "out.gguf"
        subprocess.run([QUANTIZE, full, quantized, "--quant", "Q8_0"], check=True)
        quantized.replace(args.out)

    sha = hashlib.sha256()
    with open(args.out, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)

    git = ["git", "-C", converter.parent, "rev-parse", "HEAD"]
    commit = subprocess.run(git, capture_output=True, text=True, check=True).stdout.strip()
    print(f"file: {args.out}")
    print(f"size: {args.out.stat().st_size}")
    print(f"sha256: {sha.hexdigest()}")
    print(f"converter: {commit}")


if __name__ == "__main__":
    main()
