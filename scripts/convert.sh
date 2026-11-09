#!/bin/sh
# Usage: [PROFILE=json] [CONVERTER=path] scripts/convert.sh FAMILY SOURCE OUT [ARG...]
#
# Converts a checkpoint into the Q8_0 GGUF that `wavo pull` downloads: the reference's
# 3rd/transcribe.cpp/scripts/convert-FAMILY.py writes it in the source's precision, in that
# converter's uv env (scripts/envs/FAMILY), and transcribe-quantize from `make reference` turns it
# into OUT, named `<full name>-Q8_0.gguf` as `hub::ModelInfo::file` says. Prints OUT's size and
# sha256 and the converter's commit, for the model card. SOURCE is what the converter takes: a
# Hugging Face repo id or a local checkpoint; ARGs go to it as they are, such as `--revision
# <commit>` to pin the source.
#
# The converters know each model by a profile named by its slug, the full name (Parakeet takes it
# from the output's folder, GigaAM from `--repo-id`). PROFILE adds one for a fine-tune the
# converter lacks: a JSON object whose `base` names the profile it copies and whose other keys
# replace that one's. CONVERTER runs a converter from another checkout of transcribe.cpp, such as
# an unmerged pull request's (3rd/ is gitignored: check it out beside 3rd/transcribe.cpp).
#
# Pin the source to the commit the model card names. Parakeet's `--revision` pins only a
# Transformers checkpoint (`"source_format": "hf_safetensors"`); NeMo loads a `.nemo` repository
# at its latest commit, so pass a local `.nemo` downloaded at the commit instead:
#
#   PROFILE='{"base": "parakeet-tdt-0.6b-v3", "source_format": "hf_safetensors", ...}' \
#     scripts/convert.sh parakeet ihnatov-yuri/parakeet-tdt-0.6b-v3-uk \
#     out/parakeet-tdt-0.6b-v3-uk-Q8_0.gguf --revision <commit>
#
# Publish a conversion under `vladkens` (its `owner` in hub::MODELS) as the repository
# `<full name>-gguf`, with the GGUF and a README.md model card that names the source and its
# commit, its license, the converter's commit and the sha256; a CC-BY source also needs its
# LICENSE text and a NOTICE of the changes, an MIT one the license with its copyright line.
# Put them in one folder and upload it, which creates the repository:
#
#   hf upload vladkens/<full name>-gguf <folder> . --commit-message "<full name> Q8_0"
#
# Needs uv and `make reference`. The converter downloads the source into the Hugging Face cache
# (HF_HUB_CACHE moves it) and holds it in memory, a few GB for a 0.6B model.
set -eu
[ $# -ge 3 ] || { echo "usage: $0 FAMILY SOURCE OUT [ARG...]" >&2; exit 2; }
family=$1 source=$2 out=$3
shift 3
name=$(basename "$out" -Q8_0.gguf)
[ "$name-Q8_0.gguf" = "$(basename "$out")" ] || { echo "$out: not <name>-Q8_0.gguf" >&2; exit 2; }
converter=${CONVERTER:-3rd/transcribe.cpp/scripts/convert-$family.py}
scripts=$(cd "$(dirname "$converter")" && pwd)
quantize=$PWD/3rd/transcribe.cpp/build/bin/transcribe-quantize
[ -x "$quantize" ] || { echo "$quantize: run make reference" >&2; exit 1; }

mkdir -p "$(dirname "$out")"
tmp=$(mktemp -d "$(dirname "$out")/.convert.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/$name"
# The converter runs as a module, so the profile goes into its table before its main().
uv run --project "$scripts/envs/$family" python -I - "$scripts/$(basename "$converter")" "$name" \
  "${PROFILE:-}" "$source" "$tmp/$name/$name.gguf" "$@" << 'EOF'
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
EOF
"$quantize" "$tmp/$name/$name.gguf" "$tmp/out.gguf" --quant Q8_0
mv "$tmp/out.gguf" "$out"

echo "file: $out"
echo "size: $(wc -c < "$out" | tr -d ' ')"
echo "sha256: $(shasum -a 256 "$out" | cut -d ' ' -f 1)"
echo "converter: $(git -C "$scripts" rev-parse HEAD)"
