#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.3"]
# ///
"""Convert a checkpoint into the Q8_0 GGUF that `wavo pull` downloads.

    [PROFILE=json] [CONVERTER=path] scripts/convert.py FAMILY SOURCE OUT [ARG...]

Runs transcribe.cpp's scripts/convert-FAMILY.py from `make reference` (or CONVERTER, another
checkout's) on SOURCE, a Hugging Face repo id or a local checkpoint, with the ARGs, such as
`--revision <commit>`; PROFILE adds a fine-tune the converter lacks, as JSON whose `base` names the
profile it copies. `--revision` pins a Transformers checkpoint, not a NeMo one: pass a local `.nemo`
downloaded at the commit. Quantizes the result to OUT, `<full name>-Q8_0.gguf`, as
transcribe-quantize does, and prints its size and sha256 and the converter's commit for the model
card. Publish it with the card and the source's license:

    hf upload vladkens/<full name>-gguf <folder> . --commit-message "<full name> Q8_0"
"""

import argparse
import hashlib
import mmap
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
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

F32, F16, Q8_0 = 0, 1, 8
# GGUF value types and the size of the fixed ones; 8 is a string and 9 an array.
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# transcribe-quantize's tools/transcribe-quantize/policy.cpp at Q8_0: what stays F32.
F32_NAMES = {
    "enc.pos_emb.weight", "dec.pos_emb.weight", "frontend.mel_filterbank", "frontend.window",
    "frontend.cmvn.shift", "frontend.cmvn.scale", "enc.embed.weight", "enc.embedder.comp.log_k",
    "dec.time_embed.inv_freq", "prj.query", "prj.window_positions", "dec.norm.weight",
}  # fmt: skip
F32_ENDS = (
    ".bias", ".running_mean", ".running_var", "_norm.weight", ".final_norm.weight",
    ".embed.norm.weight", ".q_norm.weight", ".k_norm.weight", ".output_norm.weight",
    ".ln_post.weight", ".ln_pre.weight", ".norm1.weight", ".norm2.weight", ".norm3.weight",
    ".pos_bias_u", ".pos_bias_v", ".pos_enc", ".after_norm.weight", ".tp_norm.weight",
    ".conv_depthwise.weight", ".attn.fsmn.weight",
)  # fmt: skip
POINTWISE = tuple(f".conv{s}pointwise{n}.weight" for s in "._" for n in "12")
# Its tokenizer tables, moved last in this order (scripts/lib/gguf_common.py's BULK_KV_KEYS).
BULK = [
    "tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges", "tokenizer.chat_template",
]  # fmt: skip


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

    default = REPO / "3rd" / "transcribe.cpp" / "scripts" / f"convert-{args.family}.py"
    converter = Path(os.environ.get("CONVERTER") or default).resolve()
    if not converter.exists():
        sys.exit(f"{converter}: run make reference")

    profile = os.environ.get("PROFILE", "")
    env = converter.parent / "envs" / args.family
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".convert.", dir=args.out.parent) as tmp:
        full = Path(tmp) / name / f"{name}.gguf"
        full.parent.mkdir()
        run = ["uv", "run", "--project", env, "python", "-I", "-c", RUN, converter, name, profile]
        subprocess.run([*run, args.source, full, *rest], check=True)
        quantized = Path(tmp) / "out.gguf"
        quantize(full, quantized)
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


def quantize(src: Path, dst: Path):
    """The F32/F16 GGUF `src` as `transcribe-quantize src dst --quant Q8_0` writes it, byte for
    byte: the same metadata with `general.file_type` 7 and the tokenizer tables last, the tensors in
    their order, 32-byte aligned (tools/transcribe-quantize/main.cpp, ggml/src/gguf.cpp)."""
    with open(src, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
        if buf[:4] != b"GGUF" or version != 3:
            sys.exit(f"{src}: not a GGUF v3 file")

        pos, kvs = 24, {}
        for _ in range(n_kv):
            key, pos = string(buf, pos)
            start, pos = pos, skip(buf, pos + 4, struct.unpack_from("<I", buf, pos)[0])
            kvs[key] = buf[start:pos]

        kvs.pop("general.file_type", None)
        kvs["general.file_type"] = struct.pack("<II", 4, 7)
        for key in BULK:
            if key in kvs:
                kvs[key] = kvs.pop(key)

        tensors = []
        for _ in range(n_tensors):
            name, pos = string(buf, pos)
            (n_dims,) = struct.unpack_from("<I", buf, pos)
            ne = struct.unpack_from(f"<{n_dims}q", buf, pos + 4)
            kind, offset = struct.unpack_from("<IQ", buf, pos + 4 + 8 * n_dims)
            pos += 16 + 8 * n_dims
            if kind not in (F32, F16):
                sys.exit(f"{src}: {name} is type {kind}, not F32 or F16")

            tensors.append((name, ne, kind, offset))

        data = pos + -pos % 32
        header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, n_tensors, len(kvs)))
        for key, value in kvs.items():
            header += pack_string(key) + value

        plan, offset = [], 0
        for name, ne, kind, at in tensors:
            n = int(np.prod(ne))
            to = target(name, ne[0])
            ne = ne[: max((i + 1 for i, x in enumerate(ne) if x > 1), default=1)]  # ggml_n_dims
            header += pack_string(name) + struct.pack(f"<I{len(ne)}qIQ", len(ne), *ne, to, offset)
            size = {F32: 4 * n, F16: 2 * n, Q8_0: n // 32 * 34}[to]
            plan.append((data + at, n, kind, to))
            offset += size + -size % 32

        with open(dst, "wb") as out:
            out.write(header + bytes(-len(header) % 32))
            for at, n, kind, to in plan:
                width = 4 if kind == F32 else 2
                x = np.frombuffer(buf[at : at + width * n], f"<f{width}")  # a copy: mmap closes
                if kind == to:
                    block = x.tobytes()
                elif to == Q8_0:
                    block = q8_0(x.astype(np.float32))
                else:
                    block = x.astype("<f4" if to == F32 else "<f2").tobytes()

                out.write(block + bytes(-len(block) % 32))


def target(name: str, ne0: int) -> int:
    """transcribe-quantize's type for a tensor at Q8_0 (policy.cpp's classify_tensor and
    resolve_target_type): norms, biases, convolutions and frontend tables stay F32, a [1, in, out]
    pointwise convolution F16, the rest is Q8_0 where a row splits into blocks of 32, else F16."""
    weight = name.endswith(".weight")
    norm = weight and ("norm_" in name or "_norms." in name)
    if name in F32_NAMES or name.endswith(F32_ENDS) or ".bn." in name or "conv_bn." in name or norm:
        return F32

    pointwise = name.endswith(POINTWISE) and "enc.blocks." in name
    if pointwise and ne0 == 1:
        return F16

    if not pointwise and weight and (name.startswith("vad.") or ".conv." in name):
        return F32

    return Q8_0 if ne0 % 32 == 0 else F16


def q8_0(x: np.ndarray) -> bytes:
    """ggml/src/ggml-quants.c's quantize_row_q8_0_ref: for each 32 values an F16 scale d, max |x| /
    127, and the 32 values over d as int8, rounded half away from zero, in F32 as there."""
    x = x.reshape(-1, 32)
    d = np.abs(x).max(axis=1) / np.float32(127)
    inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0)
    q = (x * inv[:, None]).astype(np.float64)  # exact, so the + 0.5 below is too
    blocks = np.empty(len(x), dtype=[("d", "<f2"), ("qs", "i1", 32)])
    blocks["d"] = d
    blocks["qs"] = np.copysign(np.floor(np.abs(q) + 0.5), q)
    return blocks.tobytes()


def string(buf, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, pos)
    return buf[pos + 8 : pos + 8 + n].decode(), pos + 8 + n


def pack_string(text: str) -> bytes:
    return struct.pack("<Q", len(text.encode())) + text.encode()


def skip(buf, pos: int, kind: int) -> int:
    """Past a GGUF value of this type at `pos`."""
    if kind == 8:
        return pos + 8 + struct.unpack_from("<Q", buf, pos)[0]

    if kind != 9:
        return pos + SIZES[kind]

    item, n = struct.unpack_from("<IQ", buf, pos)
    pos += 12
    if item in SIZES:
        return pos + n * SIZES[item]

    for _ in range(n):
        pos = skip(buf, pos, item)

    return pos


if __name__ == "__main__":
    main()
