# /// script
# requires-python = ">=3.10"
# dependencies = ["gguf", "regex"]
# ///
"""Usage: uv run scripts/whisper-prompts.py WHISPER.gguf

Writes the token ids of each sentence in src/whisper/prompt.rs: " " + sentence, as Hugging Face's
`get_prompt_ids` tokenizes a prompt, through GPT-2's byte-level BPE with the vocabulary and merges
of a multilingual Whisper GGUF (all multilingual Whisper models share their text tokens).
"""

import re
import sys
from pathlib import Path

import regex
from gguf import GGUFReader

PATH = Path(__file__).parent.parent / "src/whisper/prompt.rs"
# GPT-2's pre-tokenizer (tokenizer.ggml.pre = "gpt2").
SPLIT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
ENTRY = re.compile(r'\(\s*("[\w-]+"),\s*"([^"]*)",\s*&\[[^\]]*\],?\s*\)')


def strings(reader, key):
    field = reader.fields[key]
    return [bytes(field.parts[i]).decode() for i in field.data]


def byte_chars():
    """GPT-2's bytes_to_unicode: printable bytes stand for themselves, the others for U+0100 on."""
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    chars, n = {}, 0
    for b in range(256):
        if b in printable:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + n)
            n += 1
    return chars


def main():
    reader = GGUFReader(sys.argv[1])
    ids = {piece: i for i, piece in enumerate(strings(reader, "tokenizer.ggml.tokens"))}
    ranks = {tuple(m.split(" ")): i for i, m in enumerate(strings(reader, "tokenizer.ggml.merges"))}
    chars = byte_chars()

    def bpe(word):
        parts = [chars[b] for b in word.encode()]
        while len(parts) > 1:
            pairs = [(ranks.get((a, b), len(ranks)), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == len(ranks):
                break
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        return [ids[p] for p in parts]

    def entry(m):
        tokens = [str(t) for word in SPLIT.findall(" " + m.group(2)) for t in bpe(word)]
        # The ids under the sentence, wrapped at 100 columns.
        lines = [""]
        for t in tokens:
            if len(lines[-1]) + len(t) + 2 > 96:
                lines.append("")
            lines[-1] += t + ", "
        ids = "".join(f"\n    {line.rstrip()}" for line in lines)
        return f'({m.group(1)}, "{m.group(2)}", &[{ids}\n  ])'

    PATH.write_text(ENTRY.sub(entry, PATH.read_text()))


main()
