# wavo

Speech-to-text in pure Rust. Runs the GGUF ASR models published by
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) on the GPU through `wgpu` with
custom WGSL kernels: no C/C++, ONNX or BLAS, just `cargo build`.

Works now, on Apple Silicon: the four GigaAM v3 models (Russian) and Parakeet TDT 0.6B v2
(English) and v3 (25 European languages), with output that matches transcribe.cpp exactly (text,
tokens and timestamps). More models are next: [docs/roadmap.md](docs/roadmap.md).

| Model | Head | Output |
|---|---|---|
| `gigaam-v3-e2e-rnnt` | RNN-T | cased, punctuated (1024 SentencePiece pieces) |
| `gigaam-v3-e2e-ctc` | CTC | cased, punctuated (256 SentencePiece pieces) |
| `gigaam-v3-rnnt` | RNN-T | lowercase letters and spaces |
| `gigaam-v3-ctc` | CTC | lowercase letters and spaces |
| `parakeet-tdt-0.6b-v2` | TDT | English, cased, punctuated (1024 SentencePiece pieces) |
| `parakeet-tdt-0.6b-v3` | TDT | 25 European languages, cased, punctuated (8192 SentencePiece pieces) |

## Usage

```sh
hf download handy-computer/gigaam-v3-e2e-rnnt-gguf gigaam-v3-e2e-rnnt-Q8_0.gguf  # prints the path
cargo build --release
target/release/wavo MODEL.gguf AUDIO.wav [--tokens]
target/release/wavo bench MODEL.gguf AUDIO.wav [-n 10]
```

Audio must be mono 16 kHz WAV (16-bit PCM or 32-bit float). `--tokens` also prints one line per
token: start time in ms, token id, piece. `bench` prints load time, the first call and the warm
median and minimum over N calls.

As a library: `wavo::Model::load(path)?.transcribe(&pcm)?` returns a `Transcript` with `text`
and `tokens` (id, piece, `start_ms`); `pcm` is 16 kHz mono `f32`.

## Speed

Apple M2 (8 CPU cores, 10-core GPU), the same Q8_0 GGUF in both, transcribe.cpp on Metal. Each
cell is wavo / transcribe.cpp. Warm is the median of 20 / 10 calls after the first; first call and
model load are from a fresh process; memory is the peak footprint. Measured back to back on
2026-10-10; details and ranges in [docs/perf.md](docs/perf.md).

| Model | Audio | Warm, ms | First call, ms | Load, ms | Peak memory, MiB |
|---|---|---|---|---|---|
| `gigaam-v3-e2e-rnnt` | 4.5 s | 43.8 / 44.6 | 46.8 / 47.6 | 77 / 123 | 294 / 323 |
|  | 11 s | 96.1 / 100 | 100 / 102 | 77 / 124 | 302 / 325 |
|  | 34 s | 299 / 310 | 302 / 323 | 77 / 126 | 322 / 334 |
| `gigaam-v3-e2e-ctc` | 4.5 s | 40.1 / 41.8 | 43.4 / 53.1 | 76 / 129 | 289 / 314 |
|  | 11 s | 85.5 / 92.7 | 88.8 / 95.6 | 75 / 126 | 295 / 315 |
|  | 34 s | 270 / 296 | 276 / 308 | 75 / 126 | 315 / 323 |
| `gigaam-v3-rnnt` | 4.5 s | 42.6 / 43.0 | 45.9 / 50.5 | 75 / 125 | 291 / 319 |
|  | 11 s | 96.1 / 97.4 | 99.2 / 99.6 | 76 / 122 | 299 / 321 |
|  | 34 s | 296 / 306 | 300 / 315 | 76 / 124 | 320 / 330 |
| `gigaam-v3-ctc` | 4.5 s | 39.8 / 41.3 | 43.1 / 52.3 | 74 / 123 | 288 / 313 |
|  | 11 s | 85.0 / 91.5 | 89.5 / 94.0 | 75 / 129 | 293 / 314 |
|  | 34 s | 269 / 291 | 275 / 295 | 77 / 125 | 314 / 322 |
| `parakeet-tdt-0.6b-v2` | 11 s | 143 / 199 | 149 / 232 | 159 / 274 | 740 / 820 |
|  | 35 s | 461 / 699 | 476 / 767 | 167 / 275 | 767 / 833 |
|  | 5 s, silent | 83.4 / 97.1 | 87.7 / 116 | 175 / 280 | 735 / 816 |
| `parakeet-tdt-0.6b-v3` | 11 s, English | 154 / 215 | 159 / 281 | 169 / 284 | 763 / 884 |
|  | 11 s, Russian | 161 / 246 | 173 / 349 | 167 / 285 | 762 / 884 |
|  | 11 s, Ukrainian | 167 / 254 | 170 / 304 | 188 / 317 | 762 / 884 |

Built by coding agents; the rules are in [agents.md](agents.md).
