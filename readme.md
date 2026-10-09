# wavo

Speech-to-text in pure Rust. Runs the GGUF ASR models published by
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) on the GPU through `wgpu` with
custom WGSL kernels: no C/C++, ONNX or BLAS, just `cargo build`.

Works now, on Apple Silicon: the four GigaAM v3 models (Russian) and Parakeet TDT 0.6B v2
(English), with output that matches transcribe.cpp exactly (text, tokens and timestamps). More
models are next: [docs/roadmap.md](docs/roadmap.md).

| Model | Head | Output |
|---|---|---|
| `gigaam-v3-e2e-rnnt` | RNN-T | cased, punctuated (1024 SentencePiece pieces) |
| `gigaam-v3-e2e-ctc` | CTC | cased, punctuated (256 SentencePiece pieces) |
| `gigaam-v3-rnnt` | RNN-T | lowercase letters and spaces |
| `gigaam-v3-ctc` | CTC | lowercase letters and spaces |
| `parakeet-tdt-0.6b-v2` | TDT | English, cased, punctuated (1024 SentencePiece pieces) |

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

Apple M2, same Q8_0 GGUF, transcribe.cpp on Metal. Warm is the median of 10–20 calls. For
`gigaam-v3-e2e-rnnt`:

| Clip | wavo warm | transcribe.cpp warm | wavo first call | transcribe.cpp first call |
|---|---|---|---|---|
| 4.5 s | 43–44 ms | 44–45 ms | 48–49 ms | 48–73 ms |
| 11 s | 96 ms | 102–103 ms | 101 ms | 103–124 ms |
| 34 s | 297–298 ms | 312–326 ms | 305 ms | 319–353 ms |

Model load: wavo 0.08–0.11 s, transcribe.cpp 0.13–0.17 s. Peak memory: wavo 0.31–0.34 GB,
transcribe.cpp 0.34–0.35 GB. The other three models are 2–17% faster than transcribe.cpp warm
with less memory; the CTC ones take 39–42 ms on the 4.5 s clip and 268–273 ms on the 34 s one.

`parakeet-tdt-0.6b-v2` on 11 s / 35 s / 5 s of silence: wavo warm 181 / 756–763 / 94 ms,
transcribe.cpp 209–213 / 799–821 / 95–97 ms (median; its multi-threaded decoder varies, minimum
702 ms on the 35 s clip). Model load: wavo 0.16–0.19 s, transcribe.cpp 0.26–0.27 s. Peak memory:
wavo 0.84–0.92 GB, transcribe.cpp 0.85–0.87 GB. Details: [docs/perf.md](docs/perf.md).

Built by coding agents; the rules are in [agents.md](agents.md).
