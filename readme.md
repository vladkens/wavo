# wavo

Speech-to-text in pure Rust. Runs the GGUF ASR models published by
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) on the GPU through `wgpu` with
custom WGSL kernels: no C/C++, ONNX or BLAS, just `cargo build`.

Works now: GigaAM v3 e2e-rnnt (Russian) on Apple Silicon. Its output matches transcribe.cpp
exactly (text, tokens and timestamps). More models are next: [docs/roadmap.md](docs/roadmap.md).

## Usage

```sh
hf download handy-computer/gigaam-v3-e2e-rnnt-gguf gigaam-v3-e2e-rnnt-Q8_0.gguf  # prints the path
cargo build --release
target/release/wavo MODEL.gguf AUDIO.wav [--tokens]
target/release/wavo bench MODEL.gguf AUDIO.wav [-n 10]
```

Audio must be mono 16 kHz WAV (16-bit PCM or 32-bit float). `--tokens` also prints one line per
token: encoder frame (40 ms each), token id, piece. `bench` prints load time, the first call and
the warm median and minimum over N calls.

As a library: `wavo::Model::load(path)?.transcribe(&pcm)?` returns a `Transcript` with `text`
and `tokens`; `pcm` is 16 kHz mono `f32`.

## Speed

Apple M2, same Q8_0 GGUF, transcribe.cpp on Metal. Warm is the median of 10–20 calls.

| Clip | wavo warm | transcribe.cpp warm | wavo first call | transcribe.cpp first call |
|---|---|---|---|---|
| 4.5 s | 43–44 ms | 44–45 ms | 48–49 ms | 48–73 ms |
| 11 s | 96 ms | 102–103 ms | 101 ms | 103–124 ms |
| 34 s | 297–298 ms | 312–326 ms | 305 ms | 319–353 ms |

Model load: wavo 0.08–0.11 s, transcribe.cpp 0.13–0.17 s. Peak memory: wavo 0.31–0.34 GB,
transcribe.cpp 0.34–0.35 GB. Details: [docs/perf.md](docs/perf.md).

Built by coding agents; the rules are in [agents.md](agents.md).
