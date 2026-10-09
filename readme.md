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

Apple M2, same Q8_0 GGUF, transcribe.cpp on Metal. Warm is the median of 10 calls.

| Clip | wavo warm | transcribe.cpp warm | wavo first call | transcribe.cpp first call |
|---|---|---|---|---|
| 4.5 s | 45 ms | 44 ms | 71–84 ms | 59–70 ms |
| 11 s | 102–104 ms | 103 ms | 131–135 ms | 103–118 ms |
| 34 s | 340–350 ms | 312–326 ms | 372–388 ms | 319–331 ms |

Model load: wavo 0.07 s, transcribe.cpp 0.13 s. Peak memory: wavo 0.9 GB, transcribe.cpp
0.33 GB. Details: [docs/perf.md](docs/perf.md).

Built by coding agents; the rules are in [agents.md](agents.md).
