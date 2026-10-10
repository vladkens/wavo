# wavo

Speech-to-text in Rust. Runs the GGUF ASR models published by
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) on the GPU through `wgpu` with
custom WGSL kernels. The library is pure Rust (no C/C++, ONNX or BLAS); the `wavo` command adds
an ollama-like model manager and decoding of common audio formats.

Works now, on Apple Silicon and (slower for now) on Linux through Vulkan: the four GigaAM v3
models (Russian) and Parakeet TDT 0.6B v2 (English) and v3 (25 European languages), with output
that matches transcribe.cpp exactly (text, tokens and timestamps). More models and platforms are
next: [docs/roadmap.md](docs/roadmap.md).

| Name | Full name | Size | Head | Output |
|---|---|---|---|---|
| `parakeet-v3` | `parakeet-tdt-0.6b-v3` | 0.74 GB | TDT | 25 European languages, cased, punctuated |
| `parakeet-v2` | `parakeet-tdt-0.6b-v2` | 0.73 GB | TDT | English, cased, punctuated |
| `gigaam-v3` | `gigaam-v3-e2e-rnnt` | 0.27 GB | RNN-T | Russian, cased, punctuated |
| `gigaam-v3-e2e-ctc` | | 0.27 GB | CTC | Russian, cased, punctuated |
| `gigaam-v3-rnnt` | | 0.27 GB | RNN-T | Russian, lowercase letters and spaces |
| `gigaam-v3-ctc` | | 0.27 GB | CTC | Russian, lowercase letters and spaces |

## Install

```sh
cargo install --git https://github.com/vladkens/wavo
```

On Linux wavo runs through Vulkan, so it needs a Vulkan driver and access to the GPU:

```sh
sudo apt install build-essential           # a linker for cargo; Rust itself from rustup.rs
sudo apt install mesa-vulkan-drivers       # the driver, with libvulkan1
sudo usermod -aG render $USER              # GPU access, then log out and back in
sudo apt install vulkan-tools              # optional: vulkaninfo
vulkaninfo --summary                       # lists the GPU
```

GPUs other than Apple's run the slower portable kernels for now: an Intel N100 transcribes 11 s
in ~2.5 s with GigaAM and ~3.9 s with Parakeet. Without GPU access wgpu may fall back to llvmpipe,
a software Vulkan driver on the CPU, which is slower than real time.

## Usage

```sh
wavo pull parakeet-v3                 # download a model
wavo run talk.m4a                     # transcribe with parakeet-v3
wavo run gigaam-v3 talk.mp3           # or another model: name, full name or a .gguf path
wavo run talk.wav --json              # text and every token with its start time
wavo run talk.wav --srt > talk.srt    # subtitles
wavo run talk.wav --segment 0         # one pass, no splitting at pauses
wavo list                             # downloaded models
wavo rm parakeet-v3                   # delete one
wavo bench gigaam-v3 talk.wav -n 20   # load time, first call, warm median and minimum
```

Audio can be wav, mp3, m4a (AAC), flac or ogg (Vorbis); it is downmixed to mono and resampled to
16 kHz. `--json` prints `{"text": ..., "tokens": [{"id", "piece", "start_ms"}, ...]}`. `--srt`
cues break after sentences or at 80 characters.

Recordings of any length work: longer audio is split at its quietest moments (by energy, no VAD
model) into segments of up to 25 s for GigaAM, which was trained on utterances up to ~25 s, and
60 s for Parakeet. The segments are transcribed one by one and joined: texts with a space, token
times shifted to the whole file. In one pass, GigaAM loses most words after about a minute and
Parakeet skips sentences; split, 10 minutes take 5.2 s with GigaAM and 8.9 s with Parakeet V3 on
an M2, and memory stays flat. `--segment SECS` sets the length; `--segment 0` runs the whole file
in one pass, as transcribe.cpp does.

Models live in the Hugging Face cache, found as `hf` finds it: `HF_HUB_CACHE` (or the legacy
`HUGGINGFACE_HUB_CACHE`), else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
`~/.cache/huggingface/hub`. The layout is the one `hf` uses, so a model fetched with
`hf download handy-computer/<full name>-gguf <full name>-Q8_0.gguf` is found by `wavo`, and one
pulled by `wavo` is seen by `hf`. Only `wavo pull` uses the network; `wavo run` names the
`wavo pull` command when a model is missing.

Known limitations: Opus is not supported.

## Library

```toml
[dependencies]
wavo = { git = "https://github.com/vladkens/wavo", default-features = false }
```

```rust
let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
let transcript = model.transcribe(&pcm)?; // pcm: 16 kHz mono f32 in [-1, 1]
println!("{}", transcript.text);
for token in &transcript.tokens {
  println!("{} ms {}", token.start_ms, token.piece);
}
```

Without default features (the `cli` feature) you get only the engine: decoding audio,
resampling, splitting long audio and downloading models are up to you. `model.max_audio_ms()`
gives the longest audio a model was trained on (`Some(25_000)` for GigaAM, `None` for Parakeet);
past it accuracy drops (GigaAM loses words after about a minute), so split longer audio first.

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

## Development

Built by coding agents; the rules are in [agents.md](agents.md).

```sh
cargo install --git https://github.com/vladkens/wavo --branch feat/NAME --locked  # a PR branch
cargo install --git https://github.com/vladkens/wavo --locked --force             # back to main
cargo install --path . --locked                                                   # a local checkout
cargo uninstall wavo                                                              # remove it
```
