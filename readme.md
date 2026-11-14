<div align="center">

<img src="docs/logo.png" alt="wavo" width="600" />

**Local speech-to-text at GPU speed**

[<img src="https://badges.ws/crates/v/wavo" alt="version" />](https://crates.io/crates/wavo)
[<img src="https://badges.ws/crates/docs/wavo" alt="docs" />](https://docs.rs/wavo)
[<img src="https://badges.ws/github/license/vladkens/wavo" alt="license" />](https://github.com/vladkens/wavo/blob/main/LICENSE)
[<img src="https://badges.ws/badge/follow-@uid127-black" alt="follow @uid127" />](https://x.com/uid127)

</div>

`wavo` turns speech into text on your own machine, on the GPU or a Mac's Neural Engine. Give it a recording of any length, from a voice note to a two-hour meeting, and get the text with timestamps. It runs the open models that [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) publishes, as a command-line tool with an ollama-like model manager or as a pure Rust library.

**Looking for dictation?** [Wavo](https://github.com/vladkens/wavo-app) is the desktop app built on this library, for macOS, Windows and Linux: hold a hotkey, speak, and the text lands where you were typing. Report the app's bugs [there](https://github.com/vladkens/wavo-app/issues).

## 🌟 Features

- 📏 Audio of any length: long recordings are split at pauses and joined back
- 🎧 Any common format: wav, mp3, m4a, flac, ogg (Vorbis), resampled for you
- ⚡ Runs on the GPU: Metal on a Mac, Vulkan on Linux and Windows
- 🧠 Runs on a Mac's Neural Engine too, for Parakeet and GigaAM
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- 📦 `pull` / `list` / `rm` models, shared with the Hugging Face cache that `hf` uses
- 📝 Plain text, JSON with a start time for every token, or SRT subtitles
- 🔌 `wavo serve`: a local server for apps built on OpenAI's transcription API

## 📥 Installation

Install it with [Homebrew](https://brew.sh/) on macOS (Apple Silicon) and Linux:

```sh
brew install vladkens/tap/wavo
```

Or download the binary for macOS (Apple Silicon), Linux or Windows from the [latest release](https://github.com/vladkens/wavo/releases/latest) and put it into a folder on your `PATH`. Or build it with Rust, which takes a few minutes:

```sh
cargo install wavo
```

<details>
<summary>Linux</summary>

wavo runs through Vulkan, so it needs a Vulkan driver and access to the GPU:

```sh
sudo apt install build-essential           # a linker for cargo; Rust itself from rustup.rs
sudo apt install mesa-vulkan-drivers       # the driver
sudo usermod -aG render $USER              # GPU access, then log out and back in
```

GPUs other than Apple's use simpler kernels for now. An Intel N100 takes 1.3–2.2s for 11s of audio.

</details>

<details>
<summary>Windows</summary>

Rust on Windows needs Microsoft's C++ build tools for its linker. In PowerShell:

```powershell
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

wavo uses the GPU through its Vulkan driver, which NVIDIA, AMD and Intel ship. On a laptop with two GPUs it takes the discrete one.

GPUs other than Apple's use simpler kernels for now, except for matrix multiplication and attention on NVIDIA's tensor cores. An RTX 4060 laptop takes 42ms for 11s of audio with `gigaam-v3`, 79ms with `parakeet-v3` and 0.2s with `whisper-turbo`.

Without a Vulkan driver, in a VM for example, wavo falls back to DirectX 12 with the simpler kernels. `WGPU_BACKEND=dx12` forces it.

A server without a GPU runs GigaAM and Parakeet on the CPU, with the same text: 30s of Russian take 5.4s with GigaAM (0.34GB of memory) and 7.7s with Parakeet V3 (0.84GB) on two N100 cores. Other models need a GPU there for now.

</details>

### Newest version

This readme follows main, which may be ahead of the latest release. Binaries of main are in the [dev release](https://github.com/vladkens/wavo/releases/tag/dev), or build it with Rust:

```sh
cargo install --git https://github.com/vladkens/wavo
```

## 🚀 Usage

```sh
wavo talk.m4a                         # transcribe with parakeet-ultra, the same as wavo run talk.m4a
wavo run gigaam-v3 talk.mp3           # or with another model
wavo run whisper-turbo talk.mp3 -y    # download a missing model without asking
wavo pull parakeet-ultra              # download a model
wavo pull gigaam-v3 whisper-turbo     # or several at once
wavo run talk.wav --json              # text, its language and every token with its start time
wavo run talk.wav --srt > talk.srt    # subtitles
wavo run whisper-turbo talk.mp3 --language he  # Whisper: set the language, not detect it
wavo list                             # downloaded models
wavo rm parakeet-ultra                # delete one
wavo bench gigaam-v3 talk.wav         # load time and speed on your machine
wavo debug                            # your system, GPUs and models, to paste into an issue
wavo run talk.wav --backend gpu       # the GPU even where the Neural Engine is ready
wavo run gigaam-v3 talk.wav --backend cpu  # on the CPU, as without a GPU
wavo devices                          # the GPUs, numbered for --device
wavo run talk.wav --device radeon     # another GPU, by part of its name or its number
```

- Long recordings are split at pauses by themselves. `--segment SECS` sets the piece length.
- `wavo run` asks before downloading a model. `-y` skips the question.
- Whisper detects the language. `--language CODE` sets it.
- Models live in the Hugging Face cache, shared with `hf download`.
- On a Mac, `wavo pull` prepares Parakeet and GigaAM for the Neural Engine, once, in 15–30s.
- `--backend gpu` stays on the GPU even where the Neural Engine is ready.

<details>
<summary>Server</summary>

`wavo serve` is a local stand-in for OpenAI's transcription API. An app built for that API works with wavo once you change its URL.

```sh
wavo serve                                       # parakeet-ultra on http://127.0.0.1:8000
wavo serve whisper-turbo gigaam-v3 --port 9000   # these models, loaded up front
wavo serve --host 0.0.0.0 --api-key KEY          # on the network, with a key
wavo serve --unload-after 10m                    # free the memory after 10 idle minutes

curl -F file=@talk.mp3 -F model=whisper-1 http://127.0.0.1:8000/v1/audio/transcriptions
```

It also works as [Bazarr](https://www.bazarr.media)'s Whisper provider, for subtitles, and names the language of an untagged audio track. `wavo --help` lists the endpoints and options.

</details>

## 🧠 Models

| Model | Languages | Size | Errors, English | Errors, Russian | 1 hour of audio |
|---|---|---:|---:|---:|---:|
| `parakeet-ultra` (default) | 25 European | 0.74GB | 1.8 | 4.7 | 32s |
| `gigaam-v3` | Russian | 0.27GB | – | 4.5 | 12s |
| `parakeet-v2` | English | 0.73GB | 1.7 | – | 36s |
| `whisper-turbo` | 100 | 0.89GB | 2.0 | 4.5 | 4.5m |
| `moonshine-base` | English | 0.08GB | 3.3 | – | 22s |

Errors are wrong words per 100, lower is better. Times are from a MacBook Air with an Apple M5. [All models and languages](docs/models.md).

## ⚡ Speed

How fast each engine gives you the text. Every engine runs the same model (Parakeet v3) on the same audio.

| Engine | On | 5s phrase | Long audio | Energy |
|---|---|---:|---:|---:|
| wavo | ANE | 0.12s | 200× | 225J |
| [FluidAudio](https://github.com/FluidInference/FluidAudio) | ANE | 0.13s | 175× | 164J |
| wavo | GPU | 0.14s | 160× | 377J |
| [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) | GPU | 0.12s | 107× | 565J |
| [parakeet.cpp](https://github.com/mudler/parakeet.cpp) | GPU | 0.77s | 90× | 472J |
| [parakeet-mlx](https://github.com/senstella/parakeet-mlx) | GPU | 0.63s | 61× | 393J |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | CPU | 0.83s | 39× | 1474J |

*5s phrase* is the whole cycle of a short recording: start the program, load the model, transcribe, exit. *Long audio* is how many times faster than real time an hour-long recording goes. *Energy* is what that hour costs on the CPU, GPU and Neural Engine, measured with [macmon](https://github.com/vladkens/macmon). ANE is Apple's Neural Engine. Measured on a MacBook Pro with an Apple M5.

GigaAM, Windows and the method are in [docs/benchmarks.md](docs/benchmarks.md).

## 📚 Library usage

```sh
cargo add wavo --no-default-features --features hub
```

```rust
let pcm: Vec<f32> = read_audio(); // 16 kHz mono in [-1, 1]; decoding is up to you
let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
let transcript = model.transcribe(&pcm)?;
println!("{}", transcript.text);
```

The [API docs](https://docs.rs/wavo) have the rest: downloading models, long recordings, progress and cancel.

## 🔍 See also

- [handy-computer/transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) – The
  reference wavo follows: C++ on ggml, many model families.
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) – Whisper in C++ on ggml.
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) – Many models on ONNX Runtime,
  streaming included.
- [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx) – Parakeet in Python on
  MLX, Apple Silicon only.

## 🤝 Contributing

Pull requests are welcome, including ones made with agents: point yours at [agents.md](agents.md). When reporting a bug, include the output of `wavo debug`: it shows the GPU wavo runs on and which of its kernels the GPU allows.

<details>
<summary>Installing a branch</summary>

```sh
cargo install --git https://github.com/vladkens/wavo --branch feat/NAME --locked  # a PR branch
cargo install --git https://github.com/vladkens/wavo --locked --force             # back to main
cargo install --path . --locked                                                   # a local checkout
cargo uninstall wavo                                                              # remove it
```

</details>

## 📝 License

Distributed under the [MIT License](LICENSE).
