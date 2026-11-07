<div align="center">

<img src="docs/logo.png" alt="wavo" width="600" />

**Local speech-to-text at GPU speed**

[<img src="https://badges.ws/crates/v/wavo" alt="version" />](https://crates.io/crates/wavo)
[<img src="https://badges.ws/crates/docs/wavo" alt="docs" />](https://docs.rs/wavo)
[<img src="https://badges.ws/github/license/vladkens/wavo" alt="license" />](https://github.com/vladkens/wavo/blob/main/LICENSE)
[<img src="https://badges.ws/badge/follow-@uid127-black" alt="follow @uid127" />](https://x.com/uid127)

</div>

`wavo` turns speech into text on your own machine, with no cloud. Give it a recording of any length and get the text with timestamps. Use it as a command-line tool or as a Rust library.

[Wavo](https://github.com/vladkens/wavo-app) is a dictation app built on this library, for macOS, Windows and Linux.

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

Download the binary for your system from the [latest release](https://github.com/vladkens/wavo/releases/latest) and put it into a folder on your `PATH`, such as `~/.local/bin`.

Or build it with Rust, which takes a few minutes:

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

</details>

<details>
<summary>Windows</summary>

Rust on Windows needs Microsoft's C++ build tools for its linker. In PowerShell:

```powershell
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

wavo uses the GPU through its Vulkan driver, which NVIDIA, AMD and Intel ship. On a laptop with two GPUs it takes the discrete one. Without a Vulkan driver, in a VM for example, wavo falls back to DirectX 12, which is slower.

</details>

### Newest version

This readme follows main, which may be ahead of the latest release. Binaries of main are in the [dev release](https://github.com/vladkens/wavo/releases/tag/dev). To build it with Rust:

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
wavo run talk.wav --json              # text and every token with its start time
wavo run talk.wav --srt > talk.srt    # subtitles
wavo run whisper-turbo talk.mp3 --language he  # Whisper: set the language, not detect it
wavo list                             # downloaded models
wavo rm parakeet-ultra                # delete one
wavo bench gigaam-v3 talk.wav         # load time and speed on your machine
wavo debug                            # your system, GPUs and models, to paste into an issue
wavo run talk.wav --backend gpu       # the GPU even where the Neural Engine is ready
```

- Long recordings are split at pauses by themselves. `--segment SECS` sets the piece length.
- `wavo run` asks before downloading a model. `-y` skips the question.
- Whisper detects the language. `--language CODE` sets it.
- Models live in the Hugging Face cache, shared with `hf download`.
- On a Mac, `wavo pull` prepares Parakeet and GigaAM for the Neural Engine, once, in 15–30 s.
- `--backend gpu` stays on the GPU even where the Neural Engine is ready.

<details>
<summary>Server</summary>

`wavo serve` is a local stand-in for OpenAI's transcription API. An app built for that API works with wavo once you change its URL.

```sh
wavo serve                                       # parakeet-ultra on http://127.0.0.1:8000
wavo serve whisper-turbo gigaam-v3 --port 9000   # these models, loaded up front
wavo serve --host 0.0.0.0 --api-key KEY          # on the network, with a key

curl -F file=@talk.mp3 -F model=whisper-1 http://127.0.0.1:8000/v1/audio/transcriptions
```

It also works as [Bazarr](https://www.bazarr.media)'s Whisper provider, for subtitles. `wavo --help` lists the endpoints and options.

</details>

## 🧠 Models

| Model | Languages | Size | Errors, English | Errors, Russian | 1 hour of audio |
|---|---|---:|---:|---:|---:|
| `parakeet-ultra` (default) | 25 European | 0.74 GB | 1.8 | 4.7 | ~30 s |
| `gigaam-v3` | Russian | 0.27 GB | – | 4.5 | 11 s |
| `parakeet-v2` | English | 0.73 GB | 1.7 | – | ~35 s |
| `whisper-turbo` | 100 | 0.89 GB | 2.0 | 4.5 | ~4 min |
| `moonshine-base` | English | 0.08 GB | 3.3 | – | ~20 s |

Errors are wrong words per 100, lower is better. [All models and languages](docs/models.md).

## ⚡ Speed

How long an hour of speech takes to transcribe, and how much energy it uses. Every engine runs the same model (Parakeet v3) on the same audio.

| Engine | On | 1 hour | Energy | Language | Platforms |
|---|---|---:|---:|---|---|
| wavo | ANE | 18 s | 225 J | Rust | Mac |
| [FluidAudio](https://github.com/FluidInference/FluidAudio) | ANE | 21 s | 164 J | Swift | Mac |
| wavo | GPU | 22 s | 377 J | Rust | Mac, Windows, Linux |
| [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) | GPU | 34 s | 565 J | C++ | Mac, Windows, Linux |
| [parakeet.cpp](https://github.com/mudler/parakeet.cpp) | GPU | 40 s | 472 J | C++ | Mac, Windows, Linux |
| [parakeet-mlx](https://github.com/senstella/parakeet-mlx) | GPU | 59 s | 393 J | Python | Mac |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | CPU | 92 s | 1474 J | C++ | Mac, Windows, Linux |

ANE is Apple's Neural Engine. Measured on a MacBook Pro with an Apple M5.

The full tables and the method are in [docs/benchmarks.md](docs/benchmarks.md).

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

The [API docs](https://docs.rs/wavo) have the rest.

## 🔍 See also

- [handy-computer/transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) – The
  reference wavo follows: C++ on ggml, many model families.
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) – Whisper in C++ on ggml.
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) – Many models on ONNX Runtime,
  streaming included.
- [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx) – Parakeet in Python on
  MLX, Apple Silicon only.

## 🤝 Contributing

Pull requests are welcome, including ones made with agents: point yours at [agents.md](agents.md). When reporting a bug, include the output of `wavo debug`.

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
