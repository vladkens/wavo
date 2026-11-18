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
- 💻 No GPU? Every model runs on the CPU, with the same text
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- 📦 `pull` / `list` / `rm` models, shared with the Hugging Face cache that `hf` uses
- 📝 Plain text, JSON with a start time for every token, or SRT subtitles
- 🗣️ Who speaks when: `--diarize` puts a speaker on every line, up to 8 speakers
- 🔌 `wavo serve`: a local server for apps built on OpenAI's transcription API
- 🎬 Subtitles for your films and series as [Bazarr](https://www.bazarr.media)'s Whisper provider

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
<summary>Docker</summary>

The image `ghcr.io/vladkens/wavo` runs on x86_64 and arm64 Linux, with Mesa's Vulkan drivers for Intel and AMD GPUs.

```sh
docker run --rm --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v wavo:/home/wavo/.cache -v "$PWD:/data" ghcr.io/vladkens/wavo run /data/talk.mp3 -y
```

- `--device /dev/dri` and `--group-add` give it the GPU; without them it runs on the CPU.
- It runs as the user `wavo` (uid 1000), not root, so it needs the GPU's group.
- The volume keeps the models and the GPU's compiled kernels between runs.

</details>

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
wavo run talk.wav --srt > talk.srt    # subtitles
wavo run talk.wav --json              # text, its language and every token with its start time
wavo run talk.wav --diarize           # Speaker 1: ... lines, also with --json and --srt
wavo pull gigaam-v3 whisper-turbo     # download models
wavo list                             # downloaded models
wavo rm gigaam-v3                     # delete one
wavo bench gigaam-v3 talk.wav         # load time and speed on your machine
wavo devices                          # the GPUs wavo sees, numbered for --device
wavo debug                            # your system, GPUs and models, to paste into an issue
```

- `wavo run` asks before downloading a model; `-y` skips the question.
- Long recordings are split at pauses by themselves; `--segment SECS` sets the piece length.
- Whisper detects the language; `--language CODE` sets it.
- Subtitles follow Netflix's style: two lines of up to 42 characters, 5/6s to 7s a cue.
- `--diarize` tells up to 8 speakers apart with NVIDIA's Nemotron 3 Diarization (0.11GB).
- `--device NAME` picks another GPU by part of its name or its number in `wavo devices`.
- `--backend cpu` runs on the CPU; `--backend gpu` skips a Mac's Neural Engine.
- On a Mac, `wavo pull` prepares Parakeet and GigaAM for the Neural Engine, once, in 15–30s.
- Models live in the Hugging Face cache, shared with `hf download`.

Without a GPU, 30s of Russian take 5.4s with `gigaam-v3` (0.34GB of memory), 7.7s with `parakeet-v3` (0.84GB) and 18s with `whisper-small` (0.56GB) on two N100 cores.

<details>
<summary>Server</summary>

`wavo serve` is a local stand-in for OpenAI's transcription API. An app built for that API works with wavo once you change its URL.

```sh
wavo serve                                       # parakeet-ultra on http://127.0.0.1:8000
wavo serve whisper-turbo gigaam-v3 --port 9000   # these models, loaded up front
wavo serve --host 0.0.0.0 --api-key KEY          # on the network, with a key
wavo serve --unload-after 10m                    # free the memory after 10 idle minutes
wavo serve parakeet-ultra nemotron-diarization   # also who speaks: diarized_json

curl -F file=@talk.mp3 -F model=whisper-1 http://127.0.0.1:8000/v1/audio/transcriptions
```

- `model` picks any downloaded model by name, loaded on first use, without a restart.
- A name wavo doesn't know, such as `whisper-1`, goes to the first model given.
- `wavo --help` lists the endpoints and options.

</details>

## 🎬 Subtitles with Bazarr

wavo works as [Bazarr](https://www.bazarr.media)'s Whisper provider: it writes subtitles for the films and episodes that have none, on an Intel or AMD GPU, or on the CPU.

1. Add wavo to Bazarr's compose file:

   ```yaml
   services:
     wavo:
       image: ghcr.io/vladkens/wavo
       command: serve parakeet-ultra --host 0.0.0.0 --port 9000 -y
       devices:
         - /dev/dri:/dev/dri # Intel or AMD GPU; without it, the CPU
       group_add:
         - "993" # yours: stat -c %g /dev/dri/renderD128
       # gpus: all         # NVIDIA, with its Container Toolkit, instead of devices (untested)
       volumes:
         - wavo:/home/wavo/.cache # the models
       restart: unless-stopped
   volumes:
     wavo:
   ```

2. In Bazarr's providers, turn on Whisper with `http://wavo:9000` as its endpoint.
3. Lower Bazarr's minimum scores below Whisper's fixed 61% for episodes and 33% for movies.

`parakeet-ultra` knows [25 European languages](docs/models.md); for others, serve `whisper-turbo`. Untagged audio tracks get their language from wavo too. The rest is on [Bazarr's Whisper page](https://wiki.bazarr.media/Additional-Configuration/Whisper-Provider/).

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

GigaAM, Windows, speaker diarization and the method are in [docs/benchmarks.md](docs/benchmarks.md).

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
