<div align="center">

<img src="docs/logo.png" alt="wavo" width="600" />

**Speech to text on your GPU, in pure Rust**

[<img src="https://badges.ws/crates/v/wavo" alt="version" />](https://crates.io/crates/wavo)
[<img src="https://badges.ws/github/license/vladkens/wavo" alt="license" />](https://github.com/vladkens/wavo/blob/main/LICENSE)

</div>

`wavo` turns speech into text locally, on the GPU, in pure Rust. Give it a recording of any
length, a voice note or a two-hour meeting, and get the text with timestamps. It runs the open ASR
models that [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) publishes with
practically the same output and comparable or better speed. Use it as a command-line tool with an
ollama-like model manager, or as a library that builds with a plain `cargo build`.

## 🌟 Features

- 📏 Audio of any length: long recordings are split at pauses and joined back, no chunking on
  your side
- 🎧 Any common format: wav, mp3, m4a, flac, ogg (Vorbis), resampled for you
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- ⚡ Runs on the GPU: Metal on Apple Silicon, Vulkan on Linux
- 🏎️ As fast as transcribe.cpp or faster on the same model files, and loads in 0.1–0.2 s
- 🎯 Same text, tokens and timestamps as transcribe.cpp on its test clips
- 📦 `pull` / `list` / `rm` models, shared with the Hugging Face cache that `hf` uses
- 📝 Plain text, JSON with a start time for every token, or SRT subtitles
- 🌍 English, Russian, 25 European languages, and 100 with Whisper

## 🧠 Models

| Name | Size | Languages | Output |
|---|---|---|---|
| `parakeet-v3` (default) | 0.74 GB | 25 European languages | cased, punctuated |
| `parakeet-v2` | 0.73 GB | English | cased, punctuated |
| `gigaam-v3` | 0.27 GB | Russian | cased, punctuated |
| `gigaam-v3-e2e-ctc` | 0.27 GB | Russian | cased, punctuated |
| `gigaam-v3-rnnt` | 0.27 GB | Russian | lowercase, no punctuation |
| `gigaam-v3-ctc` | 0.27 GB | Russian | lowercase, no punctuation |
| `whisper-turbo` | 0.89 GB | 100 languages | cased, punctuated |

Parakeet is NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
GigaAM is Sber's [GigaAM v3](https://github.com/salute-developers/GigaAM), Whisper is OpenAI's
[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
Short names stand for the published ones: `parakeet-v3` is `parakeet-tdt-0.6b-v3`, `parakeet-v2`
is `parakeet-tdt-0.6b-v2`, `gigaam-v3` is `gigaam-v3-e2e-rnnt` and `whisper-turbo` is
`whisper-large-v3-turbo`; full names and a path to a `.gguf` work too.

## 📥 Installation

```sh
cargo install wavo
```

Or take prebuilt dev binaries for macOS, Linux and Windows from the
[dev release](https://github.com/vladkens/wavo/releases/tag/dev).

<details>
<summary>Linux</summary>

wavo runs through Vulkan, so it needs a Vulkan driver and access to the GPU:

```sh
sudo apt install build-essential           # a linker for cargo; Rust itself from rustup.rs
sudo apt install mesa-vulkan-drivers       # the driver
sudo usermod -aG render $USER              # GPU access, then log out and back in
```

GPUs other than Apple's use simpler kernels for now: an Intel N100 transcribes 11 s of audio in
2.1–3.6 s.

</details>

<details>
<summary>Windows</summary>

Rust on Windows needs Microsoft's C++ build tools for its linker. In PowerShell:

```powershell
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

wavo uses the GPU through its Vulkan driver (NVIDIA, AMD and Intel ship one); on a laptop with
two GPUs it takes the discrete one. GPUs other than Apple's use simpler kernels for now: an
RTX 4060 laptop transcribes 11 s of audio in 64 ms (`gigaam-v3`), 102 ms (`parakeet-v3`) or
1.1 s (`whisper-turbo`).

</details>

## 🚀 Usage

```sh
wavo pull parakeet-v3                 # download a model
wavo run talk.m4a                     # transcribe with parakeet-v3
wavo run gigaam-v3 talk.mp3           # or with another model
wavo run talk.wav --json              # text and every token with its start time
wavo run talk.wav --srt > talk.srt    # subtitles
wavo list                             # downloaded models
wavo rm parakeet-v3                   # delete one
wavo bench gigaam-v3 talk.wav         # load time and speed on your machine
```

Long recordings are split at their quietest moments and transcribed piece by piece, so an hour of
audio works as well as a minute; `--segment SECS` sets the piece length, `--segment 0` runs one
pass. Models are stored in the Hugging Face cache (`~/.cache/huggingface/hub`), so a model fetched
with `hf download` is found by `wavo` and the other way round. Only `wavo pull` uses the network.

## ⚡ Speed

Apple M2, the same Q8_0 model file, wavo against transcribe.cpp on Metal, an 11 s clip:

| Model | Transcribe, ms | Load, ms | Memory, MiB |
|---|---|---|---|
| `gigaam-v3` | **96** / 100 | **77** / 124 | **302** / 325 |
| `parakeet-v2` | **143** / 199 | **159** / 274 | **740** / 820 |
| `parakeet-v3` | **154** / 215 | **169** / 284 | **763** / 884 |
| `whisper-turbo` | **1240** / 1390 | **215** / 310 | **1018** / 1055 |

For dictation, where a model is loaded for each recording, the whole run (start, load, transcribe)
is about a quarter to a third shorter. More numbers in [docs/perf.md](docs/perf.md).

## 📚 Library usage

```sh
cargo add wavo --no-default-features
```

```rust
let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
let transcript = model.transcribe(&pcm)?; // pcm: 16 kHz mono f32 in [-1, 1]
println!("{}", transcript.text);
for token in &transcript.tokens {
  println!("{} ms {}", token.start_ms, token.piece);
}
```

Without default features you get only the engine: decoding and resampling audio, splitting long
recordings and downloading models are up to you. `model.max_audio_ms()` tells how long a piece
the model was trained on (25 s for GigaAM, 30 s for Whisper), so you know where to split.

## 🤝 Contributing

wavo is written by AI coding agents. A person set the direction; the agents wrote the code and
the GPU kernels, matched the reference output and searched for the fastest solutions themselves.
Pull requests are welcome, and so are your agents: point them at [agents.md](agents.md) and let
them make it better.

<details>
<summary>Installing a branch</summary>

```sh
cargo install --git https://github.com/vladkens/wavo --branch feat/NAME --locked  # a PR branch
cargo install --git https://github.com/vladkens/wavo --locked --force             # back to main
cargo install --path . --locked                                                   # a local checkout
cargo uninstall wavo                                                              # remove it
```

</details>

<details>
<summary>A Windows machine for testing over SSH</summary>

In an administrator PowerShell: OpenSSH server, Git (its bash becomes the SSH shell), CMake and
make, and the Rust toolchain with Microsoft's build tools, as in Installation → Windows. sshd
reads an administrator's key from `C:\ProgramData\ssh`, not from `~\.ssh`.

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd; Set-Service sshd -StartupType Automatic
winget install Git.Git Kitware.CMake ezwinports.make
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Program Files\Git\bin\bash.exe" -PropertyType String -Force
Add-Content C:\ProgramData\ssh\administrators_authorized_keys "ssh-ed25519 AAAA... you@host"
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
```

`ssh user@host` then opens Git Bash, where `make check` and `make test` work as on macOS (rustup
fetches the nightly toolchain for `cargo +nightly fmt` on first use), and `make reference` builds
transcribe.cpp for the CPU with Visual Studio's compiler; for its GPU build run `winget install
KhronosGroup.VulkanSDK` (in PowerShell) and add `-DTRANSCRIBE_VULKAN=ON` to the cmake configure
step. `ssh user@host wsl` opens the default WSL distribution instead, a separate Linux with its own
toolchain.

</details>

## 📝 License

Distributed under the [MIT License](LICENSE).

## 🔍 See also

- [handy-computer/transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) – The
  reference wavo follows: C++ on ggml, many model families.
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) – Whisper in C++ on ggml.
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) – Many models on ONNX Runtime,
  streaming included.
- [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx) – Parakeet in Python on
  MLX, Apple Silicon only.
