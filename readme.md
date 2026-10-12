<div align="center">

<img src="docs/logo.png" alt="wavo" width="600" />

**Local speech-to-text at GPU speed**

[<img src="https://badges.ws/crates/v/wavo" alt="version" />](https://crates.io/crates/wavo)
[<img src="https://badges.ws/crates/docs/wavo" alt="docs" />](https://docs.rs/wavo)
[<img src="https://badges.ws/github/license/vladkens/wavo" alt="license" />](https://github.com/vladkens/wavo/blob/main/LICENSE)
[<img src="https://badges.ws/badge/follow-@uid127-black" alt="follow @uid127" />](https://x.com/uid127)

</div>

`wavo` turns speech into text locally, on the GPU or a Mac's Neural Engine, in pure Rust. Give it
a recording of any length, a voice note or a two-hour meeting, and get the text with timestamps.
It runs the open ASR models that [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
publishes with practically the same output and comparable or better speed. Use it as a
command-line tool with an ollama-like model manager, or as a library that builds with a plain
`cargo build`.

## 🌟 Features

- 📏 Audio of any length: long recordings are split at pauses and joined back, no chunking on
  your side
- 🎧 Any common format: wav, mp3, m4a, flac, ogg (Vorbis), resampled for you
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- ⚡ Runs on the GPU: Metal on Apple Silicon, Vulkan on Linux
- 🧠 On Apple Silicon Macs, Parakeet and GigaAM run on the Neural Engine, 2–3.6× faster than
  the GPU on 11 s of audio, prepared once by `wavo pull`
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
| `whisper-tiny` | 0.05 GB | 99 languages | cased, punctuated |
| `whisper-base` | 0.08 GB | 99 languages | cased, punctuated |
| `whisper-small` | 0.27 GB | 99 languages | cased, punctuated |
| `whisper-medium` | 0.83 GB | 99 languages | cased, punctuated |
| `whisper-large-v3` | 1.67 GB | 100 languages | cased, punctuated |
| `whisper-tiny.en` | 0.05 GB | English | cased, punctuated |
| `whisper-base.en` | 0.08 GB | English | cased, punctuated |
| `whisper-small.en` | 0.27 GB | English | cased, punctuated |
| `whisper-medium.en` | 0.83 GB | English | cased, punctuated |
| `whisper-large` | 1.67 GB | 99 languages | cased, punctuated |
| `whisper-large-v2` | 1.67 GB | 99 languages | cased, punctuated |
| `breeze-asr-25` | 1.67 GB | Taiwanese Mandarin and English | cased, punctuated |
| `moonshine-tiny` | 0.04 GB | English | cased, punctuated |
| `moonshine-base` | 0.08 GB | English | cased, punctuated |

Parakeet is NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
GigaAM is Sber's [GigaAM v3](https://github.com/salute-developers/GigaAM), Whisper is OpenAI's
[Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), Breeze-ASR-25 is MediaTek
Research's [Whisper large-v2 fine-tune](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)
and Moonshine is Useful Sensors' [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny),
small and fast. Moonshine has no timestamps: its tokens all start at 0, and `--srt` gives one cue
per piece of the split audio.
Short names stand for the published ones: `parakeet-v3` is `parakeet-tdt-0.6b-v3`, `parakeet-v2`
is `parakeet-tdt-0.6b-v2`, `gigaam-v3` is `gigaam-v3-e2e-rnnt`, `whisper-turbo` is
`whisper-large-v3-turbo` and `breeze-asr-25` is `Breeze-ASR-25`; full names and a path to a
`.gguf` work too.

## 📥 Installation

Download the binary for macOS (Apple Silicon), Linux or Windows from the
[latest release](https://github.com/vladkens/wavo/releases/latest) and put `wavo` (`wavo.exe`)
into a folder on your `PATH`, such as `~/.local/bin`. Or build it with Rust, which takes a few
minutes:

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
two GPUs it takes the discrete one. GPUs other than Apple's use simpler kernels for now, apart
from matrix multiplication and attention on NVIDIA's tensor cores: an RTX 4060 laptop transcribes
11 s of audio in 42 ms (`gigaam-v3`), 79 ms (`parakeet-v3`) or 0.2 s (`whisper-turbo`).

</details>

### Newest version

This readme follows main, which may be ahead of the latest release. Binaries of main are in the
[dev release](https://github.com/vladkens/wavo/releases/tag/dev), or build it with Rust:

```sh
cargo install --git https://github.com/vladkens/wavo
```

## 🚀 Usage

```sh
wavo talk.m4a                         # transcribe with parakeet-v3, the same as wavo run talk.m4a
wavo run gigaam-v3 talk.mp3           # or with another model
wavo run whisper-turbo talk.mp3 -y    # download a missing model without asking
wavo pull parakeet-v3                 # download a model
wavo pull gigaam-v3 whisper-turbo     # or several at once
wavo run talk.wav --json              # text and every token with its start time
wavo run talk.wav --srt > talk.srt    # subtitles
wavo list                             # downloaded models
wavo rm parakeet-v3                   # delete one
wavo bench gigaam-v3 talk.wav         # load time and speed on your machine
wavo run talk.wav --backend gpu       # the GPU even where the Neural Engine is ready
```

When the model isn't downloaded yet, `wavo run` asks before downloading it, in a terminal; `-y`
downloads without asking, for scripts. Without a terminal and `-y` it fails with the `wavo pull`
command to run, and `wavo bench` never downloads. Long recordings are split at their quietest
moments and transcribed piece by piece, so an hour of audio works as well as a minute;
`--segment SECS` sets the piece length, `--segment 0` runs one pass. Models are stored in the
Hugging Face cache (`~/.cache/huggingface/hub`), so a model fetched with `hf download` is found by
`wavo` and the other way round.

On an Apple Silicon Mac with macOS 15 or later, `wavo pull` also prepares Parakeet and GigaAM for
the Neural Engine: it compiles the model's encoder for Core ML once, which takes 15–30 s and as much
disk as the model, in `~/Library/Caches/wavo`. From then on `wavo run` uses the Neural Engine for
them (`wavo list` shows which are ready) and splits audio into pieces of up to 15 s; the words
are the same as on the GPU on the test clips, while now and then a comma or a capital letter
differs and some tokens start a frame or two apart. `--backend gpu` runs on the
GPU anyway, `--backend ane` fails rather than fall back. Whisper and Moonshine always run on the
GPU, where they are faster.

## ⚡ Speed

Apple M2, the same Q8_0 model file, wavo against transcribe.cpp on Metal, an 11 s clip:

| Model | Transcribe, ms | Load, ms | Memory, MiB |
|---|---|---|---|
| `gigaam-v3` | **96** / 100 | **44** / 104 | **302** / 325 |
| `parakeet-v2` | **143** / 199 | **81** / 245 | **740** / 820 |
| `parakeet-v3` | **154** / 215 | **89** / 245 | **763** / 884 |
| `whisper-turbo` | **1240** / 1390 | **105** / 298 | **1018** / 1055 |

For dictation, where a model is loaded for each recording, the whole run (start, load, transcribe)
is about a quarter to a third shorter. More numbers in [docs/perf.md](docs/perf.md).

On the Neural Engine, warm ms for an 11 s clip / a 4.5 s one, against wavo on the GPU and
transcribe.cpp on Metal:

| Model | Mac | Neural Engine | GPU | transcribe.cpp |
|---|---|---|---|---|
| `parakeet-v3` | M1 | **70** / **53** | 252 / 115 | 295 / 142 |
| `parakeet-v3` | M2 | **57** / **44** | 149 / 72 | 269 / 96 |
| `gigaam-v3` | M1 | **52** / **42** | 170 / 70 | 180 / 74 |
| `gigaam-v3` | M2 | **46** / **40** | 92 / 47 | 105 / 53 |

A load takes 0.1–0.15 s, and the wavo process needs 16–21 MiB: the weights live in the system's
Neural Engine service.

## 📚 Library usage

```sh
cargo add wavo --no-default-features --features hub
```

```rust
use std::ops::ControlFlow;
use wavo::hub::Hub;

let hub = Hub::new(); // the Hugging Face cache, shared with `hf` and `wavo pull`
let path = match hub.find("parakeet-v3") {
  Some(path) => path,
  None => hub.pull("parakeet-v3", |p| {
    eprint!("\r{} / {} MB", p.done / 1_000_000, p.total / 1_000_000);
    ControlFlow::Continue(()) // Break cancels; the next pull resumes
  })?,
};
let model = wavo::Model::load(path)?;
let transcript = model.transcribe(&pcm)?; // pcm: 16 kHz mono f32 in [-1, 1]
println!("{}", transcript.text);
for token in &transcript.tokens {
  println!("{} ms {}", token.start_ms, token.piece);
}
```

The `hub` feature downloads models by name (it adds an HTTP client); leave it out to load `.gguf`
files you fetch yourself, with a pure Rust build. Decoding and resampling audio are up to you.
`model.transcribe_long(&pcm)` splits long recordings at pauses into pieces of up to
`model.max_audio_ms()` (15 s on the Neural Engine, 25 s for GigaAM, 30 s for Whisper and
Moonshine) and joins the transcripts; `wavo::split` gives the pieces to transcribe yourself. On a
Mac, `hub.pull` also prepares Parakeet and GigaAM for the Neural Engine, `wavo::Model::prepare`
does it for a file you manage, and `Model::load` then runs them there. The system caches this
per program, so your app prepares for itself even when `wavo pull` already did (then it takes
one load of 10–30 s, without rebuilding); `Model::load_with(path, Backend::Gpu)` picks the GPU. The [API docs](https://docs.rs/wavo) have
the details, including downloading on a thread with progress and cancel, as
[examples/hub.rs](examples/hub.rs) does.

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

For a headless machine, Remote Desktop (built into Windows Pro) works without a monitor. Allow it
and SSH only from the local network (here 192.168.8.0/23; `LocalSubnet` would leave out a client
in the neighbouring subnet). A Windows update can switch the network to Public, which blocks the
default SSH rule, so allow both rules on every profile:

```powershell
Set-NetFirewallRule -DisplayGroup 'Remote Desktop' -Profile Any -RemoteAddress 192.168.8.0/23
Set-NetFirewallRule -Name OpenSSH-Server-In-TCP -Profile Any -RemoteAddress 192.168.8.0/23
Enable-NetFirewallRule -DisplayGroup 'Remote Desktop'
Set-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name fDenyTSConnections -Value 0
Set-NetConnectionProfile -InterfaceAlias '<adapter>' -NetworkCategory Private
```

On a Mac, connect with Windows App (App Store), signing in with the account's password, not its
PIN. It needs Local Network access in System Settings, and an outbound firewall such as LuLu must
allow it to reach the machine on TCP 3389. Error 0x4 is a generic connection failure: when
`nc -z <host> 3389` works from the Terminal, the block is on the Mac.

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
