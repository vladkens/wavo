<div align="center">

<img src="docs/logo.png" alt="wavo" width="600" />

**Local speech-to-text at GPU speed**

[<img src="https://badges.ws/crates/v/wavo" alt="version" />](https://crates.io/crates/wavo)
[<img src="https://badges.ws/crates/docs/wavo" alt="docs" />](https://docs.rs/wavo)
[<img src="https://badges.ws/github/license/vladkens/wavo" alt="license" />](https://github.com/vladkens/wavo/blob/main/LICENSE)
[<img src="https://badges.ws/badge/follow-@uid127-black" alt="follow @uid127" />](https://x.com/uid127)

</div>

`wavo` turns speech into text locally, on the GPU, in pure Rust. Give it a recording of any
length, a voice note or a two-hour meeting, and get the text with timestamps. It runs the open ASR
models that [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) publishes with
practically the same output, faster than transcribe.cpp on Apple Silicon and NVIDIA GPUs, on par
or slower on integrated Intel GPUs for some models. Use it as a command-line tool with an
ollama-like model manager, or as a library that builds with a plain `cargo build`.

## 🌟 Features

- 📏 Audio of any length: long recordings are split at pauses and joined back, no chunking on
  your side
- 🎧 Any common format: wav, mp3, m4a, flac, ogg (Vorbis), resampled for you
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- ⚡ Runs on the GPU: Metal on Apple Silicon, Vulkan on Linux and Windows (DX12 without Vulkan)
- 🏎️ Faster than transcribe.cpp on the same model files on Apple Silicon and NVIDIA (see
  [Speed](#-speed)), and loads in about 0.1 s on an M2
- 🎯 Same text, tokens and timestamps as transcribe.cpp on its test clips
- 📦 `pull` / `list` / `rm` models, shared with the Hugging Face cache that `hf` uses
- 📝 Plain text, JSON with a start time for every token, or SRT subtitles
- 🌍 English, Russian, 25 European languages, and 100 with Whisper

## 🤖 How it's made

wavo is written by AI agents (Claude Code, Codex) following [agents.md](agents.md); I choose what
to build and review every pull request. Each model's output is checked against transcribe.cpp's
own ([fixtures](tests/fixtures/)), and every speed change is measured and logged in
[docs/perf.md](docs/perf.md). This isn't a weekend prompt: I've written software by hand for years
— [macmon](https://github.com/vladkens/macmon) (Rust, Apple Silicon system monitor),
[twscrape](https://github.com/vladkens/twscrape) (Python, X/Twitter scraping) and
[ecloop](https://github.com/vladkens/ecloop) (C, low-level elliptic-curve math) — and agents are
how I build now.

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
1.3–2.2 s.

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

Without a Vulkan driver (e.g. a VM) wavo falls back to DirectX 12 with the simpler kernels only;
`WGPU_BACKEND=dx12` forces it.

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
wavo debug                            # your system, GPUs and models, to paste into an issue
```

When the model isn't downloaded yet, `wavo run` asks before downloading it, in a terminal; `-y`
downloads without asking, for scripts. Without a terminal and `-y` it fails with the `wavo pull`
command to run, and `wavo bench` never downloads. Long recordings are split at their quietest
moments and transcribed piece by piece, so an hour of audio works as well as a minute;
`--segment SECS` sets the piece length, `--segment 0` runs one pass (up to 37.5 s for
GigaAM and 45 s for Moonshine). Models are stored in the
Hugging Face cache (`~/.cache/huggingface/hub`), so a model fetched with `hf download` is found by
`wavo` and the other way round.

## ⚡ Speed

Seconds to transcribe one minute of audio on an **Apple M2** (10-core GPU), 2026-10-21. The
short clip is a warm call with the model loaded (`wavo bench`, 35 s `dots`, `ru-long` for
GigaAM); the 1 h file is `wavo run` on an hour-long English
[LibriVox reading](https://archive.org/details/memoirsofsherlockholmesv3_1502_librivox), with the
model load and mp3 decoding. Whisper pads audio to 30 s windows, so the 35 s clip costs it two.

| Model | Short clip, s per min | 1 h file, s per min | 1 h file, × real time |
|---|---|---|---|
| `gigaam-v3` | 0.45 | 0.51 | 118× |
| `parakeet-v3` | 0.81 | 0.88 | 68× |
| `parakeet-v2` | 0.74 | 0.82 | 74× |
| `whisper-turbo` | 4.6 | 3.3 | 18× |
| `whisper-tiny` | 0.34 | 0.36 | 165× |
| `moonshine-tiny` | 0.33 | 0.33 | 180× |
| `moonshine-base` | 0.53 | 0.50 | 120× |

Against transcribe.cpp: transcribe times are warm medians of `wavo bench MODEL CLIP` against
`transcribe-bench` of transcribe.cpp at 5bb2deb, on the same Q8_0 file and clip; cells are
wavo / transcribe.cpp. Bold marks the better side where the gap is over 3%.

**Apple M2** (10-core GPU, 24 GiB, macOS 27) against transcribe.cpp on Metal, an 11 s clip (`jfk`,
GigaAM `ru-short`), 2026-10-09 to 11; the GPU changes since left these models within noise on the
M2.

| Model | Transcribe, ms | Load, ms | Peak memory, MiB |
|---|---|---|---|
| `gigaam-v3` | **96** / 100 | **44** / 104 | **302** / 325 |
| `parakeet-v2` | **143** / 199 | **81** / 245 | **740** / 820 |
| `parakeet-v3` | **154** / 215 | **89** / 245 | **763** / 884 |
| `whisper-turbo` | **1240** / 1390 | **105** / 298 | **1018** / 1055 |

For dictation, where a model is loaded for each recording, load plus the first call takes 142 /
217 ms (`gigaam-v3`), 256 / 468 ms (`parakeet-v3`) and 1437 / 1756 ms (`whisper-turbo`).
Whisper's other sizes and Moonshine are faster too
([Whisper plan](docs/plans/completed/20261010-whisper.md),
[Moonshine plan](docs/plans/20261011-moonshine.md)).

**Apple M5** (10-core GPU, macOS 27.0.1) against transcribe.cpp on Metal, 2026-10-12. On the M5
both multiply on the GPU's neural accelerators through the Metal 4 tensor API. Clips: `jfk` 11 s,
`ru` 4.5 s, `ru-long` 34 s, `dots` 35 s.

| Model | Clip | wavo, ms | transcribe.cpp, ms |
|---|---|---|---|
| `gigaam-v3` | `ru` / `ru-long` | **17.3 / 87** | 20.7 / 120 |
| `parakeet-v3` | `jfk` / `dots` | **60 / 185** | 94 / 313 |
| `whisper-turbo` | `jfk` | **420** | 573 |
| `moonshine-tiny` | `jfk` | **20.3** | 70 |
| `moonshine-base` | `jfk` | **30.9** | 120 |

**NVIDIA RTX 4060 Laptop** (Windows 11, Vulkan) against transcribe.cpp's Vulkan build, 2026-10-11
to 13. Clips: `jfk` 11 s, `ru` 4.5 s, `ru-long` 34 s, `dots` 35 s.

| Model | Clip | wavo, ms | transcribe.cpp, ms |
|---|---|---|---|
| `gigaam-v3` | `ru` / `ru-long` | **16.9 / 70** | 134 / 1064 |
| `parakeet-v3` | `jfk` / `dots` | **68 / 180** | 88 / 249 |
| `parakeet-v3`, first call | `jfk` | **77** | 99 |
| `whisper-turbo` | `jfk` / `ru-long` | **197 / 452** | 226 / 503 |
| `whisper-small` | `jfk` | **69** | 101 |
| `whisper-medium` | `jfk` | **198** | 226 |
| `whisper-large-v3` | `jfk` | **375** | 427 |
| `whisper-tiny` | `jfk` / `dots` | **23 / 89** | 43 / 135 |
| `moonshine-tiny` | `dots` | **90** | 144 |
| `moonshine-base` | `dots` | **141** | 227 |

**Intel N100** (integrated GPU, Linux, portable kernels) against transcribe.cpp's Vulkan build,
2026-10-12. `whisper-turbo` against the reference's CPU build, as its Vulkan build times out there.
Details in the [speed plan](docs/plans/20261012-speed.md).

| Model | Clip | wavo, ms | transcribe.cpp, ms |
|---|---|---|---|
| `gigaam-v3` | `ru` / `ru-long` | **524 / 4170** | 1091 / 7994 |
| `parakeet-v3` | `jfk` / `dots` | **2199 / 5702** | 3271 / 8830 |
| `whisper-tiny` | `jfk` / `dots` | **649 / 1696** | 1340 / 3052 |
| `whisper-turbo` | `jfk` | **23224** | 41676 (CPU) |

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
eprintln!("running on {}", model.device()); // e.g. Apple M2 (Metal, integrated)
let transcript = model.transcribe(&pcm)?; // pcm: 16 kHz mono f32 in [-1, 1]

println!("{}", transcript.text);

for token in &transcript.tokens {
  println!("{} ms {}", token.start_ms, token.piece);
}
```

The `hub` feature downloads models by name (it adds an HTTP client); leave it out to load `.gguf`
files you fetch yourself, with a pure Rust build. Decoding and resampling audio and splitting long
recordings are up to you: `model.max_audio_ms()` tells how long a piece the model was trained on
(25 s for GigaAM, 30 s for Whisper and Moonshine), and GigaAM and Moonshine return
`Error::AudioTooLong` past 1.5 times that. The [API docs](https://docs.rs/wavo) have the details,
including downloading on a thread with progress and cancel, as
[examples/hub.rs](examples/hub.rs) does.

## 🤝 Contributing

Pull requests are welcome, and so are your agents: point them at [agents.md](agents.md). When
reporting a bug, include the output of `wavo debug`: it shows the GPU wavo runs on and which of
its kernels the GPU allows.

<details>
<summary>Installing a branch</summary>

```sh
cargo install --git https://github.com/vladkens/wavo --branch feat/NAME --locked  # a PR branch
cargo install --git https://github.com/vladkens/wavo --locked --force             # back to main
cargo install --path . --locked                                                   # a local checkout
cargo uninstall wavo                                                              # remove it
```

</details>

## 🙏 Credits

wavo's kernels and algorithms are built from the ideas and code of these projects:

- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT): the reference wavo
  matches, its GGUF format and converters, the published models and the test clips. wavo's FFT,
  mel frontends, TDT and Whisper decoding, repetition guard and SentencePiece decoding follow its
  code.
- [ggml](https://github.com/ggml-org/ggml) (MIT): kernels adapted from it. Flash attention after
  Vulkan's `flash_attn_cm1.comp`, the Q8_0 matrix multiplication after ggml-metal's
  `kernel_mul_mm`, and GELU's erf.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT),
  [Hugging Face transformers](https://github.com/huggingface/transformers) and
  [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Apache-2.0): algorithms that reached wavo through
  transcribe.cpp's ports.

**Models.** wavo ships no weights: `wavo pull` downloads the GGUF conversions that
[handy-computer](https://huggingface.co/handy-computer) publishes on Hugging Face, under the
original licenses:

- Parakeet TDT ([v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
  [v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) © NVIDIA, CC-BY-4.0
- [GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3) © Sber, MIT
- Whisper © OpenAI, MIT for [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  and Apache-2.0 for the other sizes (such as
  [large-v3](https://huggingface.co/openai/whisper-large-v3)), as their model cards state
- Moonshine ([tiny](https://huggingface.co/UsefulSensors/moonshine-tiny),
  [base](https://huggingface.co/UsefulSensors/moonshine-base)) © Useful Sensors, MIT
- [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) © MediaTek Research,
  Apache-2.0

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
