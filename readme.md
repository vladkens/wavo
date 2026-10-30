<div align="center">

<img src="docs/logo.png" alt="wavo" width="600" />

**Local speech-to-text at GPU speed**

[<img src="https://badges.ws/crates/v/wavo" alt="version" />](https://crates.io/crates/wavo)
[<img src="https://badges.ws/crates/docs/wavo" alt="docs" />](https://docs.rs/wavo)
[<img src="https://badges.ws/github/license/vladkens/wavo" alt="license" />](https://github.com/vladkens/wavo/blob/main/LICENSE)
[<img src="https://badges.ws/badge/follow-@uid127-black" alt="follow @uid127" />](https://x.com/uid127)

</div>

`wavo` turns speech into text locally, on the GPU or a Mac's Neural Engine, in pure Rust. Give it a
recording of any length, a voice note or a two-hour meeting, and get the text with timestamps. It
runs the open ASR models that [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
publishes. Use it as a command-line tool with an ollama-like model manager, or as a library that
builds with a plain `cargo build`.

## 🌟 Features

- 📏 Audio of any length: long recordings are split at pauses and joined back, no chunking on
  your side
- 🎧 Any common format: wav, mp3, m4a, flac, ogg (Vorbis), resampled for you
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- ⚡ Runs on the GPU: Metal on Apple Silicon, Vulkan on Linux and Windows (DX12 without Vulkan)
- 🧠 On Apple Silicon Macs, Parakeet and GigaAM run on the Neural Engine, about 2× faster than the
  GPU on 11 s of audio at a sixth of its energy, prepared once by `wavo pull`
- 🏎️ Fast: an hour of audio in under a minute on an M2 with Parakeet or GigaAM (see
  [Speed](#-speed)), and a model loads in about 0.1 s
- 📦 `pull` / `list` / `rm` models, shared with the Hugging Face cache that `hf` uses
- 📝 Plain text, JSON with a start time for every token, or SRT subtitles
- 🔌 `wavo serve`: a local server for apps built on OpenAI's transcription API
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
| `parakeet-v3` | 0.74 GB | 25 European languages | cased, punctuated |
| `parakeet-v2` | 0.73 GB | English | cased, punctuated |
| `parakeet-ultra` (default) | 0.74 GB | 25 European languages | cased, punctuated |
| `parakeet-he` | 0.74 GB | Hebrew | cased, punctuated |
| `gigaam-v3` | 0.27 GB | Russian | cased, punctuated |
| `gigaam-v3-e2e-ctc` | 0.27 GB | Russian | cased, punctuated |
| `gigaam-v3-rnnt` | 0.27 GB | Russian | lowercase, no punctuation |
| `gigaam-v3-ctc` | 0.27 GB | Russian | lowercase, no punctuation |
| `whisper-turbo` | 0.89 GB | 100 languages | cased, punctuated |
| `whisper-turbo-he` | 0.89 GB | Hebrew | cased, punctuated |
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
[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) is Moondream's retraining of v3
with fewer errors in each of its 25 languages,
GigaAM is Sber's [GigaAM v3](https://github.com/salute-developers/GigaAM), Whisper is OpenAI's
[Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), Breeze-ASR-25 is MediaTek
Research's [Whisper large-v2 fine-tune](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)
and Moonshine is Useful Sensors' [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny),
small and fast. Moonshine has no timestamps: its tokens all start at 0, and `--srt` gives one cue
per piece of the split audio. The Hebrew models are ivrit.ai's
[Whisper turbo fine-tune](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) and Yakov
Kolani's [Parakeet v3 fine-tune](https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he).
Short names stand for the published ones: `parakeet-v3` is `parakeet-tdt-0.6b-v3`, `parakeet-v2`
is `parakeet-tdt-0.6b-v2`, `parakeet-he` is `parakeet-tdt-0.6b-v3-he`, `gigaam-v3` is
`gigaam-v3-e2e-rnnt`, `whisper-turbo` is `whisper-large-v3-turbo`, `whisper-turbo-he` is
`ivrit-whisper-large-v3-turbo` and `breeze-asr-25` is `Breeze-ASR-25`; full names and a path to a
`.gguf` work too.

`parakeet-ultra` is the default: it makes fewer errors than `parakeet-v3` on every benchmark
group of [Moondream's model card](https://huggingface.co/moondream/parakeet-ultra#benchmarks),
which also lists each language. Word error rate, %, lower is better, measured by Moondream on an
NVIDIA GPU, not with wavo:

| Benchmark | `parakeet-v3` | `parakeet-ultra` |
|---|---:|---:|
| English, 7 sets of the Open ASR Leaderboard | 6.26 | 5.80 |
| FLEURS, 25 languages | 11.62 | 9.55 |
| Business speech (AMI, VoxPopuli, Earnings-22) | 6.15 | 5.79 |
| Background noise, 9 MUSAN conditions | 6.72 | 5.82 |
| Long talks, TED-LIUM 3 | 2.71 | 1.94 |

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

When the model isn't downloaded yet, `wavo run` asks before downloading it, in a terminal; `-y`
downloads without asking, for scripts. Without a terminal and `-y` it fails with the `wavo pull`
command to run, and `wavo bench` never downloads. Long recordings are split at their quietest
moments and transcribed piece by piece, so an hour of audio works as well as a minute;
`--segment SECS` sets the piece length, `--segment 0` runs one pass (up to 37.5 s for
GigaAM and 45 s for Moonshine). Whisper detects the language of each piece; `--language CODE`
sets it instead, for speech it mistakes for a close language, such as Hebrew heard as Arabic.
Audio without recognized speech prints nothing (with `--json`, an empty text) and
`no speech recognized` on stderr, and still exits with 0, so a batch goes on. Models are stored in
the Hugging Face cache (`~/.cache/huggingface/hub`), so a model fetched with `hf download` is
found by `wavo` and the other way round.

On an Apple Silicon Mac with macOS 15 or later, `wavo pull` also prepares Parakeet and GigaAM for
the Neural Engine: it compiles the model's encoder for Core ML once, which takes 15–30 s and as much
disk as the model, in `~/Library/Caches/wavo`. From then on `wavo run` uses the Neural Engine for
them (`wavo list` shows which are ready) and splits audio into pieces of up to 15 s. Its text is
close to the GPU's, not equal ([numbers](https://docs.rs/wavo/latest/wavo/#neural-engine)).
`--backend gpu` runs on the GPU anyway, `--backend ane` fails rather than fall back. Whisper and Moonshine always run on the GPU,
where they are faster.

### Server

`wavo serve` answers OpenAI's transcription API, so apps and scripts written for OpenAI's
`whisper-1` (the `openai` Python and JS clients, apps that let you set the API's URL) transcribe
locally instead:

```sh
wavo serve                                       # parakeet-ultra on http://127.0.0.1:8000
wavo serve whisper-turbo gigaam-v3 --port 9000   # these models, loaded up front
wavo serve --host 0.0.0.0 --api-key KEY          # on the network, with a key (or WAVO_API_KEY)

curl -F file=@talk.mp3 -F model=whisper-1 http://127.0.0.1:8000/v1/audio/transcriptions
curl -F file=@talk.mp3 -F model=gigaam-v3 -F response_format=srt http://127.0.0.1:8000/v1/audio/transcriptions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="KEY")  # any key without --api-key
with open("talk.mp3", "rb") as f:
  print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

`POST /v1/audio/transcriptions` takes the form fields `file`, `model`, `language` and
`response_format`: `json` (`{"text": ...}`), `text`, `srt`, `vtt` or `verbose_json` (segments
with start and end times, and the words for models with token times: not Whisper or Moonshine).
A `model` that names no wavo model, such as `whisper-1`, goes to the first model given; another
downloaded model loads on its first request, replacing the one loaded that way before. Long audio
is split at pauses as in `wavo run`. `GET /v1/models` lists the downloaded models and `GET /health`
the GPU and the loaded models. `serve` downloads the models it is started with, after asking or
with `-y`; requests never download. It serves 16 requests at once, each up to 512 MB and 3 hours
of audio, and answers more with 503.

For [Bazarr](https://www.bazarr.media)'s subtitles, `POST /asr` answers as
[whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) does: enable Bazarr's
Whisper provider and set its endpoint to `http://HOST:8000`, with `wavo serve` started without
`--api-key`, as Bazarr sends none; its Test Connection shows wavo's version (`GET /status`). The
first model transcribes, in the language of the audio track's tag, which it must know; wavo
doesn't translate, nor detect the language of an untagged track (`/detect-language`).

## ⚡ Speed

Time to transcribe one minute of audio and the real-time factor (RTF: processing time ÷ audio
length); lower is better. Measured with `wavo run` on an hour-long
[LibriVox reading](https://archive.org/details/memoirsofsherlockholmesv3_1502_librivox), model
load and mp3 decoding included, on an **Apple M2** (10-core GPU), 2026-10-21 (`parakeet-ultra`
2026-10-23).

| Model | 1 min of audio | RTF |
|---|---:|---:|
| `gigaam-v3` | 0.51 s | 0.0085 |
| `parakeet-v3` | 0.88 s | 0.0146 |
| `parakeet-ultra` | 0.87 s | 0.0144 |
| `parakeet-v2` | 0.82 s | 0.0136 |
| `whisper-turbo` | 3.32 s | 0.0553 |
| `whisper-tiny` | 0.36 s | 0.0061 |
| `moonshine-tiny` | 0.33 s | 0.0055 |
| `moonshine-base` | 0.50 s | 0.0083 |

On the Neural Engine, warm ms for an 11 s clip / a 4.5 s one, against wavo on the GPU and
transcribe.cpp on Metal:

| Model | Mac | Neural Engine | GPU | transcribe.cpp |
|---|---|---|---|---|
| `parakeet-v3` | M1 | **70** / **53** | 252 / 115 | 295 / 142 |
| `parakeet-v3` | M2 | **61** / **46** | 151 / 73 | 269 / 96 |
| `gigaam-v3` | M1 | **52** / **42** | 170 / 70 | 180 / 74 |
| `gigaam-v3` | M2 | **51** / **39** | 92 / 42 | 105 / 53 |

A load takes 0.06–0.2 s, and the wavo process peaks at 25–60 MiB against 300–760 MiB on the GPU:
the weights live in the system's Neural Engine service. A call takes 0.15–0.25 J on the Neural
Engine against 0.9–1.4 J on the GPU (M2, the accelerator's own power).

## 📚 Library usage

```sh
cargo add wavo --no-default-features --features hub
```

```rust
use std::ops::ControlFlow;
use wavo::hub::Hub;

let hub = Hub::new(); // the Hugging Face cache, shared with `hf` and `wavo pull`
let path = match hub.find("parakeet-ultra") {
  Some(path) => path,
  None => hub.pull("parakeet-ultra", |p| {
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
files you fetch yourself, with a pure Rust build. Decoding and resampling audio are up to you.
`model.transcribe` runs the audio in one pass, up to 1.5 times the length the model was trained on
for GigaAM and Moonshine (`model.max_audio_ms()`: 25 s for GigaAM, 30 s for Whisper and Moonshine);
`model.transcribe_long` takes recordings of any length, split at pauses into pieces of up to that
length, as `wavo run` does, with a callback after each piece for progress and cancel. On the Neural
Engine the window is 15 s, and `model.transcribe` splits longer audio there itself. On a Mac,
`hub.pull` also prepares Parakeet and GigaAM for the Neural Engine, `wavo::Model::prepare` does it
for a file you manage, and `Model::load` then runs them there; your app prepares for itself even
when `wavo pull` already did ([why](https://docs.rs/wavo/latest/wavo/#neural-engine)).
`Model::load_with(path, Backend::Gpu)` picks the GPU. The [API docs](https://docs.rs/wavo) have the
details, including downloading on a thread with progress and cancel, as
[examples/hub.rs](examples/hub.rs) does.

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
[handy-computer](https://huggingface.co/handy-computer) publishes on Hugging Face, and
[vladkens](https://huggingface.co/vladkens) for the Hebrew models, under the original licenses:

- Parakeet TDT ([v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
  [v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) © NVIDIA, CC-BY-4.0
- [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) © Moondream, CC-BY-4.0
- [GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3) © Sber, MIT
- Whisper © OpenAI, MIT for [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  and Apache-2.0 for the other sizes (such as
  [large-v3](https://huggingface.co/openai/whisper-large-v3)), as their model cards state
- Moonshine ([tiny](https://huggingface.co/UsefulSensors/moonshine-tiny),
  [base](https://huggingface.co/UsefulSensors/moonshine-base)) © Useful Sensors, MIT
- [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) © MediaTek Research,
  Apache-2.0
- [ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) ©
  ivrit.ai, Apache-2.0
- [parakeet-tdt-0.6b-v3-he](https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he) © Yakov
  Kolani, CC-BY-4.0

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
