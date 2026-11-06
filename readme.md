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

[Wavo](https://github.com/vladkens/wavo-app) is a real app built on this library: offline push-to-talk dictation for macOS, Windows and Linux. Install it to try wavo without a terminal, or read its code as an example for your own app.

## 🌟 Features

- 📏 Audio of any length: long recordings are split at pauses and joined back
- 🎧 Any common format: wav, mp3, m4a, flac, ogg (Vorbis), resampled for you
- ⚡ Runs on the GPU: Metal on Apple Silicon, Vulkan on Linux and Windows; Parakeet and GigaAM also on a Mac's Neural Engine
- 🦀 Pure Rust library: no CMake, C++ toolchain, Python or ONNX Runtime to install
- 📦 `pull` / `list` / `rm` models, shared with the Hugging Face cache that `hf` uses
- 📝 Plain text, JSON with a start time for every token, or SRT subtitles
- 🔌 `wavo serve`: a local server for apps built on OpenAI's transcription API

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

<details>
<summary>Server</summary>

`wavo serve` answers OpenAI's transcription API, so apps and scripts written for OpenAI's
`whisper-1` (the `openai` Python and JS clients, apps that let you set the API's URL) transcribe
locally instead:

```sh
wavo serve                                       # parakeet-ultra on http://127.0.0.1:8000
wavo serve whisper-turbo gigaam-v3 --port 9000   # these models, loaded up front
wavo serve --host 0.0.0.0 --api-key KEY          # on the network, with a key (or WAVO_API_KEY)
wavo serve parakeet-v3 --backend gpu             # the GPU even where the Neural Engine is ready

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

</details>

## 🧠 Models

| Model | Languages | Size | Errors, English | Errors, Russian | 1 hour of audio |
|---|---|---:|---:|---:|---:|
| `parakeet-ultra` (default) | 25 European | 0.74 GB | 1.8 | 4.7 | ~30 s |
| `gigaam-v3` | Russian | 0.27 GB | – | 4.5 | 11 s |
| `parakeet-v2` | English | 0.73 GB | 1.7 | – | ~35 s |
| `whisper-turbo` | 100 | 0.89 GB | 2.0 | 4.5 | ~4 min |
| `moonshine-base` | English | 0.08 GB | 3.3 | – | ~20 s |

Errors are wrong words per 100, on LibriSpeech test-clean and FLEURS Russian, measured with wavo. Time is on an Apple M5's GPU. [All 24 models](docs/models.md), [all the numbers](docs/benchmarks.md).

## ⚡ Speed

How long you wait for the text after a recording of that length ends. Lower is better. Parakeet v3 on a MacBook Pro with Apple M5.

| | 5 s | 15 s | 30 s | 1 hour |
|---|---:|---:|---:|---:|
| wavo, Neural Engine | 0.04 s | 0.06 s | 0.15 s | 18 s |
| FluidAudio, Neural Engine | 0.05 s | 0.08 s | 0.14 s | 21 s |
| wavo, GPU | 0.04 s | 0.09 s | 0.15 s | 24 s |
| transcribe.cpp, GPU | 0.06 s | 0.14 s | 0.26 s | 35 s |
| parakeet.cpp, GPU | 0.07 s | 0.17 s | 0.31 s | 39 s |
| parakeet-mlx, GPU | 0.12 s | 0.25 s | 0.39 s | 61 s |
| sherpa-onnx, CPU | 0.12 s | 0.34 s | 0.73 s | 146 s |

GigaAM, other machines and how it's measured: [docs/benchmarks.md](docs/benchmarks.md).

## 📚 Library usage

```sh
cargo add wavo --no-default-features --features hub
```

```rust
let model = wavo::Model::load(path)?;
let transcript = model.transcribe(&pcm)?; // pcm: 16 kHz mono f32 in [-1, 1]
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
