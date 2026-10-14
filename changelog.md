## v0.3.1 – 2026-10-14

### Features

- Added `Model::device()`, which tells which API (Metal, Vulkan, DX12) and which GPU a loaded model runs on; `wavo bench` prints it (#49)
- Added a DirectX 12 fallback on Windows machines without a Vulkan driver, such as virtual machines; it shipped in 0.3.0 but was missing from its notes (#46)

### Fixes

- Fixed a crash at the first transcription on software renderers such as WARP in Windows virtual machines (#48)

**Full Changelog**: https://github.com/vladkens/wavo/compare/v0.3.0...v0.3.1

---

## v0.3.0 – 2026-10-14

### Features

- Added a fast path for Apple M5 and newer GPUs that runs matrix math on their neural accelerators through the Metal 4 tensor API, about twice as fast as before on GigaAM, Parakeet and Whisper (#43)
- `wavo rm` and `Hub::remove` now also delete models that `hf` keeps in its shared blob store, once no other model uses the file (#41)

### Improvements

- Improved Whisper speed on NVIDIA GPUs about 3× and Moonshine's speed and memory everywhere with flash attention (#36)
- Improved Whisper decoding speed on NVIDIA GPUs (#38)
- Improved the first transcription after loading a model on discrete GPUs, e.g. Parakeet on an RTX 4060 from ~100 to ~77 ms (#44)
- Improved speed on integrated and other GPUs without fast kernels: Parakeet about 1.6–2× faster on an Intel N100 (#45)

### Docs

- Updated the readme with per-machine speed numbers (Apple M2 and M5, RTX 4060, Intel N100), how wavo is made and tested, and credits (#42)

**Full Changelog**: https://github.com/vladkens/wavo/compare/v0.2.0...v0.3.0

---

## v0.2.0 – 2026-10-12

### Breaking Changes

- Marked the library's `Transcript`, `Token` and `Error` as non-exhaustive, so code outside wavo can no longer build them with struct literals or match them without a wildcard (#21)

### Features

- Added `wavo AUDIO` as a shortcut for `wavo run AUDIO`; `wavo run` now offers to download a missing model, and `-y` downloads it without asking (#29)
- Added Moonshine tiny and base models for fast English transcription (#30)
- Added the other Whisper models: tiny, base, small, medium and their English-only versions, large, large-v2, large-v3 and Breeze-ASR-25 for Taiwanese Mandarin (#20)
- Added model downloads to the library: `wavo::hub` behind the `hub` feature finds, downloads and removes models in the Hugging Face cache (#21)
- Added downloading several models at once with `wavo pull` (#17)
- Added progress bars for `wavo pull` and `wavo run` (#18)
- Added prebuilt binaries for macOS, Linux and Windows on GitHub releases, and builds of the latest code in the dev release (#9, #32)

### Improvements

- Improved speed on NVIDIA GPUs by 10–40% with tensor cores (#28)
- Improved model load time by about half, so dictation with a model loaded per recording finishes sooner (#24)
- Improved downloads to retry and resume after a dropped connection (#19)

**Full Changelog**: https://github.com/vladkens/wavo/compare/v0.1.0...v0.2.0

---

## v0.1.0 – 2026-10-10

### Features

- Added local speech-to-text with Parakeet TDT v2 and v3, plus GigaAM v3 E2E-RNNT, E2E-CTC, RNNT and CTC models.
- Added Whisper large-v3-turbo for multilingual transcription with segment timestamps ([#8](https://github.com/vladkens/wavo/pull/8)).
- Added GPU acceleration through Metal on Apple Silicon and Vulkan on Linux and Windows ([#5](https://github.com/vladkens/wavo/pull/5), [#7](https://github.com/vladkens/wavo/pull/7)).
- Added a pure Rust library for loading models and transcribing audio into text and timestamped tokens, with optional CLI dependencies ([#2](https://github.com/vladkens/wavo/pull/2)).
- Added `pull`, `list` and `rm` commands to manage models in the shared Hugging Face cache, plus `run` for offline transcription and `bench` to measure speed ([#2](https://github.com/vladkens/wavo/pull/2)).
- Added WAV, MP3, M4A/AAC, FLAC and Ogg/Vorbis input with automatic downmixing and resampling ([#2](https://github.com/vladkens/wavo/pull/2)).
- Added plain text, timestamped JSON and SRT subtitle output, with one subtitle cue per Whisper segment ([#2](https://github.com/vladkens/wavo/pull/2), [#10](https://github.com/vladkens/wavo/pull/10)).
- Added automatic splitting of long recordings at pauses, with `--segment` to control or disable splitting ([#3](https://github.com/vladkens/wavo/pull/3)).

**Full Changelog**: https://github.com/vladkens/wavo/commits/v0.1.0
