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
