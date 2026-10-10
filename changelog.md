## v0.1.0 – unreleased

First release.

- Speech-to-text on the GPU through wgpu (Metal on Apple Silicon, Vulkan on Linux and Windows), in pure Rust
- Models: Parakeet TDT 0.6B v2 / v3, GigaAM v3 (four heads) and Whisper large-v3-turbo, with output matching transcribe.cpp on its test clips
- CLI: `wavo pull`, `list`, `rm`, `run` (wav, mp3, m4a, flac, ogg; text, JSON or SRT; audio of any length, split at pauses) and `bench`, sharing the Hugging Face cache with `hf`
