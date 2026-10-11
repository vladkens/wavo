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

