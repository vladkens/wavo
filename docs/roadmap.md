# Roadmap

Priority models: GigaAM v3 e2e-rnnt (Russian) and Parakeet TDT V2 (English). Work on one model
at a time: port it, match the reference exactly, make it fast, move on. Tick a phase when done.

- [x] **1. GigaAM v3 e2e-rnnt, correct.** Crate skeleton with the `Model` API and CLI, GGUF
      reader, CPU frontend, GPU encoder, CPU RNN-T greedy decoder. Weights stay Q8_0/F16 on the
      GPU. Fixtures `ru`, `ru-short`, `ru-long` match exactly; first benchmark against the
      reference recorded in `docs/perf.md`.
- [x] **2. GigaAM v3 e2e-rnnt, fast.** On Apple Silicon: warm faster than transcribe.cpp Metal on
      all three clips, load ≤ 0.25 s, small peak memory. Work through `docs/perf.md` → "Not tried
      yet", re-profiling with GPU timestamps after big changes. Decision point: if warm is still
      slower than the reference once that list is exhausted, stop and report to the user before
      porting more models.
- [x] **3. Other GigaAM v3 variants.** `e2e-ctc` (CTC head), then `rnnt` and `ctc` (33-char
      vocab).
- [x] **4. Parakeet TDT V2, correct.** `parakeet-tdt-0.6b-v2`: FastConformer, relative
      positional attention, TDT decoder. Fixtures `jfk`, `dots`, `jobs-silence` match exactly;
      first benchmark against the reference recorded in `docs/perf.md`.
- [x] **5. Parakeet TDT V2, fast.** Warm, first call, load and peak memory at least on par with
      transcribe.cpp, the same way as phase 2.
- [x] **6. Parakeet TDT V3.** `parakeet-tdt-0.6b-v3`: same architecture with a bigger vocab/joint.
      Fixtures `jfk`, `ru-short`, `uk-short`.
- [x] **7. Ollama-like CLI.** `wavo pull/list/rm` by short name in the Hugging Face cache,
      compatible with `hf`; `wavo run [MODEL] AUDIO` on wav, mp3, m4a, flac and ogg resampled to
      16 kHz, as text, `--json` or `--srt`, never downloading; `wavo bench` kept. The CLI is the
      default `cli` feature; with `--no-default-features` the library stays pure Rust with the
      same API. Long-audio time and memory measured, no chunking yet.
- [ ] **8. Linux and Windows.** Vulkan / DX12 through wgpu, tested on real machines.
- [x] **9. Long audio.** `wavo run` splits recordings of any length at pauses (frame energy, no
      VAD model) into segments of at most the model's window (`Model::max_audio_ms()`, GigaAM
      25 s) or a default for models without one, and joins the transcripts; `--segment` sets the
      length, `0` is one pass. Time linear in length, peak memory flat.
- [x] **10. Real recordings vs transcribe.cpp.** `make compare` (`examples/batch.rs` and the
      reference's batch mode, one model load each) on 2,923 of the person's dictation
      recordings: the same text in 99.9% (GigaAM) and 98.9% (Parakeet V3) of them, every
      difference from a near-tie; speed by length in `docs/perf.md` → "Real recordings".
- [x] **11. Whisper large-v3-turbo, correct.** Encoder and decoder on the GPU (F16 K/V caches,
      one pass per token), greedy search with segment timestamps and 30 s windows on the CPU, as
      transcribe.cpp runs it by default. Fixtures `jfk`, `zh-short`, `ru-long`, `jobs-silence`
      (text and segments) match exactly; first benchmark recorded in `docs/perf.md`.
- [x] **12. Whisper large-v3-turbo, fast.** Warm, first call, load and peak memory at least on par
      with transcribe.cpp on `jfk`, `ru-long` and a ~30 s clip; accuracy and memory over every
      sample the model's languages cover.
- [x] **13. Other Whisper variants.** `whisper-tiny`, `-base`, `-small`, `-medium`,
      `-large-v3`, the `.en` ones, `whisper-large`, `-large-v2` and `Breeze-ASR-25`: config only
      (80-mel conv padded to K = 256, `.en` prompts with SOT alone). CI tests `whisper-tiny`.

After Whisper: **other engines.** Measure wavo against the other ways to run the same models on a
Mac, not only transcribe.cpp. Plan: [docs/plans/20261010-other-engines.md](plans/20261010-other-engines.md).

**Fast build.** The own FFT replaced `rustfft`: a clean release build takes ~29 s (from ~40),
mostly the wgpu chain (naga → wgpu-core → wgpu), and the library rebuilds in ~2 s (from ~12).
Left: wgpu features per target. Plan: [docs/plans/20261010-fast-build.md](plans/20261010-fast-build.md).

Later, only on request: streaming, slowing fast speech down before recognition
(pitch-keeping time-stretch such as WSOLA; first measure WER on samples sped up with ffmpeg
`atempo`, with and without slowing them back down).
