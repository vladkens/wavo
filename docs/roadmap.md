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
- [ ] **9. Long audio.** `wavo run` splits recordings of any length at pauses (frame energy, no
      VAD model) into segments of at most the model's window (`Model::max_audio_ms()`, GigaAM
      25 s) or a default for models without one, and joins the transcripts; `--segment` sets the
      length, `0` is one pass. Time linear in length, peak memory flat.

Later, only on request: Whisper, streaming, slowing fast speech down before recognition
(pitch-keeping time-stretch such as WSOLA; first measure WER on samples sped up with ffmpeg
`atempo`, with and without slowing them back down).
