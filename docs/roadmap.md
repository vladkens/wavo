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
- [ ] **4. Parakeet TDT V2.** FastConformer, relative positional attention, TDT decoder. Fixtures
      `jfk`, `dots`, `jobs-silence`. Then make it fast the same way.
- [ ] **5. Parakeet TDT V3.** Same architecture with a bigger vocab/joint. Fixtures `jfk`,
      `ru-short`, `uk-short`.

Later, only on request: Whisper, streaming, platforms beyond a working wgpu fallback.
