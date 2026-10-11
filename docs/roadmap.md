# Roadmap

Priority models: GigaAM v3 e2e-rnnt (Russian) and Parakeet TDT V2 (English). Work on one model
at a time: port it, match the reference exactly, make it fast, move on. Tick a phase when done.

- [x] **1. GigaAM v3 e2e-rnnt, correct.** [Plan](plans/completed/20261009-gigaam-e2e-rnnt.md)
- [x] **2. GigaAM v3 e2e-rnnt, fast.** [Plan](plans/completed/20261009-gigaam-fast.md)
- [x] **3. Other GigaAM v3 variants:** `e2e-ctc`, `rnnt`, `ctc`. [Plan](plans/completed/20261009-gigaam-variants.md)
- [x] **4. Parakeet TDT V2, correct.** [Plan](plans/completed/20261009-parakeet-tdt-v2.md)
- [x] **5. Parakeet TDT V2, fast.** [Plan](plans/completed/20261010-parakeet-fast.md)
- [x] **6. Parakeet TDT V3.** [Plan](plans/completed/20261010-parakeet-tdt-v3.md)
- [x] **7. Ollama-like CLI** (#2). [Plan](plans/completed/20261010-cli.md)
- [x] **8. Linux and Windows** through Vulkan, working and tested (#5, #7, #12).
      [Linux](plans/20261010-linux.md), [Windows](plans/completed/20261010-windows.md)
- [x] **9. Long audio,** split at pauses (#3). [Plan](plans/completed/20261010-long-audio.md)
- [x] **10. Real recordings vs transcribe.cpp** (#6). [Plan](plans/completed/20261010-real-recordings.md)
- [x] **11. Whisper large-v3-turbo, correct** (#8). [Plan](plans/completed/20261010-whisper.md)
- [x] **12. Whisper large-v3-turbo, fast** (#8).
- [x] **13. Other Whisper variants** and Breeze-ASR-25 (#20).
- [x] **14. Model downloads in the library:** `wavo::hub` behind the `hub` feature (#21).
- [x] **15. Moonshine tiny and base, correct.** [Plan](plans/20261011-moonshine.md)

Open:

- **Moonshine:** fast (phase 16), the language fine-tunes (17), the streaming variants one-shot
  (18, optional). [Plan](plans/20261011-moonshine.md)
- **Fast kernels beyond Apple.** The GEMM on F16 16×16 cooperative matrices runs on NVIDIA
  (#28, [plan](plans/completed/20261011-nvidia-fast.md)): 10–40% faster on an RTX 4060, Parakeet ahead of
  transcribe.cpp, Whisper still 2–3× behind. Open: attention on 16×16 matrices, the tier on AMD
  and Intel Arc, and the Linux plan's portable-encoder items.
- **Fast build.** The in-house FFT replaced `rustfft` (#15); left: wgpu features per target.
  [Plan](plans/20261010-fast-build.md)
- **Other engines.** Measure wavo against the other ways to run the same models on a Mac, not
  only transcribe.cpp. [Plan](plans/20261010-other-engines.md)
- **GPU headroom on Parakeet.** Core ML on the same M2 GPU takes 8–15% less time end to end
  (FluidAudio GPU 134 vs wavo 145 ms on jfk, 372 vs 437 ms on dots). Profile our encoder against
  it and close the gap.
- **Neural Engine backend** (opt-in, macOS) for Parakeet, then GigaAM; not scheduled.
  [Plan](plans/20261011-ane.md)

Later research: **accuracy against the originals.** Plan: [docs/plans/20261011-accuracy.md](plans/20261011-accuracy.md).

Later, only on request: streaming, slowing fast speech down before recognition
(pitch-keeping time-stretch such as WSOLA; first measure WER on samples sped up with ffmpeg
`atempo`, with and without slowing them back down).
