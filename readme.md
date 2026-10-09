# wavo

Speech-to-text in pure Rust. Runs GGUF ASR models (GigaAM v3, Parakeet TDT) on the GPU through
`wgpu` with custom WGSL kernels — no C/C++, ONNX or BLAS. The goal: a plain `cargo build` and a
small engine that is at least as fast as transcribe.cpp on the same model files.

Early development: nothing runs yet. Plan: [docs/roadmap.md](docs/roadmap.md). Built by coding
agents; rules are in [agents.md](agents.md).
