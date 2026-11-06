# Benchmarks

How long you wait for the text after a recording of that length ends. Lower is better. MacBook Pro with Apple M5 and 32 GB, macOS 27.

## Parakeet v3, English

| Engine | Runs on | 5 s | 15 s | 30 s | 1 hour |
|---|---|---:|---:|---:|---:|
| wavo 0.4.0 | Neural Engine | 0.04 s | 0.06 s | 0.15 s | 18 s |
| FluidAudio 0.17.7 | Neural Engine | 0.05 s | 0.08 s | 0.14 s | 21 s |
| wavo 0.4.0 | GPU | 0.04 s | 0.09 s | 0.15 s | 24 s |
| transcribe.cpp 0.4.1 | GPU | 0.06 s | 0.14 s | 0.26 s | 35 s |
| parakeet.cpp 0.6.1 | GPU | 0.07 s | 0.17 s | 0.31 s | 39 s |
| parakeet-mlx 0.7.0 | GPU | 0.12 s | 0.25 s | 0.39 s | 61 s |
| sherpa-onnx 2.0.1 | CPU | 0.12 s | 0.34 s | 0.73 s | 146 s |

## GigaAM v3, Russian

| Engine | Runs on | 5 s | 15 s | 30 s | 1 hour |
|---|---|---:|---:|---:|---:|
| wavo 0.4.0 | GPU | 0.02 s | 0.04 s | 0.08 s | 11 s |
| wavo 0.4.0 | Neural Engine | 0.03 s | 0.04 s | 0.10 s | 13 s |
| transcribe.cpp 0.4.1 | GPU | 0.02 s | 0.05 s | 0.12 s | 14 s |
| sherpa-onnx 2.0.1 | CPU | 0.07 s | 0.22 s | 0.73 s | 95 s |

## Other machines

| wavo 0.4.0 | Model | 5 s | 15 s | 30 s | 1 hour |
|---|---|---:|---:|---:|---:|
| NVIDIA RTX 4060, Windows 11 | Parakeet v3 | 0.05 s | 0.10 s | 0.16 s | 29 s |
| NVIDIA RTX 4060, Windows 11 | GigaAM v3 | 0.02 s | 0.04 s | 0.06 s | 13 s |

## How it's measured

- Same model and same audio for every engine, one at a time, on one machine, with the model loaded once.
- 5, 15 and 30 s: one clip, 6 calls in a row, the median of the last 5.
- 1 hour: the recording cut at pauses into 2–15 s pieces, the same pieces for every engine, the whole process with the model load, the median of 3 rounds.
- Engines store weights differently: wavo and transcribe.cpp 8-bit (Q8_0), FluidAudio and sherpa-onnx int8, parakeet-mlx bf16. transcribe.cpp's time is the sum of its own stage timers.

Rerun it with `scripts/speed.py run`. The raw numbers are in [results/](../results).
