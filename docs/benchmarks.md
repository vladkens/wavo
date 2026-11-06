# Benchmarks

Seconds you wait for the text after a recording of that length ends. Lower is better, the best in bold. MacBook Pro with Apple M5 and 32 GB, macOS 27.

## Parakeet v3, English

| Engine | Runs on | 5s | 15s | 30s | 1h |
|---|---|---:|---:|---:|---:|
| wavo 0.4.0 | Neural Engine | **0.04** | **0.06** | 0.15 | **18** |
| FluidAudio 0.17.7 | Neural Engine | 0.05 | 0.08 | **0.14** | 21 |
| wavo 0.4.0 | GPU | **0.04** | 0.09 | 0.15 | 24 |
| transcribe.cpp 0.4.1 | GPU | 0.06 | 0.14 | 0.26 | 35 |
| parakeet.cpp 0.6.1 | GPU | 0.07 | 0.17 | 0.31 | 39 |
| parakeet-mlx 0.7.0 | GPU | 0.12 | 0.25 | 0.39 | 61 |
| sherpa-onnx 2.0.1 | CPU | 0.12 | 0.34 | 0.73 | 146 |

## GigaAM v3, Russian

| Engine | Runs on | 5s | 15s | 30s | 1h |
|---|---|---:|---:|---:|---:|
| wavo 0.4.0 | GPU | **0.02** | **0.04** | **0.08** | **11** |
| wavo 0.4.0 | Neural Engine | 0.03 | **0.04** | 0.10 | 13 |
| transcribe.cpp 0.4.1 | GPU | **0.02** | 0.05 | 0.12 | 14 |
| sherpa-onnx 2.0.1 | CPU | 0.07 | 0.22 | 0.73 | 95 |

## Other machines

| wavo 0.4.0 | Model | 5s | 15s | 30s | 1h |
|---|---|---:|---:|---:|---:|
| NVIDIA RTX 4060, Windows 11 | Parakeet v3 | 0.05 | 0.10 | 0.16 | 29 |
| NVIDIA RTX 4060, Windows 11 | GigaAM v3 | **0.02** | **0.04** | **0.06** | **13** |

## How it's measured

- Same model and same audio for every engine, one at a time, on one machine, with the model loaded once.
- 5, 15 and 30 s: one clip, 6 calls in a row, the median of the last 5.
- 1 hour: the recording cut at pauses into 2–15 s pieces, the same pieces for every engine, the whole process with the model load, the median of 3 rounds.
- Engines store weights differently: wavo and transcribe.cpp 8-bit (Q8_0), FluidAudio and sherpa-onnx int8, parakeet-mlx bf16. transcribe.cpp's time is the sum of its own stage timers.

Rerun it with `scripts/speed.py run`. The raw numbers are in [results/](../results).
