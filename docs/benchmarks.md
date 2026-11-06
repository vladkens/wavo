# Benchmarks

How long each engine takes to give you the text, and how much energy that uses. Every engine runs the same model on the same audio. Times are in seconds, energy in joules, lower is better, the best in bold.

A short phrase is instant everywhere, so the 1-hour column is the one that shows a difference.

## Parakeet v3, English

| Engine | On | 5s | 15s | 30s | 1h | Energy |
|---|---|---:|---:|---:|---:|---:|
| wavo 0.4.0 | ANE | **0.04** | **0.06** | 0.15 | **18** | 225 |
| FluidAudio 0.17.7 | ANE | 0.05 | 0.08 | **0.14** | 21 | **164** |
| wavo 0.4.0 | GPU | **0.04** | 0.09 | 0.15 | 22 | 377 |
| transcribe.cpp 0.4.1 | GPU | 0.06 | 0.14 | 0.26 | 34 | 565 |
| parakeet.cpp 0.6.1 | GPU | 0.07 | 0.17 | 0.31 | 40 | 472 |
| parakeet-mlx 0.7.0 | GPU | 0.12 | 0.25 | 0.39 | 59 | 393 |
| sherpa-onnx 2.0.1 | CPU | 0.12 | 0.34 | 0.73 | 92 | 1474 |

## GigaAM v3, Russian

| Engine | On | 5s | 15s | 30s | 1h | Energy |
|---|---|---:|---:|---:|---:|---:|
| wavo 0.4.0 | GPU | **0.02** | **0.04** | **0.08** | **11** | 156 |
| wavo 0.4.0 | ANE | 0.03 | **0.04** | 0.10 | 13 | **83** |
| transcribe.cpp 0.4.1 | GPU | **0.02** | 0.05 | 0.12 | 14 | 221 |
| sherpa-onnx 2.0.1 | CPU | 0.07 | 0.22 | 0.73 | 63 | 992 |

## Other machines

| wavo 0.4.0 | Model | 5s | 15s | 30s | 1h |
|---|---|---:|---:|---:|---:|
| NVIDIA RTX 4060, Windows 11 | Parakeet v3 | 0.05 | 0.10 | 0.16 | 29 |
| NVIDIA RTX 4060, Windows 11 | GigaAM v3 | 0.02 | 0.04 | 0.06 | 13 |

## How it's measured

- Same model and same audio for every engine, one at a time, on one machine, with the model loaded once.
- 5, 15 and 30 s: one clip, 6 calls in a row, the median of the last 5.
- 1 hour: the recording cut at pauses into 2–15 s pieces, the same pieces for every engine, the whole process with the model load, the median of 3 rounds.
- Energy: the CPU, GPU and Neural Engine power from [macmon](https://github.com/vladkens/macmon) over the 1-hour run, minus the idle power of the 15 s before it, nothing else running, the median of 3 rounds. Good for comparing engines on one Mac, not as an absolute number; memory isn't counted, macmon has no reading for it on this chip.
- Engines store weights differently: wavo and transcribe.cpp 8-bit (Q8_0), FluidAudio and sherpa-onnx int8, parakeet-mlx bf16. transcribe.cpp's time is the sum of its own stage timers.

Measured on a MacBook Pro with an Apple M5 and 32 GB, macOS 27. ANE is Apple's Neural Engine. Rerun it with `scripts/speed.py run`. The raw numbers are in [results/](../results).
