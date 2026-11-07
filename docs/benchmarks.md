# Benchmarks

How fast each engine gives you the text. Every engine runs the same model on the same audio, one at a time, on one machine.

## Parakeet v3, English

| Engine | On | 5s phrase | Long audio | Energy |
|---|---|---:|---:|---:|
| wavo 0.4.0 | ANE | 0.12s | 200× | 225J |
| FluidAudio 0.17.7 | ANE | 0.13s | 175× | 164J |
| wavo 0.4.0 | GPU | 0.14s | 160× | 377J |
| transcribe.cpp 0.4.1 | GPU | 0.12s | 107× | 565J |
| parakeet.cpp 0.6.1 | GPU | 0.77s | 90× | 472J |
| parakeet-mlx 0.7.0 | GPU | 0.63s | 61× | 393J |
| sherpa-onnx 2.0.1 | CPU | 0.83s | 39× | 1474J |

## GigaAM v3, Russian

| Engine | On | 5s phrase | Long audio | Energy |
|---|---|---:|---:|---:|
| wavo 0.4.0 | GPU | 0.10s | 330× | 156J |
| wavo 0.4.0 | ANE | 0.08s | 287× | 83J |
| transcribe.cpp 0.4.1 | GPU | 0.07s | 262× | 221J |
| sherpa-onnx 2.0.1 | CPU | 0.48s | 59× | 992J |

## Other machines

wavo on an NVIDIA RTX 4060 under Windows 11: Parakeet v3 runs at 124×, GigaAM v3 at 291×.

## How it's measured

- 5s phrase: the whole cycle of a short recording. A new process starts, loads the model, transcribes a 5s clip and exits. The time is the median of 5 runs, with the model file already in the disk cache.
- Long audio: how many times faster than real time. An hour-long recording is cut at pauses into pieces of 2–15s, the same pieces for every engine, and goes through one process, model load included. The number is the median of 3 rounds.
- Energy: joules for that hour on the CPU, GPU and Neural Engine, read with [macmon](https://github.com/vladkens/macmon) every 200ms, minus the idle power of the 15s before the run, with nothing else running. The number is the median of 3 rounds. Good for comparing engines on one Mac, not as an absolute number. Memory isn't counted: macmon has no reading for it on this chip.

Engines store weights differently: wavo and transcribe.cpp in 8-bit Q8_0, FluidAudio and sherpa-onnx in int8, parakeet-mlx in bf16.

ANE is Apple's Neural Engine. Measured on a MacBook Pro with an Apple M5 and 32GB, macOS 27.

Rerun it with `scripts/speed.py run`. The raw numbers, with clips of 5s, 15s and 30s, are in [results/](../results).
