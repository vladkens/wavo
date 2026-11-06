# Benchmarks

Generated numbers from `results/`: rerun them with `scripts/wer.py` and `scripts/speed.py`. Numbers depend on the machine and on heat: compare rows of one table, measured in the same rounds, not numbers across machines.

## Speed

MacBook Pro, Apple M5, 32 GB, macOS 27.0.1, 2026-11-06. `scripts/speed.py run`: same model, same audio, one engine at a time.

- A 1-hour recording: the recording cut at pauses into 2–15 s pieces, the same list for every engine, in one process with the model load. Median of 3 rounds with the engines alternating.
- A phrase: a clip of that length, listed 6 times in one process. Warm is the median of calls 2–6, first is call 1, load is the model's.
- Engines differ in quantization, shown in each row. transcribe.cpp's call time is the sum of its stage timers, a few ms below the wall time of a call. Peak memory isn't comparable across engines: weights on the Neural Engine live in a system service, and transcribe.cpp's aren't counted either.

### Parakeet v3, English

| Engine | Version | Backend | Precision | 1 hour, s | Words | Load, ms | 1 s | 2 s | 5 s | 10 s | 30 s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wavo-gpu | 0.4.0  | Metal | Q8_0 | 24.2 | 9677 | 104 | 30 | 32 | 43 | 46 | 155 |
| wavo-ane | 0.4.0  | Neural Engine | int8 encoder | 18.3 | 9675 | 64 | 30 | 32 | 40 | 39 | 145 |
| transcribe.cpp | v0.4.1 | Metal | Q8_0 | 35.0 | 9677 | 47 | 21 | 39 | 60 | 76 | 260 |
| fluidaudio | v0.17.7 | Neural Engine | int8 encoder | 20.9 | 9701 | 85 | 32 | 35 | 47 | 88 | 145 |
| parakeet-mlx | 0.7.0 mlx 0.32.4 | Metal (MLX) | bf16 | 61.2 | 9689 | 299 | 69 | 82 | 123 | 148 | 394 |
| sherpa-onnx | 2.0.1 4 threads | CPU | int8 | 146.0 | 9648 | 636 | 35 | 54 | 122 | 224 | 733 |
| parakeet.cpp | v0.6.1 | Metal | q8_0 | 39.4 | 10083 | 218 | 38 | 47 | 74 | 91 | 312 |

Warm time per call in ms.

### GigaAM v3, Russian

| Engine | Version | Backend | Precision | 1 hour, s | Words | Load, ms | 1 s | 2 s | 5 s | 10 s | 30 s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wavo-gpu | 0.4.0  | Metal | Q8_0 | 11.4 | 7644 | 75 | 13 | 14 | 18 | 28 | 80 |
| wavo-ane | 0.4.0  | Neural Engine | int8 encoder | 12.7 | 7649 | 41 | 26 | 26 | 28 | 32 | 96 |
| transcribe.cpp | v0.4.1 | Metal | Q8_0 | 14.4 | 7645 | 46 | 13 | 14 | 20 | 33 | 115 |
| sherpa-onnx | 2.0.1 4 threads | CPU | int8 encoder | 94.9 | 7652 | 320 | 20 | 35 | 74 | 196 | 730 |

Warm time per call in ms.

NVIDIA GeForce RTX 4060, AMD Ryzen 9 7945HX, Windows 11, wavo only, Vulkan: Parakeet v3 29.0 s for the hour and 45 ms after 5 s; GigaAM v3 12.6 s and 19 ms.

## Accuracy

Word error rate in %, on the GPU, Apple M5, from `scripts/wer.py run`: LibriSpeech test-clean and test-other, the FLEURS test split, one normalizer for all, all errors over all reference words. FLEURS writes numbers as digits, so Russian also has a row without those clips.

| Model | Set | WER | Clips |
|---|---|---:|---:|
| gigaam-v3 | fleurs-ru | 4.52 | 775 |
| gigaam-v3 | fleurs-ru no digits | 3.27 | 620 |
| parakeet-v3 | test-clean | 1.91 | 2620 |
| parakeet-v3 | test-other | 3.59 | 2939 |
| parakeet-v3 | fleurs-en | 4.86 | 647 |
| parakeet-v3 | fleurs-fr | 5.22 | 676 |
| parakeet-v3 | fleurs-de | 5.23 | 862 |
| parakeet-v3 | fleurs-it | 3.03 | 865 |
| parakeet-v3 | fleurs-pl | 7.4 | 758 |
| parakeet-v3 | fleurs-pt | 4.95 | 919 |
| parakeet-v3 | fleurs-ru | 5.53 | 775 |
| parakeet-v3 | fleurs-ru no digits | 4.69 | 620 |
| parakeet-v3 | fleurs-es | 3.61 | 908 |
| parakeet-v3 | fleurs-uk | 6.85 | 750 |
| parakeet-ultra | test-clean | 1.8 | 2620 |
| parakeet-ultra | test-other | 3.43 | 2939 |
| parakeet-ultra | fleurs-en | 4.16 | 647 |
| parakeet-ultra | fleurs-fr | 4.63 | 676 |
| parakeet-ultra | fleurs-de | 4.53 | 862 |
| parakeet-ultra | fleurs-it | 2.31 | 865 |
| parakeet-ultra | fleurs-pl | 6.19 | 758 |
| parakeet-ultra | fleurs-pt | 4.13 | 919 |
| parakeet-ultra | fleurs-ru | 4.66 | 775 |
| parakeet-ultra | fleurs-ru no digits | 3.9 | 620 |
| parakeet-ultra | fleurs-es | 3.03 | 908 |
| parakeet-ultra | fleurs-uk | 5.63 | 750 |
| parakeet-v2 | test-clean | 1.69 | 2620 |
| parakeet-v2 | test-other | 3.2 | 2939 |
| parakeet-v2 | fleurs-en | 4.12 | 647 |
| whisper-turbo | fleurs-de | 5.73 | 862 |
| whisper-turbo | test-clean | 1.96 | 2620 |
| whisper-turbo | test-other | 3.92 | 2939 |
| whisper-turbo | fleurs-en | 4.44 | 647 |
| whisper-turbo | fleurs-es | 3.16 | 908 |
| whisper-turbo | fleurs-fr | 5.51 | 676 |
| whisper-turbo | fleurs-it | 2.7 | 865 |
| whisper-turbo | fleurs-pl | 5.84 | 758 |
| whisper-turbo | fleurs-pt | 4.25 | 919 |
| whisper-turbo | fleurs-ru | 4.52 | 775 |
| whisper-turbo | fleurs-ru no digits | 3.74 | 620 |
| whisper-turbo | fleurs-uk | 7.35 | 750 |
| moonshine-base | test-clean | 3.27 | 2620 |
| moonshine-base | test-other | 8.19 | 2939 |
| moonshine-base | fleurs-en | 11.54 | 647 |
| parakeet-he | fleurs-he | 23.14 | 792 |
| whisper-turbo-he | fleurs-he | 18.4 | 792 |

The Neural Engine stays close: on the sets checked, 4.59 against 4.52 for GigaAM on FLEURS ru and 1.99 against 1.91 for Parakeet v3 on test-clean.

## Rerun

```sh
scripts/wer.py run           # downloads what's missing, then measures
scripts/speed.py run         # builds each engine in .cache/engines/, then measures
```
