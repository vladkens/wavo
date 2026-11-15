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

## Speaker diarization

Who speaks when in AMI's 16 test meetings: 9.1h of headset mix, 3–4 people each.

| Engine | On | Model | Error, aligned | Error, manual | Wrong speaker | Speaker count | Speed |
|---|---|---|---:|---:|---:|---:|---:|
| wavo | GPU | Nemotron 3 Diarization | 9.2% | 26.0% | 0.7% | 14/16 | 581× |
| FluidAudio 0.17.7 | ANE | Nemotron 3 Diarization | 9.5% | 26.3% | 0.7% | 14/16 | 499× |
| FluidAudio 0.17.7 | ANE | pyannote community-1 | 38.5% | 24.2% | 3.0% | 15/16 | 255× |
| wavo | CPU | Nemotron 3 Diarization | 9.2% | 26.0% | 0.7% | 14/16 | 38× |
| pyannote.audio 4.0.7 | GPU | pyannote community-1 | 34.7% | 17.1% | 4.0% | 12/16 | 13× |

- Error: the diarization error rate of pyannote.metrics with no collar and overlapping speech counted. It is the share of speech time with a speaker missed, made up or confused, all meetings together.
- AMI has two sets of labels, and they rank the models the other way round. *Aligned* are word times from a forced alignment, the pauses between words left out ([nttcslab-sp/diar-forced-alignment](https://github.com/nttcslab-sp/diar-forced-alignment)); NVIDIA's model card scores against them. *Manual* are the annotators' word times ([pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup), only_words); pyannote's model card scores against them. Each model gets its own card's number: Nemotron 9.2% against 9.25%, pyannote 17.1% against 17.0%. Against the manual labels Nemotron misses more speech (24.1% against 9.5%); against the aligned ones pyannote marks more speech where there is none (24.9% against 3.7%).
- Wrong speaker: the part of the error where speech went to the wrong person, against the manual labels.
- Speaker count: the meetings where the engine finds as many speakers as the labels have.
- Speed: how many times faster than real time. wavo's time is the whole `wavo run` of each meeting, model load and audio decode included. The others' time is the diarization call alone, with the model loaded once.
- Nemotron 3 Diarization tells at most 8 speakers apart. NVIDIA trained it on AMI's train and dev meetings, not test, and on VoxConverse, so VoxConverse is left out.

Measured on a MacBook Air with an Apple M5 and 32GB, macOS 27. Rerun it with `scripts/diarization.py run`. The raw numbers are in [results/](../results).
