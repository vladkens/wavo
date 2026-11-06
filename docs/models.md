# Models

## Which model for your language

Wrong words per 100 on the FLEURS test set, lower is better. The best model of each language is in the second column.

| Language | Best | `parakeet-ultra` | `parakeet-v3` | `whisper-turbo` |
|---|---|---:|---:|---:|
| Bulgarian | `parakeet-ultra`, 11.1 | 11.1 | 12.8 | 13.6 |
| Croatian | `parakeet-ultra`, 11.3 | 11.3 | 12.6 | 12.4 |
| Czech | `parakeet-ultra`, 11.1 | 11.1 | 12.2 | 11.7 |
| Danish | `whisper-turbo`, 13.8 | 15.9 | 18.6 | 13.8 |
| Dutch | `whisper-turbo`, 5.9 | 6.7 | 7.6 | 5.9 |
| English | `parakeet-v2`, 4.1 | 4.2 | 4.9 | 4.4 |
| Estonian | `parakeet-ultra`, 14.5 | 14.5 | 18.0 | 18.6 |
| Finnish | `whisper-turbo`, 8.6 | 11.2 | 13.5 | 8.6 |
| French | `parakeet-ultra`, 4.6 | 4.6 | 5.2 | 5.5 |
| German | `parakeet-ultra`, 4.5 | 4.5 | 5.2 | 5.7 |
| Greek | `whisper-turbo`, 13.2 | 32.2 | 35.3 | 13.2 |
| Hebrew | `whisper-turbo-he`, 18.4 | – | – | – |
| Hungarian | `parakeet-ultra`, 13.2 | 13.2 | 16.1 | 14.8 |
| Italian | `parakeet-ultra`, 2.3 | 2.3 | 3.0 | 2.7 |
| Latvian | `parakeet-ultra`, 17.9 | 17.9 | 23.9 | 19.2 |
| Lithuanian | `parakeet-ultra`, 17.1 | 17.1 | 22.1 | 25.0 |
| Maltese | `parakeet-ultra`, 17.0 | 17.0 | 20.6 | 72.8 |
| Polish | `whisper-turbo`, 5.8 | 6.2 | 7.4 | 5.8 |
| Portuguese | `parakeet-ultra`, 4.1 | 4.1 | 5.0 | 4.2 |
| Romanian | `parakeet-ultra`, 10.1 | 10.1 | 12.7 | 10.8 |
| Russian | `gigaam-v3`, 4.5 | 4.7 | 5.5 | 4.5 |
| Slovak | `parakeet-ultra`, 7.8 | 7.8 | 10.2 | 10.3 |
| Slovenian | `parakeet-ultra`, 18.9 | 18.9 | 24.2 | 20.9 |
| Spanish | `parakeet-ultra`, 3.0 | 3.0 | 3.6 | 3.2 |
| Swedish | `whisper-turbo`, 8.8 | 12.8 | 15.2 | 8.8 |
| Ukrainian | `parakeet-ultra`, 5.6 | 5.6 | 6.8 | 7.3 |
| 1 hour of audio takes | | ~30 s | 22 s | ~4 min |

`gigaam-v3` takes 11 s for an hour of Russian, and `parakeet-he` 23.1 on Hebrew at about Parakeet's speed. On English LibriSpeech test-clean: `parakeet-v2` 1.7, `parakeet-ultra` 1.8, `parakeet-v3` 1.9, `whisper-turbo` 2.0, `moonshine-base` 3.3. Measured with wavo on the GPU by `scripts/wer.py run`; the other languages of Whisper aren't measured yet. The library has these numbers too: `ModelInfo::wer`, `ModelInfo::speed` and `hub::best_for`.

## All models

| Name | Size | Languages | Output |
|---|---|---|---|
| `parakeet-v3` | 0.74 GB | 25 European languages | cased, punctuated |
| `parakeet-v2` | 0.73 GB | English | cased, punctuated |
| `parakeet-ultra` (default) | 0.74 GB | 25 European languages | cased, punctuated |
| `parakeet-he` | 0.74 GB | Hebrew | cased, punctuated |
| `gigaam-v3` | 0.27 GB | Russian | cased, punctuated |
| `gigaam-v3-e2e-ctc` | 0.27 GB | Russian | cased, punctuated |
| `gigaam-v3-rnnt` | 0.27 GB | Russian | lowercase, no punctuation |
| `gigaam-v3-ctc` | 0.27 GB | Russian | lowercase, no punctuation |
| `whisper-turbo` | 0.89 GB | 100 languages | cased, punctuated |
| `whisper-turbo-he` | 0.89 GB | Hebrew | cased, punctuated |
| `whisper-tiny` | 0.05 GB | 99 languages | cased, punctuated |
| `whisper-base` | 0.08 GB | 99 languages | cased, punctuated |
| `whisper-small` | 0.27 GB | 99 languages | cased, punctuated |
| `whisper-medium` | 0.83 GB | 99 languages | cased, punctuated |
| `whisper-large-v3` | 1.67 GB | 100 languages | cased, punctuated |
| `whisper-tiny.en` | 0.05 GB | English | cased, punctuated |
| `whisper-base.en` | 0.08 GB | English | cased, punctuated |
| `whisper-small.en` | 0.27 GB | English | cased, punctuated |
| `whisper-medium.en` | 0.83 GB | English | cased, punctuated |
| `whisper-large` | 1.67 GB | 99 languages | cased, punctuated |
| `whisper-large-v2` | 1.67 GB | 99 languages | cased, punctuated |
| `breeze-asr-25` | 1.67 GB | Taiwanese Mandarin and English | cased, punctuated |
| `moonshine-tiny` | 0.04 GB | English | cased, punctuated |
| `moonshine-base` | 0.08 GB | English | cased, punctuated |

Parakeet is NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) is Moondream's retraining of v3 with fewer errors in each of its 25 languages, GigaAM is Sber's [GigaAM v3](https://github.com/salute-developers/GigaAM), Whisper is OpenAI's [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), Breeze-ASR-25 is MediaTek Research's [Whisper large-v2 fine-tune](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) and Moonshine is Useful Sensors' [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny), small and fast. Moonshine has no timestamps: its tokens all start at 0, and `--srt` gives one cue per piece of the split audio. The Hebrew models are ivrit.ai's [Whisper turbo fine-tune](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) and Yakov Kolani's [Parakeet v3 fine-tune](https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he). Short names stand for the published ones: `parakeet-v3` is `parakeet-tdt-0.6b-v3`, `parakeet-v2` is `parakeet-tdt-0.6b-v2`, `parakeet-he` is `parakeet-tdt-0.6b-v3-he`, `gigaam-v3` is `gigaam-v3-e2e-rnnt`, `whisper-turbo` is `whisper-large-v3-turbo`, `whisper-turbo-he` is `ivrit-whisper-large-v3-turbo` and `breeze-asr-25` is `Breeze-ASR-25`; full names and a path to a `.gguf` work too.

`parakeet-ultra` is the default: it makes fewer errors than `parakeet-v3` on every benchmark group of [Moondream's model card](https://huggingface.co/moondream/parakeet-ultra#benchmarks), which also lists each language. Word error rate, %, lower is better, measured by Moondream on an NVIDIA GPU, not with wavo:

| Benchmark | `parakeet-v3` | `parakeet-ultra` |
|---|---:|---:|
| English, 7 sets of the Open ASR Leaderboard | 6.26 | 5.80 |
| FLEURS, 25 languages | 11.62 | 9.55 |
| Business speech (AMI, VoxPopuli, Earnings-22) | 6.15 | 5.79 |
| Background noise, 9 MUSAN conditions | 6.72 | 5.82 |
| Long talks, TED-LIUM 3 | 2.71 | 1.94 |

## Credits

wavo's kernels and algorithms are built from the ideas and code of these projects:

- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT): the reference wavo
  matches, its GGUF format and converters, the published models and the test clips. wavo's FFT,   mel frontends, TDT and Whisper decoding, repetition guard and SentencePiece decoding follow its   code.
- [ggml](https://github.com/ggml-org/ggml) (MIT): kernels adapted from it. Flash attention after
  Vulkan's `flash_attn_cm1.comp`, the Q8_0 matrix multiplication after ggml-metal's   `kernel_mul_mm`, and GELU's erf.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT),
  [Hugging Face transformers](https://github.com/huggingface/transformers) and   [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Apache-2.0): algorithms that reached wavo through   transcribe.cpp's ports.

**Models.** wavo ships no weights: `wavo pull` downloads the GGUF conversions that [handy-computer](https://huggingface.co/handy-computer) publishes on Hugging Face, and [vladkens](https://huggingface.co/vladkens) for the Hebrew models, under the original licenses:

- Parakeet TDT ([v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
  [v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) © NVIDIA, CC-BY-4.0
- [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) © Moondream, CC-BY-4.0
- [GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3) © Sber, MIT
- Whisper © OpenAI, MIT for [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  and Apache-2.0 for the other sizes (such as   [large-v3](https://huggingface.co/openai/whisper-large-v3)), as their model cards state
- Moonshine ([tiny](https://huggingface.co/UsefulSensors/moonshine-tiny),
  [base](https://huggingface.co/UsefulSensors/moonshine-base)) © Useful Sensors, MIT
- [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) © MediaTek Research,
  Apache-2.0
- [ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) ©
  ivrit.ai, Apache-2.0
- [parakeet-tdt-0.6b-v3-he](https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he) © Yakov
  Kolani, CC-BY-4.0
