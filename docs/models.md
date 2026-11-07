# Models

## Which model for your language

Wrong words per 100, lower is better, the best in bold. *Made for it* is a model trained for that language alone.

| Language | `parakeet-ultra` | `parakeet-v3` | `whisper-turbo` | Made for it |
|---|---:|---:|---:|---|
| Bulgarian | **11.1** | 12.8 | 13.6 | – |
| Croatian | **11.3** | 12.6 | 12.4 | – |
| Czech | **11.1** | 12.2 | 11.7 | – |
| Danish | 15.9 | 18.6 | **13.8** | – |
| Dutch | 6.7 | 7.6 | **5.9** | – |
| English | 4.2 | 4.9 | 4.4 | `parakeet-v2` **4.1** |
| Estonian | **14.5** | 18.0 | 18.6 | – |
| Finnish | 11.2 | 13.5 | **8.6** | – |
| French | **4.6** | 5.2 | 5.5 | – |
| German | **4.5** | 5.2 | 5.7 | – |
| Greek | 32.2 | 35.3 | **13.2** | – |
| Hebrew | – | – | – | `whisper-turbo-he` **18.4**, `parakeet-he` 23.1 |
| Hungarian | **13.2** | 16.1 | 14.8 | – |
| Italian | **2.3** | 3.0 | 2.7 | – |
| Latvian | **17.9** | 23.9 | 19.2 | – |
| Lithuanian | **17.1** | 22.1 | 25.0 | – |
| Maltese | **17.0** | 20.6 | 72.8 | – |
| Polish | 6.2 | 7.4 | **5.8** | – |
| Portuguese | **4.1** | 5.0 | 4.2 | – |
| Romanian | **10.1** | 12.7 | 10.8 | – |
| Russian | 4.7 | 5.5 | **4.5** | `gigaam-v3` **4.5** |
| Slovak | **7.8** | 10.2 | 10.3 | – |
| Slovenian | **18.9** | 24.2 | 20.9 | – |
| Spanish | **3.0** | 3.6 | 3.2 | – |
| Swedish | 12.8 | 15.2 | **8.8** | – |
| Ukrainian | **5.6** | 6.8 | 7.3 | – |

Measured on the FLEURS test set. Whisper's other languages aren't measured.

## All models

| Name | Size | Languages | Output | Speed |
|---|---|---|---|---:|
| `parakeet-v3` | 0.74 GB | 25 European languages | cased, punctuated | 8.5× |
| `parakeet-v2` | 0.73 GB | English | cased, punctuated | 5.7× |
| `parakeet-ultra` (default) | 0.74 GB | 25 European languages | cased, punctuated | 7.8× |
| `parakeet-he` | 0.74 GB | Hebrew | cased, punctuated | 6.3× |
| `gigaam-v3` | 0.27 GB | Russian | cased, punctuated | 16.8× |
| `gigaam-v3-e2e-ctc` | 0.27 GB | Russian | cased, punctuated | – |
| `gigaam-v3-rnnt` | 0.27 GB | Russian | lowercase, no punctuation | – |
| `gigaam-v3-ctc` | 0.27 GB | Russian | lowercase, no punctuation | – |
| `whisper-turbo` | 0.89 GB | 100 languages | cased, punctuated | 1× |
| `whisper-turbo-he` | 0.89 GB | Hebrew | cased, punctuated | 0.7× |
| `whisper-tiny` | 0.05 GB | 99 languages | cased, punctuated | – |
| `whisper-base` | 0.08 GB | 99 languages | cased, punctuated | – |
| `whisper-small` | 0.27 GB | 99 languages | cased, punctuated | – |
| `whisper-medium` | 0.83 GB | 99 languages | cased, punctuated | – |
| `whisper-large-v3` | 1.67 GB | 100 languages | cased, punctuated | – |
| `whisper-tiny.en` | 0.05 GB | English | cased, punctuated | – |
| `whisper-base.en` | 0.08 GB | English | cased, punctuated | – |
| `whisper-small.en` | 0.27 GB | English | cased, punctuated | – |
| `whisper-medium.en` | 0.83 GB | English | cased, punctuated | – |
| `whisper-large` | 1.67 GB | 99 languages | cased, punctuated | – |
| `whisper-large-v2` | 1.67 GB | 99 languages | cased, punctuated | – |
| `breeze-asr-25` | 1.67 GB | Taiwanese Mandarin and English | cased, punctuated | – |
| `moonshine-tiny` | 0.04 GB | English | cased, punctuated | – |
| `moonshine-base` | 0.08 GB | English | cased, punctuated | 9.6× |

Speed is how many times faster than `whisper-turbo`. Trust the order, not the digits.

Moonshine gives no timestamps. Full names, such as `parakeet-tdt-0.6b-v3`, and a path to a `.gguf` work too.

## Credits

wavo's kernels and algorithms are built from the ideas and code of these projects:

- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT): the reference wavo matches, its GGUF format, converters, models and test clips. wavo's FFT, mel frontends, TDT and Whisper decoding, repetition guard and SentencePiece decoding follow its code.
- [ggml](https://github.com/ggml-org/ggml) (MIT): kernels adapted from it. Flash attention after Vulkan's `flash_attn_cm1.comp`, the Q8_0 matrix multiplication after ggml-metal's `kernel_mul_mm`, and GELU's erf.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT), [Hugging Face transformers](https://github.com/huggingface/transformers) and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Apache-2.0): algorithms that reached wavo through transcribe.cpp's ports.

**Models.** wavo ships no weights: `wavo pull` downloads the GGUF conversions that [handy-computer](https://huggingface.co/handy-computer) publishes on Hugging Face, and [vladkens](https://huggingface.co/vladkens) for the Hebrew models, under the original licenses:

- Parakeet TDT ([v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) © NVIDIA, CC-BY-4.0
- [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) © Moondream, CC-BY-4.0
- [GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3) © Sber, MIT
- Whisper © OpenAI, MIT for [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) and Apache-2.0 for the other sizes (such as [large-v3](https://huggingface.co/openai/whisper-large-v3)), as their model cards state
- Moonshine ([tiny](https://huggingface.co/UsefulSensors/moonshine-tiny), [base](https://huggingface.co/UsefulSensors/moonshine-base)) © Useful Sensors, MIT
- [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) © MediaTek Research, Apache-2.0
- [ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) © ivrit.ai, Apache-2.0
- [parakeet-tdt-0.6b-v3-he](https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he) © Yakov Kolani, CC-BY-4.0
