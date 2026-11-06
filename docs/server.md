# Server

`wavo serve` answers OpenAI's transcription API, so apps and scripts written for OpenAI's
`whisper-1` (the `openai` Python and JS clients, apps that let you set the API's URL) transcribe
locally instead:

```sh
wavo serve                                       # parakeet-ultra on http://127.0.0.1:8000
wavo serve whisper-turbo gigaam-v3 --port 9000   # these models, loaded up front
wavo serve --host 0.0.0.0 --api-key KEY          # on the network, with a key (or WAVO_API_KEY)
wavo serve parakeet-v3 --backend gpu             # the GPU even where the Neural Engine is ready

curl -F file=@talk.mp3 -F model=whisper-1 http://127.0.0.1:8000/v1/audio/transcriptions
curl -F file=@talk.mp3 -F model=gigaam-v3 -F response_format=srt http://127.0.0.1:8000/v1/audio/transcriptions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="KEY")  # any key without --api-key
with open("talk.mp3", "rb") as f:
  print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

`POST /v1/audio/transcriptions` takes the form fields `file`, `model`, `language` and
`response_format`: `json` (`{"text": ...}`), `text`, `srt`, `vtt` or `verbose_json` (segments
with start and end times, and the words for models with token times: not Whisper or Moonshine).
A `model` that names no wavo model, such as `whisper-1`, goes to the first model given; another
downloaded model loads on its first request, replacing the one loaded that way before. Long audio
is split at pauses as in `wavo run`. `GET /v1/models` lists the downloaded models and `GET /health`
the GPU and the loaded models. `serve` downloads the models it is started with, after asking or
with `-y`; requests never download. It serves 16 requests at once, each up to 512 MB and 3 hours
of audio, and answers more with 503.

For [Bazarr](https://www.bazarr.media)'s subtitles, `POST /asr` answers as
[whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) does: enable Bazarr's
Whisper provider and set its endpoint to `http://HOST:8000`, with `wavo serve` started without
`--api-key`, as Bazarr sends none; its Test Connection shows wavo's version (`GET /status`). The
first model transcribes, in the language of the audio track's tag, which it must know; wavo
doesn't translate, nor detect the language of an untagged track (`/detect-language`).
