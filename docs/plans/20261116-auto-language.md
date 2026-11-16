# Auto language

One rule for `wavo run`, `wavo serve` and wavo-app's dictation: the speech's language picks the
model among the downloaded ones, unless the user names a model or a language. Nothing downloads
for a request.

Measured before (notes of PR #77, FLEURS): Parakeet's own text through whatlang names the
language right in 88/97/99% of 3/5/10 s clips of its 25 languages, in ~50 µs; whisper-tiny's
language token in 72% on 3 s, 89% on whole clips, 98% with a prior on the user's languages.

- [ ] Detection on the first ~10 s of speech: Parakeet v3 or ultra, when downloaded, by its text
      (whatlang); else Whisper's language token, whisper-tiny when downloaded; a prior on the
      languages of the downloaded models.
- [ ] The choice: the downloaded model for that language, by one ranking in `hub::MODELS`
      (a language's own model first: GigaAM for Russian, the Hebrew and Ukrainian fine-tunes;
      then Parakeet ultra and v3, then Whisper). When the detector is also the chosen model, its
      text is kept, so detection costs nothing.
- [ ] `wavo run talk.m4a` picks so; `--language CODE` picks without detecting; a named model is
      used as now. With nothing downloaded: parakeet-ultra, asking to download, as now.
- [ ] `wavo serve`: `model` absent or `auto` picks so among all downloaded models; `language`
      picks the model for it; a named model is used when downloaded, else 404. Models on the
      command line only load up front. Bazarr's `/asr` sends its language: the model for it.
- [ ] wavo-app (its session's answer, 2026-11-16): one language per recording, picked now by the
      keyboard layout (wavo-app #69); the right model must be loaded by the key's release, at
      most 100–150 ms after it. `Model::detect_language` of whisper-tiny on the first 1–3 s, run
      while the person speaks, ranked with probabilities, already does it: the app keeps its
      candidates (layouts, downloaded models) and falls back to the layout when unsure. Measure
      it on short accented phrases; the text check (whatlang) only after, as a second opinion.
- [ ] Measure: the right model on FLEURS clips of 2, 3, 5 and 10 s in ru, uk, en, he and de, and
      the time it adds.
