# Subtitles for Bazarr, Plex and Jellyfin

What users of AI subtitles in Bazarr, Plex and Jellyfin expect beyond the text: cues that read
well, timing that follows the speech, and the right model for each audio track's language.
Netflix's Timed Text Style Guide is the standard; SubGen's `refactor/drop-stable-ts` branch has a
cue splitter to compare with.

- [x] Cues of Parakeet and GigaAM: at most 2 lines of 42 characters, balanced, the break after
      punctuation or before a conjunction; 0.83–7 s; a new cue at a sentence end, else at a pause
      of 0.4 s or more, else after a comma; shown until 0.5 s after the last word, at most up to
      2 frames before the next cue.
- [x] Cues of Whisper, which times segments, not words: a segment past 7 s or 84 characters
      split at sentences and commas, its time shared by characters; a cue ends when its speech
      ends by the same estimate, not at the next segment (a line before music stays for minutes
      now).
- [x] The same cues in `--srt`, `serve`'s srt, vtt and verbose_json segments, and Bazarr's
      `/asr`.
- [ ] `/asr` takes the language Bazarr sends to the first model given that lists it, instead of
      a 400; untagged audio goes to the first model, as now (auto language later).
- [ ] Measure a 45-minute episode and a film against SubGen: time on an M5, an N100 and an RTX
      4060, and a sample SRT for the posts.
