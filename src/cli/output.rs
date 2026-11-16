// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The renderings of a transcript: `--json`, and for `wavo serve` OpenAI's `verbose_json` and
//! `diarized_json`; with `--diarize` by speaker, and a diarization model's turns alone. Subtitles
//! are in `subtitles`.

use std::fmt::Write;

use wavo::{Transcript, Turn, Utterance};

use crate::subtitles::{Cue, time};

/// `{"text": ..., "language": ..., "tokens": [{"id", "piece", "start_ms"}, ...]}`, one token per
/// line; `language` only when the model says it. With `utterances`,
/// `"utterances": [{"speaker", "start_ms", "end_ms", "text"}, ...]` follow; speakers count from 1,
/// `null` when unknown.
pub fn json(t: &Transcript, utterances: Option<&[Utterance]>) -> String {
  let language = t.language.as_deref().map(|l| format!("\"language\": {}, ", quote(l)));
  let mut out =
    format!("{{\"text\": {}, {}\"tokens\": [", quote(&t.text), language.unwrap_or_default());
  for (i, token) in t.tokens.iter().enumerate() {
    let (sep, id, piece, start) =
      (if i == 0 { "" } else { "," }, token.id, quote(&token.piece), token.start_ms);
    write!(out, "{sep}\n  {{\"id\": {id}, \"piece\": {piece}, \"start_ms\": {start}}}").unwrap();
  }

  out += if t.tokens.is_empty() { "]" } else { "\n]" };
  let Some(utterances) = utterances else { return out + "}" };

  out += ", \"utterances\": [";
  for (i, u) in utterances.iter().enumerate() {
    let speaker = u.speaker.map_or("null".into(), |s| (s + 1).to_string());
    let (sep, start, end, text) =
      (if i == 0 { "" } else { "," }, u.start_ms, u.end_ms, quote(&u.text));
    write!(
      out,
      "{sep}\n  {{\"speaker\": {speaker}, \"start_ms\": {start}, \"end_ms\": {end}, \"text\": {text}}}"
    )
    .unwrap();
  }

  out + if utterances.is_empty() { "]}" } else { "\n]}" }
}

/// `Speaker 1: text` per utterance, one per line.
pub fn by_speaker(utterances: &[Utterance]) -> String {
  utterances.iter().map(|u| format!("{}: {}\n", label(u.speaker), u.text)).collect()
}

fn label(speaker: Option<u32>) -> String {
  speaker.map_or("Speaker ?".into(), |s| format!("Speaker {}", s + 1))
}

/// A diarization model's turns: `00:00:00.320 --> 00:00:02.400  Speaker 1` per line, or with
/// `json` `{"turns": [{"speaker", "start_ms", "end_ms"}, ...]}` (speakers from 1).
pub fn turns(turns: &[Turn], json: bool) -> String {
  if !json {
    let line = |t: &Turn| {
      let (start, end) = (time(t.start_ms, '.'), time(t.end_ms, '.'));
      format!("{start} --> {end}  {}\n", label(Some(t.speaker)))
    };
    return turns.iter().map(line).collect();
  }

  let turns: Vec<String> = (turns.iter())
    .map(|t| {
      let (s, a, b) = (t.speaker + 1, t.start_ms, t.end_ms);
      format!("\n  {{\"speaker\": {s}, \"start_ms\": {a}, \"end_ms\": {b}}}")
    })
    .collect();
  let end = if turns.is_empty() { "" } else { "\n" };
  format!("{{\"turns\": [{}{end}]}}\n", turns.join(","))
}

/// A JSON string.
pub fn quote(s: &str) -> String {
  let mut out = String::from("\"");
  for c in s.chars() {
    match c {
      '"' | '\\' => write!(out, "\\{c}").unwrap(),
      '\n' => out += "\\n",
      c if c < ' ' => write!(out, "\\u{:04x}", c as u32).unwrap(),
      c => out.push(c),
    }
  }
  out + "\""
}

/// OpenAI's `verbose_json`: the text, the cues as segments and, from token times, the words, with
/// times in seconds. `language` is the code asked for, or empty.
pub fn verbose_json(t: &Transcript, cues: &[Cue], language: &str, audio_ms: u32) -> String {
  let (language, duration, text) = (quote(language), secs(audio_ms), quote(&t.text));
  let mut out = format!(
    "{{\"task\": \"transcribe\", \"language\": {language}, \"duration\": {duration}, \
     \"text\": {text}, \"segments\": ["
  );
  for (i, c) in cues.iter().enumerate() {
    let ids: Vec<String> = t.tokens[c.tokens.clone()].iter().map(|t| t.id.to_string()).collect();
    let (sep, start, end, text) =
      (if i == 0 { "" } else { "," }, secs(c.start), secs(c.end), quote(&c.text));
    write!(
      out,
      "{sep}\n  {{\"id\": {i}, \"seek\": 0, \"start\": {start}, \"end\": {end}, \"text\": {text}, \
       \"tokens\": [{}], \"temperature\": 0.0}}",
      ids.join(", ")
    )
    .unwrap();
  }

  out += if cues.is_empty() { "]" } else { "\n]" };
  let words: Vec<&(u32, u32, String)> = cues.iter().flat_map(|c| &c.words).collect();
  if words.is_empty() {
    return out + "}";
  }

  out += ", \"words\": [";
  for (i, (start, end, word)) in words.into_iter().enumerate() {
    let (sep, start, end, word) =
      (if i == 0 { "" } else { "," }, secs(*start), secs(*end), quote(word));
    write!(out, "{sep}\n  {{\"word\": {word}, \"start\": {start}, \"end\": {end}}}").unwrap();
  }
  out + "\n]}"
}

/// Utterances of a model without word times (Whisper) end where the next starts, the last at
/// `audio_ms`, as its cues do: the library ends them a word's length after their start.
pub fn stretch(utterances: &mut [Utterance], audio_ms: u32) {
  let nexts: Vec<u32> = utterances.iter().skip(1).map(|u| u.start_ms).chain([audio_ms]).collect();
  utterances.iter_mut().zip(nexts).for_each(|(u, next)| u.end_ms = next.max(u.start_ms));
}

/// OpenAI's `diarized_json`: the text, and the utterances as segments with times in seconds and
/// speakers as letters, `A` for the first to talk (and when there were no turns).
pub fn diarized_json(text: &str, utterances: &[Utterance], audio_ms: u32) -> String {
  let (duration, text) = (secs(audio_ms), quote(text));
  let mut out = format!(
    "{{\"task\": \"transcribe\", \"duration\": {duration}, \"text\": {text}, \"segments\": ["
  );
  for (i, u) in utterances.iter().enumerate() {
    let speaker = char::from(b'A' + u.speaker.unwrap_or(0) as u8);
    let (sep, start, end, text) =
      (if i == 0 { "" } else { "," }, secs(u.start_ms), secs(u.end_ms), quote(&u.text));
    write!(
      out,
      "{sep}\n  {{\"type\": \"transcript.text.segment\", \"id\": \"seg_{i}\", \"start\": {start}, \
       \"end\": {end}, \"text\": {text}, \"speaker\": \"{speaker}\"}}"
    )
    .unwrap();
  }

  out + if utterances.is_empty() { "]}" } else { "\n]}" }
}

/// Milliseconds as exact decimal seconds: `1.5`, `0.04`, `2.0`.
fn secs(ms: u32) -> String {
  let frac = format!("{:03}", ms % 1000);
  let frac = frac.trim_end_matches('0');
  format!("{}.{}", ms / 1000, if frac.is_empty() { "0" } else { frac })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::subtitles::tests::tokens;
  use crate::subtitles::{cues, speaker_cues, srt};

  #[test]
  fn json_escapes() {
    let mut t = Transcript::default();
    (t.text, t.tokens) = ("a \"b\"\\\n\u{1}é".into(), tokens(&[("▁a", 40)]));
    let expected = "{\"text\": \"a \\\"b\\\"\\\\\\n\\u0001é\", \"tokens\": [\n  {\"id\": 0, \"piece\": \"▁a\", \"start_ms\": 40}\n]}";
    assert_eq!(json(&t, None), expected);
    assert_eq!(json(&Transcript::default(), None), "{\"text\": \"\", \"tokens\": []}");
  }

  #[test]
  fn verbose_json_segments_and_words() {
    let mut t = Transcript::default();
    let pieces = [("▁Hel", 0), ("lo", 80), ("▁you.", 400), ("▁Go", 4000)];
    (t.text, t.tokens) = ("Hello you. Go".into(), tokens(&pieces));
    let cues = cues(&t.tokens, &[], 4500, true);
    assert_eq!(
      verbose_json(&t, &cues, "en", 4500),
      "{\"task\": \"transcribe\", \"language\": \"en\", \"duration\": 4.5, \"text\": \"Hello you. Go\", \"segments\": [\
       \n  {\"id\": 0, \"seek\": 0, \"start\": 0.0, \"end\": 1.4, \"text\": \"Hello you.\", \"tokens\": [0, 1, 2], \"temperature\": 0.0},\
       \n  {\"id\": 1, \"seek\": 0, \"start\": 4.0, \"end\": 4.5, \"text\": \"Go\", \"tokens\": [3], \"temperature\": 0.0}\
       \n], \"words\": [\
       \n  {\"word\": \"Hello\", \"start\": 0.0, \"end\": 0.4},\
       \n  {\"word\": \"you.\", \"start\": 0.4, \"end\": 1.4},\
       \n  {\"word\": \"Go\", \"start\": 4.0, \"end\": 4.5}\
       \n]}"
    );
    // Segment times only: no words.
    let cues = crate::subtitles::cues(&t.tokens, &[], 4500, false);
    let json = verbose_json(&t, &cues, "", 4500);
    assert!(json.contains("\"language\": \"\"") && !json.contains("words"), "{json}");
    assert!(json.ends_with("\"temperature\": 0.0}\n]}"), "{json}");
    let empty = verbose_json(&Transcript::default(), &[], "", 40);
    assert!(empty.ends_with("\"duration\": 0.04, \"text\": \"\", \"segments\": []}"), "{empty}");
  }

  fn turn(speaker: u32, start_ms: u32, end_ms: u32) -> Turn {
    let mut t = Turn::default();
    (t.speaker, t.start_ms, t.end_ms) = (speaker, start_ms, end_ms);
    t
  }

  #[test]
  fn by_speakers() {
    let mut t = Transcript::default();
    let pieces = [("▁Hi.", 0), ("▁Yes.", 1000), ("▁And", 4000), ("▁then", 4200)];
    (t.text, t.tokens) = ("Hi. Yes. And then".into(), tokens(&pieces));
    let u = t.utterances(&[turn(0, 0, 800), turn(1, 900, 1500), turn(0, 3900, 5000)]);
    assert_eq!(by_speaker(&u), "Speaker 1: Hi.\nSpeaker 2: Yes.\nSpeaker 1: And then\n");
    // No cue spans two speakers; each shows until the next speaker's starts.
    assert_eq!(
      srt(&speaker_cues(&t.tokens, &[], &u, 6000, true)),
      "1\n00:00:00,000 --> 00:00:01,000\nHi.\n\n2\n00:00:01,000 --> 00:00:02,000\nYes.\n\n\
       3\n00:00:04,000 --> 00:00:05,200\nAnd then\n\n"
    );
    let rendered = json(&t, Some(&u));
    assert!(rendered.contains("\"piece\": \"▁Yes.\", \"start_ms\": 1000},"), "{rendered}");
    assert!(
      rendered.ends_with(
        "\"utterances\": [\n  {\"speaker\": 1, \"start_ms\": 0, \"end_ms\": 200, \"text\": \"Hi.\"},\
         \n  {\"speaker\": 2, \"start_ms\": 1000, \"end_ms\": 1200, \"text\": \"Yes.\"},\
         \n  {\"speaker\": 1, \"start_ms\": 4000, \"end_ms\": 4400, \"text\": \"And then\"}\n]}"
      ),
      "{rendered}"
    );
    // No turns at all: no speaker.
    let none = t.utterances(&[]);
    assert_eq!(by_speaker(&none), "Speaker ?: Hi. Yes. And then\n");
    assert!(json(&t, Some(&none)).contains("\"speaker\": null"));
    // OpenAI's diarized_json: speakers as letters, times in seconds.
    assert_eq!(
      diarized_json(&t.text, &u, 6000),
      "{\"task\": \"transcribe\", \"duration\": 6.0, \"text\": \"Hi. Yes. And then\", \
       \"segments\": [\
       \n  {\"type\": \"transcript.text.segment\", \"id\": \"seg_0\", \"start\": 0.0, \"end\": 0.2, \
       \"text\": \"Hi.\", \"speaker\": \"A\"},\
       \n  {\"type\": \"transcript.text.segment\", \"id\": \"seg_1\", \"start\": 1.0, \"end\": 1.2, \
       \"text\": \"Yes.\", \"speaker\": \"B\"},\
       \n  {\"type\": \"transcript.text.segment\", \"id\": \"seg_2\", \"start\": 4.0, \"end\": 4.4, \
       \"text\": \"And then\", \"speaker\": \"A\"}\n]}"
    );
    assert!(diarized_json(&t.text, &none, 6000).contains("\"speaker\": \"A\""));
    assert!(diarized_json("", &[], 40).ends_with("\"segments\": []}"));

    // Whisper's: each to the next one's start, the last to the audio's end.
    let mut u = u;
    stretch(&mut u, 6000);
    let ends: Vec<u32> = u.iter().map(|u| u.end_ms).collect();
    assert_eq!(ends, [1000, 4000, 6000]);
  }

  #[test]
  fn turns_alone() {
    let t = [turn(0, 320, 2400), turn(1, 2000, 3_600_080)];
    assert_eq!(
      turns(&t, false),
      "00:00:00.320 --> 00:00:02.400  Speaker 1\n00:00:02.000 --> 01:00:00.080  Speaker 2\n"
    );
    assert_eq!(
      turns(&t, true),
      "{\"turns\": [\n  {\"speaker\": 1, \"start_ms\": 320, \"end_ms\": 2400},\
       \n  {\"speaker\": 2, \"start_ms\": 2000, \"end_ms\": 3600080}\n]}\n"
    );
    assert_eq!(turns(&[], true), "{\"turns\": []}\n");
  }
}
