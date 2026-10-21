// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The renderings of a transcript: `--json` and `--srt`, and for `wavo serve` also WebVTT and
//! OpenAI's `verbose_json`.

use std::fmt::Write;
use std::ops::Range;

use wavo::{Token, Transcript};

/// A cue ends before it would grow past this many characters.
const MAX_CUE: usize = 80;
/// A cue shows at most this long after its last token starts.
const LINGER_MS: u32 = 1500;

/// `{"text": ..., "tokens": [{"id", "piece", "start_ms"}, ...]}`, one token per line.
pub fn json(t: &Transcript) -> String {
  let mut out = format!("{{\"text\": {}, \"tokens\": [", quote(&t.text));
  for (i, token) in t.tokens.iter().enumerate() {
    let (sep, id, piece, start) =
      (if i == 0 { "" } else { "," }, token.id, quote(&token.piece), token.start_ms);
    write!(out, "{sep}\n  {{\"id\": {id}, \"piece\": {piece}, \"start_ms\": {start}}}").unwrap();
  }
  out + if t.tokens.is_empty() { "]}" } else { "\n]}" }
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

/// A subtitle cue, and a segment of `verbose_json`.
pub struct Cue {
  /// Start and end in milliseconds.
  pub start: u32,
  pub end: u32,
  pub text: String,
  /// Its tokens: a range of the transcript's.
  pub tokens: Range<usize>,
  /// Its words with start and end in milliseconds; none from a model that times segments only.
  pub words: Vec<(u32, u32, String)>,
}

/// The cues of `tokens` over `audio_ms` of audio: from token times when `timed`, else (Whisper,
/// Moonshine) from segments. `starts` holds the first token of each split part, sorted.
pub fn cues(tokens: &[Token], starts: &[usize], audio_ms: u32, timed: bool) -> Vec<Cue> {
  if timed { word_cues(tokens, starts, audio_ms) } else { segment_cues(tokens, audio_ms) }
}

/// Words from pieces (`▁` or a space starts a word, and so does each token index in `starts`)
/// grouped into cues that end after sentence punctuation or before passing `MAX_CUE` characters.
/// A cue shows until the next one starts, the audio ends or `LINGER_MS` after its last token,
/// whichever is first; a word until the next word starts or its cue ends.
fn word_cues(tokens: &[Token], starts: &[usize], audio_ms: u32) -> Vec<Cue> {
  // first and last token start, text, tokens
  let mut words: Vec<(u32, u32, String, Range<usize>)> = Vec::new();
  for (i, t) in tokens.iter().enumerate() {
    let piece = t.piece.replace('▁', " ");
    match words.last_mut() {
      Some(word) if !piece.starts_with(' ') && starts.binary_search(&i).is_err() => {
        (word.1, word.3.end) = (t.start_ms, i + 1);
        word.2 += &piece;
      }
      _ => words.push((t.start_ms, t.start_ms, piece, i..i + 1)),
    }
  }

  // A cue's `end` holds its last token's start until the loop below.
  let mut cues: Vec<Cue> = Vec::new();
  for (first, last, text, tokens) in words {
    let text = text.trim();
    match cues.last_mut() {
      _ if text.is_empty() => {}
      // A word that starts with the cue (several in one frame) stays in it, else the cue would
      // last no time at all.
      Some(cue)
        if first == cue.start
          || !cue.text.ends_with(['.', '!', '?', '…'])
            && cue.text.chars().count() + 1 + text.chars().count() <= MAX_CUE =>
      {
        (cue.end, cue.tokens.end) = (last, tokens.end);
        cue.text = format!("{} {text}", cue.text);
        cue.words.push((first, first, text.into()));
      }
      _ => {
        let words = vec![(first, first, text.into())];
        cues.push(Cue { start: first, end: last, text: text.into(), tokens, words });
      }
    }
  }

  let nexts: Vec<u32> = cues.iter().skip(1).map(|c| c.start).chain([audio_ms]).collect();
  for (cue, next) in cues.iter_mut().zip(nexts) {
    cue.end = next.min(cue.end + LINGER_MS).max(cue.start);
    let ends: Vec<u32> = cue.words.iter().skip(1).map(|w| w.0).chain([cue.end]).collect();
    cue.words.iter_mut().zip(ends).for_each(|(w, end)| w.1 = end);
  }

  cues
}

/// One cue per segment of a model that times segments, not tokens (Whisper: every token carries
/// its segment's start; Moonshine: 0 in each piece of the audio), shown until the next one starts
/// or the audio ends.
fn segment_cues(tokens: &[Token], audio_ms: u32) -> Vec<Cue> {
  let mut cues: Vec<Cue> = Vec::new();
  for (i, t) in tokens.iter().enumerate() {
    let piece = t.piece.replace('▁', " ");
    match cues.last_mut() {
      Some(cue) if cue.start == t.start_ms => {
        cue.text += &piece;
        cue.tokens.end = i + 1;
      }
      _ => {
        let (start, tokens, words) = (t.start_ms, i..i + 1, Vec::new());
        cues.push(Cue { start, end: start, text: piece, tokens, words });
      }
    }
  }

  cues.iter_mut().for_each(|c| c.text = c.text.trim().to_string());
  cues.retain(|c| !c.text.is_empty());
  let nexts: Vec<u32> = cues.iter().skip(1).map(|c| c.start).chain([audio_ms]).collect();
  cues.iter_mut().zip(nexts).for_each(|(c, next)| c.end = next.max(c.start));
  cues
}

/// SubRip subtitles: numbered cues with times like `00:00:01,500`.
pub fn srt(cues: &[Cue]) -> String {
  let mut out = String::new();
  for (i, c) in cues.iter().enumerate() {
    let (start, end) = (time(c.start, ','), time(c.end, ','));
    write!(out, "{}\n{start} --> {end}\n{}\n\n", i + 1, c.text).unwrap();
  }
  out
}

/// WebVTT subtitles: a `WEBVTT` header, then cues with times like `00:00:01.500`.
pub fn vtt(cues: &[Cue]) -> String {
  let mut out = String::from("WEBVTT\n\n");
  for c in cues {
    write!(out, "{} --> {}\n{}\n\n", time(c.start, '.'), time(c.end, '.'), c.text).unwrap();
  }
  out
}

fn time(ms: u32, sep: char) -> String {
  let (h, m, s) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60);
  format!("{h:02}:{m:02}:{s:02}{sep}{:03}", ms % 1000)
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

/// Milliseconds as exact decimal seconds: `1.5`, `0.04`, `2.0`.
fn secs(ms: u32) -> String {
  let frac = format!("{:03}", ms % 1000);
  let frac = frac.trim_end_matches('0');
  format!("{}.{}", ms / 1000, if frac.is_empty() { "0" } else { frac })
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use super::*;

  fn tokens(pieces: &[(&str, u32)]) -> Vec<Token> {
    let token = |(i, &(piece, start_ms)): (usize, &(&str, u32))| {
      let mut t = Token::default();
      (t.id, t.piece, t.start_ms) = (i as u32, piece.to_string(), start_ms);
      t
    };
    pieces.iter().enumerate().map(token).collect()
  }

  fn srt_of(tokens: &[Token], starts: &[usize], audio_ms: u32) -> String {
    srt(&cues(tokens, starts, audio_ms, true))
  }

  fn srt_segments(tokens: &[Token], audio_ms: u32) -> String {
    srt(&cues(tokens, &[], audio_ms, false))
  }

  #[test]
  fn json_escapes() {
    let mut t = Transcript::default();
    (t.text, t.tokens) = ("a \"b\"\\\n\u{1}é".into(), tokens(&[("▁a", 40)]));
    let expected = "{\"text\": \"a \\\"b\\\"\\\\\\n\\u0001é\", \"tokens\": [\n  {\"id\": 0, \"piece\": \"▁a\", \"start_ms\": 40}\n]}";
    assert_eq!(json(&t), expected);
    assert_eq!(json(&Transcript::default()), "{\"text\": \"\", \"tokens\": []}");
  }

  #[test]
  fn cues_split_on_sentences_and_length() {
    let t = tokens(&[("▁Hel", 0), ("lo", 80), (".", 160), ("▁How", 4000), ("▁are", 4100)]);
    assert_eq!(
      srt_of(&t, &[], 6000),
      "1\n00:00:00,000 --> 00:00:01,660\nHello.\n\n2\n00:00:04,000 --> 00:00:05,600\nHow are\n\n"
    );
    let long: Vec<(String, u32)> = (0..30).map(|i| (format!("▁w{i:02}"), i * 100)).collect();
    let long: Vec<(&str, u32)> = long.iter().map(|(p, s)| (p.as_str(), *s)).collect();
    let srt = srt_of(&tokens(&long), &[], 3000);
    let lines: Vec<&str> = srt.lines().filter(|l| l.starts_with('w')).collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0].len(), 79, "20 words of 3 chars and their spaces");
    assert!(srt.contains("00:00:00,000 --> 00:00:02,000\n"), "ends where the next cue starts");
    assert!(srt.contains("00:00:02,000 --> 00:00:03,000\n"), "the last ends with the audio");
  }

  #[test]
  fn character_vocabularies_and_hours() {
    let t = tokens(&[
      ("д", 3_600_000),
      ("а", 3_600_040),
      (" ", 3_600_080),
      ("н", 3_600_120),
      (" ", 3_600_200),
    ]);
    assert_eq!(srt_of(&t, &[], 3_700_000), "1\n01:00:00,000 --> 01:00:01,620\nда н\n\n");
    assert_eq!(srt_of(&[], &[], 1000), "");
    // A segment of a split recording starts a word without a space token.
    let t = tokens(&[("д", 0), ("а", 40), ("н", 25_000), ("е", 25_040), ("т", 25_080)]);
    assert_eq!(srt_of(&t, &[0, 2], 30_000), "1\n00:00:00,000 --> 00:00:26,580\nда нет\n\n");
  }

  #[test]
  fn words_of_one_frame_stay_in_a_cue() {
    let t = tokens(&[("▁Yes.", 400), ("▁Right", 400), ("▁now.", 600), ("▁Go", 3000)]);
    assert_eq!(
      srt_of(&t, &[], 4000),
      "1\n00:00:00,400 --> 00:00:02,100\nYes. Right now.\n\n2\n00:00:03,000 --> 00:00:04,000\nGo\n\n"
    );
  }

  #[test]
  fn segments_last_until_the_next() {
    // Whisper gives every token of a segment the segment's start.
    let t = tokens(&[(" Ask", 0), (" not.", 0), (" Ask", 0), (" what.", 0), (" Next", 5000)]);
    assert_eq!(
      srt_segments(&t, 6000),
      "1\n00:00:00,000 --> 00:00:05,000\nAsk not. Ask what.\n\n2\n00:00:05,000 --> 00:00:06,000\nNext\n\n"
    );
    // Without spaces, and a one-token segment.
    let t = tokens(&[("我们", 0), ("走吧。", 0), ("好", 2000)]);
    assert_eq!(
      srt_segments(&t, 3000),
      "1\n00:00:00,000 --> 00:00:02,000\n我们走吧。\n\n2\n00:00:02,000 --> 00:00:03,000\n好\n\n"
    );
    // SentencePiece pieces (Moonshine).
    let t = tokens(&[("▁And", 0), ("▁so", 0), ("▁ask", 4000)]);
    assert_eq!(
      srt_segments(&t, 5000),
      "1\n00:00:00,000 --> 00:00:04,000\nAnd so\n\n2\n00:00:04,000 --> 00:00:05,000\nask\n\n"
    );
  }

  #[test]
  fn vtt_times_and_header() {
    let t = tokens(&[("▁Hel", 0), ("lo.", 80), ("▁Bye", 3_600_500)]);
    assert_eq!(
      vtt(&cues(&t, &[], 3_601_000, true)),
      "WEBVTT\n\n00:00:00.000 --> 00:00:01.580\nHello.\n\n01:00:00.500 --> 01:00:01.000\nBye\n\n"
    );
    assert_eq!(vtt(&[]), "WEBVTT\n\n");
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
       \n  {\"id\": 0, \"seek\": 0, \"start\": 0.0, \"end\": 1.9, \"text\": \"Hello you.\", \"tokens\": [0, 1, 2], \"temperature\": 0.0},\
       \n  {\"id\": 1, \"seek\": 0, \"start\": 4.0, \"end\": 4.5, \"text\": \"Go\", \"tokens\": [3], \"temperature\": 0.0}\
       \n], \"words\": [\
       \n  {\"word\": \"Hello\", \"start\": 0.0, \"end\": 0.4},\
       \n  {\"word\": \"you.\", \"start\": 0.4, \"end\": 1.9},\
       \n  {\"word\": \"Go\", \"start\": 4.0, \"end\": 4.5}\
       \n]}"
    );
    // Segment times only: no words.
    let cues = super::cues(&t.tokens, &[], 4500, false);
    let json = verbose_json(&t, &cues, "", 4500);
    assert!(json.contains("\"language\": \"\"") && !json.contains("words"), "{json}");
    assert!(json.ends_with("\"temperature\": 0.0}\n]}"), "{json}");
    let empty = verbose_json(&Transcript::default(), &[], "", 40);
    assert!(empty.ends_with("\"duration\": 0.04, \"text\": \"\", \"segments\": []}"), "{empty}");
  }

  /// Cue texts give each fixture's transcript text, up to whitespace: Chinese segments join
  /// without spaces.
  #[test]
  fn cues_keep_the_fixture_text() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for file in std::fs::read_dir(fixtures).unwrap() {
      let fixture = std::fs::read_to_string(file.unwrap().path()).unwrap();
      // A block per clip: a `## <clip>` header, then body lines, none of which starts with `## `.
      for block in fixture.strip_prefix("## ").unwrap().split("\n## ") {
        let mut lines = block.lines().skip(1).filter(|l| !l.is_empty());
        let text = lines.next().unwrap().strip_prefix("text: ").unwrap().replace("(empty)", "");
        // A block of text alone has no pieces to check.
        let Some(head) = lines.next() else { continue };
        let segments = head.starts_with("segments: ");
        // `  [   0.96 ->    1.00] p=0.000 ▁В`, with `▁` sometimes printed as a space, or a
        // Whisper segment `  [   0.00 ->   10.40] text`, one piece here.
        let pieces: Vec<String> = (lines.map(|l| l.split_once("] ").unwrap().1))
          .map(|r| {
            if segments {
              format!(" {r}")
            } else {
              r.trim_start().split_once(' ').unwrap().1.into()
            }
          })
          .collect();
        let pieces: Vec<(&str, u32)> = pieces.iter().map(|p| (p.as_str(), 0)).collect();
        let cues = cues(&tokens(&pieces), &[], 1000, true);
        let texts: Vec<&str> = cues.iter().map(|c| c.text.as_str()).collect();
        let squeeze = |s: &str| s.split_whitespace().collect::<String>();
        assert_eq!(squeeze(&texts.join(" ")), squeeze(&text));
      }
    }
  }
}
