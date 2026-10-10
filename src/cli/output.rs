// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The `--json` and `--srt` renderings of a transcript.

use std::fmt::Write;

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

fn quote(s: &str) -> String {
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

/// Subtitles: words from pieces (`▁` or a space starts a word, and so does each token index in
/// `starts`, sorted) grouped into cues that end after sentence punctuation or before passing
/// `MAX_CUE` characters. A cue shows until the next one starts, the audio ends or `LINGER_MS`
/// after its last token, whichever is first.
pub fn srt(tokens: &[Token], starts: &[usize], audio_ms: u32) -> String {
  let mut words: Vec<(u32, u32, String)> = Vec::new(); // first and last token start, text
  for (i, t) in tokens.iter().enumerate() {
    let piece = t.piece.replace('▁', " ");
    match words.last_mut() {
      Some(word) if !piece.starts_with(' ') && starts.binary_search(&i).is_err() => {
        word.1 = t.start_ms;
        word.2 += &piece;
      }
      _ => words.push((t.start_ms, t.start_ms, piece)),
    }
  }
  // Start, last word's start, text, and whether it is a timed segment (Whisper) rather than words.
  let mut cues: Vec<(u32, u32, String, bool)> = Vec::new();
  for (first, last, text) in &words {
    let text = text.trim();
    match cues.last_mut() {
      _ if text.is_empty() => {}
      // Words that share the cue's start (a Whisper segment) stay in it: a cut there would give the
      // cue before an empty interval.
      Some(cue) if *first == cue.0 => (cue.2, cue.3) = (format!("{} {text}", cue.2), true),
      Some(cue)
        if !cue.2.ends_with(['.', '!', '?', '…'])
          && cue.2.chars().count() + 1 + text.chars().count() <= MAX_CUE =>
      {
        (cue.1, cue.2) = (*last, format!("{} {text}", cue.2));
      }
      _ => cues.push((*first, *last, text.to_string(), false)),
    }
  }
  let time = |ms: u32| {
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
  };
  let mut out = String::new();
  for (i, (start, last, text, segment)) in cues.iter().enumerate() {
    // A segment lasts until the next one; words linger a little after the last one starts.
    let next = cues.get(i + 1).map_or(audio_ms, |c| c.0);
    let end = if *segment { next } else { next.min(last + LINGER_MS) }.max(*start);
    write!(out, "{}\n{} --> {}\n{text}\n\n", i + 1, time(*start), time(end)).unwrap();
  }
  out
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use super::*;

  fn tokens(pieces: &[(&str, u32)]) -> Vec<Token> {
    let token = |(i, &(piece, start_ms)): (usize, &(&str, u32))| Token {
      id: i as u32,
      piece: piece.to_string(),
      start_ms,
    };
    pieces.iter().enumerate().map(token).collect()
  }

  #[test]
  fn json_escapes() {
    let t = Transcript { text: "a \"b\"\\\n\u{1}é".into(), tokens: tokens(&[("▁a", 40)]) };
    let expected = "{\"text\": \"a \\\"b\\\"\\\\\\n\\u0001é\", \"tokens\": [\n  {\"id\": 0, \"piece\": \"▁a\", \"start_ms\": 40}\n]}";
    assert_eq!(json(&t), expected);
    assert_eq!(json(&Transcript::default()), "{\"text\": \"\", \"tokens\": []}");
  }

  #[test]
  fn cues_split_on_sentences_and_length() {
    let t = tokens(&[("▁Hel", 0), ("lo", 80), (".", 160), ("▁How", 4000), ("▁are", 4100)]);
    assert_eq!(
      srt(&t, &[], 6000),
      "1\n00:00:00,000 --> 00:00:01,660\nHello.\n\n2\n00:00:04,000 --> 00:00:05,600\nHow are\n\n"
    );
    let long: Vec<(String, u32)> = (0..30).map(|i| (format!("▁w{i:02}"), i * 100)).collect();
    let long: Vec<(&str, u32)> = long.iter().map(|(p, s)| (p.as_str(), *s)).collect();
    let srt = srt(&tokens(&long), &[], 3000);
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
    assert_eq!(srt(&t, &[], 3_700_000), "1\n01:00:00,000 --> 01:00:01,620\nда н\n\n");
    assert_eq!(srt(&[], &[], 1000), "");
    // A segment of a split recording starts a word without a space token.
    let t = tokens(&[("д", 0), ("а", 40), ("н", 25_000), ("е", 25_040), ("т", 25_080)]);
    assert_eq!(srt(&t, &[0, 2], 30_000), "1\n00:00:00,000 --> 00:00:26,580\nда нет\n\n");
  }

  #[test]
  fn segment_starts_keep_one_cue() {
    // Whisper gives every token of a segment the segment's start.
    let t = tokens(&[(" Ask", 0), (" not.", 0), (" Ask", 0), (" what.", 0), (" Next", 5000)]);
    assert_eq!(
      srt(&t, &[], 6000),
      "1\n00:00:00,000 --> 00:00:05,000\nAsk not. Ask what.\n\n2\n00:00:05,000 --> 00:00:06,000\nNext\n\n"
    );
  }

  /// Cue texts joined with spaces give each fixture's transcript text.
  #[test]
  fn cues_keep_the_fixture_text() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for model in std::fs::read_dir(fixtures).unwrap() {
      for file in std::fs::read_dir(model.unwrap().path()).unwrap() {
        let fixture = std::fs::read_to_string(file.unwrap().path()).unwrap();
        let mut lines = fixture.lines();
        let text = lines.next().unwrap().strip_prefix("text: ").unwrap().replace("(empty)", "");
        let segments = lines.next().is_some_and(|l| l.starts_with("segments: "));
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
        let srt = srt(&tokens(&pieces), &[], 1000);
        let cues: Vec<&str> = srt.split("\n\n").filter_map(|c| c.lines().nth(2)).collect();
        assert_eq!(cues.join(" "), text.split_whitespace().collect::<Vec<_>>().join(" "));
      }
    }
  }
}
