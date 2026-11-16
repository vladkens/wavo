// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The renderings of a transcript: `--json` and `--srt`, and for `wavo serve` also WebVTT and
//! OpenAI's `verbose_json` and `diarized_json`; with `--diarize` by speaker, and a diarization
//! model's turns alone.

use std::fmt::Write;
use std::ops::Range;

use wavo::{Token, Transcript, Turn, Utterance};

// Cues as Netflix's Timed Text Style Guide sets them: two lines of 42 characters, 5/6 s to 7 s,
// two frames apart, shown about half a second past the speech.
/// The characters of a cue's line; a cue has two.
const LINE: usize = 42;
/// The shortest and the longest cue, in milliseconds.
const MIN_MS: u32 = 833;
const MAX_MS: u32 = 7000;
/// Between a cue's end and the next one's start: two frames at 24 fps.
const GAP_MS: u32 = 83;
/// A cue shows this long after its last token starts: the token, then half a second.
const LINGER_MS: u32 = 1000;
/// From a word's last token to the next word's start, a pause: a new cue.
const PAUSE_MS: u32 = 700;
/// Whisper's segments carry their start alone: their words are spread over them at this pace,
/// that of speech, faster when the next segment starts sooner.
const CHAR_MS: u32 = 70;

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

/// The cues of each utterance on its own, so that none spans two speakers, shown at most until
/// the next utterance starts.
pub fn speaker_cues(
  tokens: &[Token],
  starts: &[usize],
  utterances: &[Utterance],
  audio_ms: u32,
  timed: bool,
) -> Vec<Cue> {
  let mut out = Vec::new();
  for (i, u) in utterances.iter().enumerate() {
    let r = u.tokens.clone();
    let end = utterances.get(i + 1).map_or(audio_ms, |n| n.start_ms);
    let within: Vec<usize> =
      starts.iter().filter(|&&s| r.contains(&s)).map(|s| s - r.start).collect();
    for mut cue in cues(&tokens[r.clone()], &within, end, timed) {
      cue.tokens = cue.tokens.start + r.start..cue.tokens.end + r.start;
      out.push(cue);
    }
  }

  out
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
  if timed { word_cues(tokens, starts, audio_ms, true) } else { segment_cues(tokens, audio_ms) }
}

/// Words from pieces (`▁` or a space starts a word, and so does each token index in `starts`)
/// grouped into phrases that end after a sentence (that lasted `MIN_MS`) or at a pause, and those
/// into cues of at most two lines and `MAX_MS` by `split`. A cue shows until
/// `LINGER_MS` after its last token, at least `MIN_MS`, and at most until `GAP_MS` before the next
/// one or the audio's end; a word until the next word starts or its cue ends. Without `pauses`
/// the token times are spread, not heard, and tell no pause.
fn word_cues(tokens: &[Token], starts: &[usize], audio_ms: u32, pauses: bool) -> Vec<Cue> {
  let mut words: Vec<Word> = Vec::new();
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

  words.iter_mut().for_each(|w| w.2 = w.2.trim().to_string());
  words.retain(|w| !w.2.is_empty());

  // Phrases: a new one after a sentence that lasted `MIN_MS` (a shorter one shares its cue with
  // the next) or at a pause. A word that starts with the phrase (several in one frame) stays in
  // it, else its cue would last no time at all.
  let mut phrases: Vec<Range<usize>> = Vec::new();
  for (i, (first, ..)) in words.iter().enumerate() {
    let Some(p) = phrases.last_mut() else {
      phrases.push(i..i + 1);
      continue;
    };

    let (prev, start) = (&words[i - 1], words[p.start].0);
    let sentence = prev.2.ends_with(['.', '!', '?', '…']) && *first >= start + MIN_MS;
    if *first != start && (sentence || pauses && *first > prev.1 + PAUSE_MS) {
      phrases.push(i..i + 1);
    } else {
      p.end = i + 1;
    }
  }

  let mut groups: Vec<Range<usize>> = Vec::new();
  phrases.into_iter().for_each(|p| split(&words, p, &mut groups));
  let mut cues: Vec<Cue> = (groups.iter())
    .map(|g| {
      let words = &words[g.clone()];
      let text: Vec<&str> = words.iter().map(|w| w.2.as_str()).collect();
      Cue {
        start: words[0].0,
        end: words[words.len() - 1].1,
        text: text.join(" "),
        tokens: words[0].3.start..words[words.len() - 1].3.end,
        words: words.iter().map(|w| (w.0, w.0, w.2.clone())).collect(),
      }
    })
    .collect();

  let nexts: Vec<u32> =
    cues.iter().skip(1).map(|c| c.start.saturating_sub(GAP_MS)).chain([audio_ms]).collect();
  for (cue, next) in cues.iter_mut().zip(nexts) {
    cue.end = (cue.end + LINGER_MS).max(cue.start + MIN_MS).min(next).max(cue.start);
    let ends: Vec<u32> = cue.words.iter().skip(1).map(|w| w.0).chain([cue.end]).collect();
    cue.words.iter_mut().zip(ends).for_each(|(w, end)| w.1 = end);
  }

  cues
}

/// A word's first and last token start, text and tokens.
type Word = (u32, u32, String, Range<usize>);

/// The characters of `words` with a space between each two.
fn chars(words: &[Word]) -> usize {
  words.iter().map(|w| w.2.chars().count() + 1).sum::<usize>().saturating_sub(1)
}

/// The phrase `r` of `words` as cues into `out`: the fewest cues of at most two lines and `MAX_MS`,
/// of the most even length and time, cut after a comma when one is near, and between words of one
/// frame only when nothing else fits.
fn split(words: &[Word], r: Range<usize>, out: &mut Vec<Range<usize>>) {
  let (a, n) = (r.start, r.len());
  // Per count of the phrase's first words: the cost of their best cues, and where the last starts.
  let mut best: Vec<(u64, usize)> = vec![(u64::MAX, 0); n + 1];
  best[0] = (0, 0);
  for j in 1..=n {
    for i in (0..j).rev() {
      let w = &words[a + i..a + j];
      let (chars, ms) = (chars(w) as u64, (w[w.len() - 1].1 + LINGER_MS - w[0].0) as u64);
      if j - i > 1 && (chars > 2 * LINE as u64 || ms > MAX_MS as u64) {
        break;
      }

      let cut = match i {
        0 => 0,
        _ if words[a + i].0 == words[a + i - 1].0 => 1 << 40,
        _ if words[a + i - 1].2.ends_with([',', ';', ':']) => 0,
        _ => 400,
      };
      // A cue costs more than any evening of the others; then the squares of its characters and
      // of its time at a character's pace.
      let cost = best[i].0 + (1 << 32) + chars * chars + ms * ms / 6400 + cut;
      if cost < best[j].0 {
        best[j] = (cost, i);
      }
    }
  }

  let mut cues = Vec::new();
  let mut j = n;
  while j > 0 {
    cues.push(a + best[j].1..a + j);
    j = best[j].1;
  }
  out.extend(cues.into_iter().rev());
}

/// The cues of a model that times segments, not tokens (Whisper: every token carries its
/// segment's start; Moonshine: 0 in each piece of the audio): each segment's words spread over it
/// at `CHAR_MS` a character, or faster to fit before the next segment, then cut as `word_cues`
/// does. They have no words of their own.
fn segment_cues(tokens: &[Token], audio_ms: u32) -> Vec<Cue> {
  let mut segments: Vec<Range<usize>> = Vec::new();
  for (i, t) in tokens.iter().enumerate() {
    match segments.last_mut() {
      Some(s) if tokens[s.start].start_ms == t.start_ms => s.end = i + 1,
      _ => segments.push(i..i + 1),
    }
  }

  let mut cues = Vec::new();
  for (k, r) in segments.iter().enumerate() {
    let start = tokens[r.start].start_ms;
    let next = segments.get(k + 1).map_or(audio_ms, |n| tokens[n.start].start_ms);
    let chars: usize = tokens[r.clone()].iter().map(|t| t.piece.chars().count()).sum();
    let pace = CHAR_MS.min(next.saturating_sub(start) / chars.max(1) as u32);
    let mut spread = tokens[r.clone()].to_vec();
    let mut at = 0;
    for t in &mut spread {
      t.start_ms = start + at * pace;
      at += t.piece.chars().count() as u32;
    }

    for mut cue in word_cues(&spread, &[], next, false) {
      cue.tokens = cue.tokens.start + r.start..cue.tokens.end + r.start;
      cue.words.clear();
      cues.push(cue);
    }
  }

  cues
}

/// A cue's text in at most two lines of `LINE` characters where it fits: broken at the space that
/// balances them best, after punctuation when nearly as balanced.
fn lines(text: &str) -> String {
  let n = text.chars().count();
  if n <= LINE {
    return text.into();
  }

  // A line past `LINE` only when no break keeps both within it.
  let at = (text.char_indices().filter(|&(_, c)| c == ' ').map(|(i, _)| i)).min_by_key(|&i| {
    let left = text[..i].chars().count();
    let punct = text[..i].ends_with([',', '.', ';', ':', '!', '?', '…']);
    let over = left.max(n - 1 - left) > LINE;
    (over, left.abs_diff(n - 1 - left) + if punct { 0 } else { 12 })
  });
  match at {
    Some(i) => format!("{}\n{}", &text[..i], &text[i + 1..]),
    None => text.into(),
  }
}

/// SubRip subtitles: numbered cues with times like `00:00:01,500`.
pub fn srt(cues: &[Cue]) -> String {
  let mut out = String::new();
  for (i, c) in cues.iter().enumerate() {
    let (start, end) = (time(c.start, ','), time(c.end, ','));
    write!(out, "{}\n{start} --> {end}\n{}\n\n", i + 1, lines(&c.text)).unwrap();
  }
  out
}

/// WebVTT subtitles: a `WEBVTT` header, then cues with times like `00:00:01.500` and `&`, `<`
/// and `>` escaped, as WebVTT reads markup in cue text.
pub fn vtt(cues: &[Cue]) -> String {
  let mut out = String::from("WEBVTT\n\n");
  for c in cues {
    let text = lines(&c.text).replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    write!(out, "{} --> {}\n{text}\n\n", time(c.start, '.'), time(c.end, '.')).unwrap();
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
    assert_eq!(json(&t, None), expected);
    assert_eq!(json(&Transcript::default(), None), "{\"text\": \"\", \"tokens\": []}");
  }

  #[test]
  fn cues_split_on_sentences_and_length() {
    let t = tokens(&[("▁Hel", 0), ("lo", 80), (".", 160), ("▁How", 4000), ("▁are", 4100)]);
    assert_eq!(
      srt_of(&t, &[], 6000),
      "1\n00:00:00,000 --> 00:00:01,160\nHello.\n\n2\n00:00:04,000 --> 00:00:05,100\nHow are\n\n"
    );
    let words = |n: u32, ms: u32| -> Vec<Token> {
      let pieces: Vec<(String, u32)> = (0..n).map(|i| (format!("▁w{i:02}"), i * ms)).collect();
      tokens(&pieces.iter().map(|(p, s)| (p.as_str(), *s)).collect::<Vec<_>>())
    };
    // 30 words of 3 characters and their spaces need two cues of even length; a cue ends two
    // frames before the next, the last with the audio, and its shorter line comes first.
    let c = cues(&words(30, 100), &[], 3000, true);
    let texts: Vec<usize> = c.iter().map(|c| c.text.len()).collect();
    assert_eq!((texts, c[0].end, c[1].end), (vec![59, 59], 1417, 3000));
    assert_eq!(srt(&c[..1]).lines().nth(2), Some("w00 w01 w02 w03 w04 w05 w06"));
    // At most 7 s, and after a comma when one is near the middle.
    let c = cues(&words(20, 500), &[], 12_000, true);
    assert_eq!(c.iter().map(|c| c.words.len()).collect::<Vec<_>>(), [10, 10]);
    let mut t = words(24, 100);
    t[10].piece = "▁w10,".into();
    let c = cues(&t, &[], 3000, true);
    assert_eq!(c.iter().map(|c| c.words.len()).collect::<Vec<_>>(), [11, 13]);
    // A pause.
    let t = tokens(&[("▁So", 0), ("▁we", 200), ("▁go", 1500)]);
    assert_eq!(cues(&t, &[], 3000, true).len(), 2);
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
    assert_eq!(srt_of(&t, &[], 3_700_000), "1\n01:00:00,000 --> 01:00:01,120\nда н\n\n");
    assert_eq!(srt_of(&[], &[], 1000), "");
    // A segment of a split recording starts a word without a space token.
    let t = tokens(&[("д", 0), ("а", 40), ("н", 25_000), ("е", 25_040), ("т", 25_080)]);
    assert_eq!(
      srt_of(&t, &[0, 2], 30_000),
      "1\n00:00:00,000 --> 00:00:01,040\nда\n\n2\n00:00:25,000 --> 00:00:26,080\nнет\n\n"
    );
  }

  #[test]
  fn words_of_one_frame_stay_in_a_cue() {
    let t = tokens(&[("▁Yes.", 400), ("▁Right", 400), ("▁now.", 600), ("▁Go", 3000)]);
    assert_eq!(
      srt_of(&t, &[], 4000),
      "1\n00:00:00,400 --> 00:00:01,600\nYes. Right now.\n\n2\n00:00:03,000 --> 00:00:04,000\nGo\n\n"
    );
  }

  #[test]
  fn segments_last_until_the_next() {
    // Whisper gives every token of a segment the segment's start.
    let t = tokens(&[(" Ask", 0), (" not.", 0), (" Ask", 0), (" what.", 0), (" Next", 5000)]);
    assert_eq!(
      srt_segments(&t, 6000),
      "1\n00:00:00,000 --> 00:00:01,910\nAsk not. Ask what.\n\n2\n00:00:05,000 --> 00:00:06,000\nNext\n\n"
    );
    // Without spaces, and a one-token segment.
    let t = tokens(&[("我们", 0), ("走吧。", 0), ("好", 2000)]);
    assert_eq!(
      srt_segments(&t, 3000),
      "1\n00:00:00,000 --> 00:00:01,140\n我们走吧。\n\n2\n00:00:02,000 --> 00:00:03,000\n好\n\n"
    );
    // SentencePiece pieces (Moonshine).
    let t = tokens(&[("▁And", 0), ("▁so", 0), ("▁ask", 4000)]);
    assert_eq!(
      srt_segments(&t, 5000),
      "1\n00:00:00,000 --> 00:00:01,280\nAnd so\n\n2\n00:00:04,000 --> 00:00:05,000\nask\n\n"
    );
  }

  #[test]
  fn vtt_times_and_header() {
    let t = tokens(&[("▁Hel", 0), ("lo.", 80), ("▁Bye", 3_600_500)]);
    assert_eq!(
      vtt(&cues(&t, &[], 3_601_000, true)),
      "WEBVTT\n\n00:00:00.000 --> 00:00:01.080\nHello.\n\n01:00:00.500 --> 01:00:01.000\nBye\n\n"
    );
    assert_eq!(vtt(&[]), "WEBVTT\n\n");
    let t = tokens(&[("▁a<b>&c", 0)]);
    let escaped = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\na&lt;b&gt;&amp;c\n\n";
    assert_eq!(vtt(&cues(&t, &[], 1000, true)), escaped);
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
    let cues = super::cues(&t.tokens, &[], 4500, false);
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
        let head = lines.next().unwrap();
        // A diarization model's speaker segments have no text.
        if head.starts_with("speaker segments: ") {
          continue;
        }
        let text = head.strip_prefix("text: ").unwrap().replace("(empty)", "");
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
