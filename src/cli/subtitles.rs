// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Subtitles of a transcript: its tokens cut into cues as Netflix's Timed Text Style Guide sets
//! them, as SubRip for `--srt` and `wavo serve`, and WebVTT for `wavo serve`; with `--diarize`,
//! no cue spans two speakers.

use std::fmt::Write;
use std::ops::Range;

use wavo::{Token, Utterance};

/// The width of a cue's line in characters, a wide one (Chinese, Japanese, Korean) counting two; a
/// cue has two lines.
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
/// Whisper's segments carry their start alone: their words are spread over them at this pace a
/// character's width, that of speech, faster when the next segment starts sooner.
const CHAR_MS: u32 = 70;
/// What ends a sentence, and what a cue may end with as well as one.
const ENDS: [char; 7] = ['.', '!', '?', '…', '。', '！', '？'];
const COMMAS: [char; 7] = [',', ';', ':', '、', '，', '；', '：'];

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

/// Words from pieces (`▁` or a space starts a word, and so does each token index in `starts` and
/// each token of Chinese or Japanese letters) grouped into phrases that end after a sentence or at
/// a pause, and those into cues of at most two lines and `MAX_MS` by `split`. A cue shows until
/// `LINGER_MS` after its last token, at least `MIN_MS`, and at most until `GAP_MS` before the next
/// one or the audio's end; a word until the next word starts or its cue ends. Without `pauses`
/// the token times are spread, not heard, and tell no pause.
fn word_cues(tokens: &[Token], starts: &[usize], audio_ms: u32, pauses: bool) -> Vec<Cue> {
  let mut words: Vec<Word> = Vec::new();
  for (i, t) in tokens.iter().enumerate() {
    let piece = t.piece.replace('▁', " ");
    let c = piece.chars().next().unwrap_or(' ');
    let prev = words.last().and_then(|w| w.text.chars().last()).unwrap_or(' ');
    // Chinese and Japanese have no spaces: a token of their letters starts a word of its own,
    // glued to the one before unless a space token came between.
    let glued = c.is_alphanumeric() && prev != ' ' && (unspaced(c) || unspaced(prev));
    match words.last_mut() {
      Some(word) if !glued && !piece.starts_with(' ') && starts.binary_search(&i).is_err() => {
        // Punctuation may come seconds after its word, at the end of a pause: not speech.
        if piece.chars().any(char::is_alphanumeric) {
          word.last = t.start_ms;
        }
        word.tokens.end = i + 1;
        word.text += &piece;
      }
      _ => words.push(Word {
        first: t.start_ms,
        last: t.start_ms,
        text: piece,
        tokens: i..i + 1,
        glued,
      }),
    }
  }

  words.iter_mut().for_each(|w| w.text = w.text.trim().to_string());
  words.retain(|w| !w.text.is_empty());

  // Phrases: a new one after a sentence or at a pause, when the one before has `MIN_MS` to show
  // and `GAP_MS` after (a shorter one, or words that start in one frame, share a cue).
  let mut phrases: Vec<Range<usize>> = Vec::new();
  for (i, w) in words.iter().enumerate() {
    let Some(p) = phrases.last_mut() else {
      phrases.push(i..i + 1);
      continue;
    };

    let prev = &words[i - 1];
    let end = prev.text.ends_with(ENDS) || pauses && w.first > prev.last + PAUSE_MS;
    if end && w.first >= words[p.start].first + MIN_MS + GAP_MS {
      phrases.push(i..i + 1);
    } else {
      p.end = i + 1;
    }
  }

  // Spread times end at the next segment: the last phrase needs `MIN_MS` before it too, else it
  // joins the one before.
  let last = |p: &[Range<usize>]| words[p[p.len() - 1].start].first;
  while !pauses && phrases.len() > 1 && last(&phrases) + MIN_MS > audio_ms {
    let p = phrases.pop().unwrap();
    phrases.last_mut().unwrap().end = p.end;
  }

  let mut groups: Vec<Range<usize>> = Vec::new();
  let nexts: Vec<u32> = (phrases.iter().skip(1))
    .map(|p| words[p.start].first)
    .chain([if pauses { u32::MAX } else { audio_ms.saturating_add(GAP_MS) }])
    .collect();
  for (p, next) in phrases.into_iter().zip(nexts) {
    split(&words, p, next, &mut groups);
  }

  let mut cues: Vec<Cue> = (groups.iter())
    .map(|g| {
      let words = &words[g.clone()];
      let mut text = String::new();
      for w in words {
        if !text.is_empty() && !w.glued {
          text.push(' ');
        }
        text += &w.text;
      }

      Cue {
        start: words[0].first,
        end: words[words.len() - 1].last,
        text,
        tokens: words[0].tokens.start..words[words.len() - 1].tokens.end,
        words: words.iter().map(|w| (w.first, w.first, w.text.clone())).collect(),
      }
    })
    .collect();

  let nexts: Vec<u32> =
    cues.iter().skip(1).map(|c| c.start.saturating_sub(GAP_MS)).chain([audio_ms]).collect();
  for (cue, next) in cues.iter_mut().zip(nexts) {
    cue.end = (cue.end + LINGER_MS).max(cue.start + MIN_MS).min(next).max(cue.end);
    let ends: Vec<u32> = cue.words.iter().skip(1).map(|w| w.0).chain([cue.end]).collect();
    cue.words.iter_mut().zip(ends).for_each(|(w, end)| w.1 = end);
  }

  cues
}

/// A word: its first and last spoken token start (not a punctuation one), text and tokens, and
/// whether it follows the one before without a space (Chinese, Japanese).
struct Word {
  first: u32,
  last: u32,
  text: String,
  tokens: Range<usize>,
  glued: bool,
}

/// The width of `words` in characters with the spaces between them.
fn chars(words: &[Word]) -> usize {
  (words.iter().enumerate()).map(|(k, w)| width(&w.text) + usize::from(k > 0 && !w.glued)).sum()
}

/// Whether `words` fit two lines of `LINE` characters, broken between words.
fn fits(words: &[Word]) -> bool {
  chars(words) <= LINE
    || (1..words.len()).any(|k| chars(&words[..k]).max(chars(&words[k..])) <= LINE)
}

/// The phrase `r` of `words`, followed by a word that starts at `next`, as cues into `out`: the
/// fewest cues of at most two lines and `MAX_MS`, of the most even length and time, cut after a
/// comma when one is near, and each with `MIN_MS` and `GAP_MS` before the next starts.
fn split(words: &[Word], r: Range<usize>, next: u32, out: &mut Vec<Range<usize>>) {
  let (a, n) = (r.start, r.len());
  let start = |j: usize| if j < n { words[a + j].first } else { next };
  // Per count of the phrase's first words: the cost of their best cues, and where the last starts.
  let mut best: Vec<(u64, usize)> = vec![(u64::MAX, 0); n + 1];
  best[0] = (0, 0);
  for j in 1..=n {
    for i in (0..j).rev() {
      let w = &words[a + i..a + j];
      let (chars, ms) = (chars(w) as u64, (w[w.len() - 1].last + LINGER_MS - w[0].first) as u64);
      if best[j].0 != u64::MAX && (ms > MAX_MS as u64 || !fits(w)) {
        break;
      }

      // The phrase rules leave the whole phrase room, unless it is all there is; a part may have
      // none.
      let whole = i == 0 && j == n;
      if best[i].0 == u64::MAX || !whole && start(j) < w[0].first + MIN_MS + GAP_MS {
        continue;
      }

      let cut = if i > 0 && !words[a + i - 1].text.ends_with(COMMAS) { 400 } else { 0 };
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
/// at `CHAR_MS` a character's width, or faster to fit before the next segment, then cut as
/// `word_cues` does. They have no words of their own. A segment that starts before the one before
/// it (Whisper may time a part's last segment past the part's end), or too soon for that one to
/// show `MIN_MS`, joins it.
fn segment_cues(tokens: &[Token], audio_ms: u32) -> Vec<Cue> {
  let mut segments: Vec<(u32, Range<usize>)> = Vec::new();
  for (i, t) in tokens.iter().enumerate() {
    segments.push((t.start_ms, i..i + 1));
    // A joined segment may start before the one before it in turn.
    while let [.., (a, r), (b, s)] = &mut segments[..]
      && *b < *a + MIN_MS + GAP_MS
    {
      (*a, r.end) = ((*b).min(*a), s.end);
      segments.pop();
    }
  }

  let mut cues = Vec::new();
  for (k, (start, r)) in segments.iter().enumerate() {
    let (start, next) = (*start, segments.get(k + 1).map_or(audio_ms, |n| n.0));
    let chars: usize = tokens[r.clone()].iter().map(|t| width(&t.piece)).sum();
    let pace = CHAR_MS.min(next.saturating_sub(start) / chars.max(1) as u32);
    let mut spread = tokens[r.clone()].to_vec();
    let mut at = 0;
    for t in &mut spread {
      t.start_ms = start + at * pace;
      at += width(&t.piece) as u32;
    }

    let end = if k + 1 < segments.len() { next.saturating_sub(GAP_MS).max(start) } else { next };
    for mut cue in word_cues(&spread, &[], end, false) {
      cue.tokens = cue.tokens.start + r.start..cue.tokens.end + r.start;
      cue.words.clear();
      cues.push(cue);
    }
  }

  cues
}

/// A cue's text in at most two lines of `LINE` characters where it fits: broken at the space, or
/// between Chinese or Japanese letters, that balances them best, after punctuation when nearly as
/// balanced.
fn lines(text: &str) -> String {
  let n = width(text);
  if n <= LINE {
    return text.into();
  }

  // Where each break's lines end and start: a space goes, letters stay.
  let chars: Vec<(usize, char)> = text.char_indices().collect();
  let breaks = chars.windows(2).filter_map(|p| match (p[0].1, p[1]) {
    (_, (i, ' ')) => Some((i, i + 1)),
    (a, (i, b)) if a != ' ' && b.is_alphanumeric() && (unspaced(a) || unspaced(b)) => Some((i, i)),
    _ => None,
  });
  // A line past `LINE` only when no break keeps both within it.
  let at = breaks.min_by_key(|&(i, j)| {
    let (left, right) = (width(&text[..i]), width(&text[j..]));
    let punct = text[..i].ends_with(ENDS) || text[..i].ends_with(COMMAS);
    (left.max(right) > LINE, left.abs_diff(right) + if punct { 0 } else { 12 })
  });
  match at {
    Some((i, j)) => format!("{}\n{}", &text[..i], &text[j..]),
    None => text.into(),
  }
}

/// The width of `text` in characters, a wide one (Chinese, Japanese, Korean) counting two.
fn width(text: &str) -> usize {
  text.chars().map(|c| if wide(c) { 2 } else { 1 }).sum()
}

/// Whether `c` is as wide as two characters: Chinese, Japanese, Korean and their punctuation.
fn wide(c: char) -> bool {
  matches!(c, '\u{1100}'..='\u{115f}' | '\u{2e80}'..='\u{a4cf}' | '\u{ac00}'..='\u{d7a3}'
    | '\u{f900}'..='\u{faff}' | '\u{fe30}'..='\u{fe4f}' | '\u{ff00}'..='\u{ff60}'
    | '\u{ffe0}'..='\u{ffe6}' | '\u{20000}'..='\u{3fffd}')
}

/// Whether `c` is a letter of a script written without spaces: Chinese and Japanese.
fn unspaced(c: char) -> bool {
  matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}'
    | '\u{f900}'..='\u{faff}' | '\u{20000}'..='\u{3fffd}')
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

/// `ms` as `hh:mm:ss` and milliseconds after `sep`.
pub fn time(ms: u32, sep: char) -> String {
  let (h, m, s) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60);
  format!("{h:02}:{m:02}:{s:02}{sep}{:03}", ms % 1000)
}

#[cfg(test)]
pub mod tests {
  use std::path::Path;

  use super::*;

  /// Tokens of `pieces` with their start times, numbered from 0.
  pub fn tokens(pieces: &[(&str, u32)]) -> Vec<Token> {
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
  fn cues_split_on_sentences_and_length() {
    let t = tokens(&[("▁Hel", 0), ("lo", 80), (".", 160), ("▁How", 4000), ("▁are", 4100)]);
    assert_eq!(
      srt_of(&t, &[], 6000),
      "1\n00:00:00,000 --> 00:00:01,080\nHello.\n\n2\n00:00:04,000 --> 00:00:05,100\nHow are\n\n"
    );
    // Punctuation at the end of a pause isn't speech: the cue ends a second after its word.
    let t = tokens(&[("▁them", 0), (".", 7000), ("▁If", 7300)]);
    assert_eq!(cues(&t, &[], 9000, true).iter().map(|c| c.end).collect::<Vec<_>>(), [1000, 8300]);
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
    // 82 characters that no break puts in two lines of 42: two cues.
    let (a, b, c) = (
      "▁".to_string() + &"a".repeat(30),
      "▁".to_string() + &"b".repeat(30),
      "▁".to_string() + &"c".repeat(20),
    );
    assert_eq!(cues(&tokens(&[(&a, 0), (&b, 600), (&c, 1200)]), &[], 4000, true).len(), 2);
    // At most 7 s, and after a comma when one is near the middle.
    let c = cues(&words(20, 500), &[], 12_000, true);
    assert_eq!(c.iter().map(|c| c.words.len()).collect::<Vec<_>>(), [10, 10]);
    let mut t = words(24, 100);
    t[10].piece = "▁w10,".into();
    let c = cues(&t, &[], 3000, true);
    assert_eq!(c.iter().map(|c| c.words.len()).collect::<Vec<_>>(), [11, 13]);
    // Never between words of one start, whatever their length.
    assert_eq!(cues(&words(30, 0), &[], 3000, true).len(), 1);
    // A pause.
    let t = tokens(&[("▁So", 0), ("▁we", 200), ("▁go", 1500)]);
    assert_eq!(cues(&t, &[], 3000, true).len(), 2);
    // A pause or a sentence too soon for the cue before to show `MIN_MS`.
    let t = tokens(&[("▁Yes", 0), ("▁Okay", 720)]);
    assert_eq!(cues(&t, &[], 3000, true).len(), 1);
    let t = tokens(&[("▁Yes.", 0), ("▁Okay", 880)]);
    assert_eq!(cues(&t, &[], 3000, true).len(), 1);
    // Nor a cut within a phrase: 50 words, the last 30 in one frame.
    let t: Vec<(String, u32)> =
      (0..50).map(|i| (format!("▁w{i:02}"), (i * 100).min(2000))).collect();
    let c =
      cues(&tokens(&t.iter().map(|(p, s)| (p.as_str(), *s)).collect::<Vec<_>>()), &[], 9000, true);
    let starts: Vec<u32> = c.iter().map(|c| c.start).collect();
    assert!(starts.windows(2).all(|s| s[1] >= s[0] + MIN_MS + GAP_MS), "{starts:?}");
  }

  #[test]
  fn chinese_and_japanese_break_between_letters() {
    // 126 Chinese letters in one Whisper segment: cues of two lines of 21 wide letters.
    let text = "我们今天讨论的问题非常重要，".repeat(9);
    let pieces: Vec<String> = text.chars().map(String::from).collect();
    let t = tokens(&pieces.iter().map(|p| (p.as_str(), 0)).collect::<Vec<_>>());
    let c = cues(&t, &[], 30_000, false);
    assert!(c.len() > 1);
    assert_eq!(c.iter().map(|c| c.text.as_str()).collect::<String>(), text);
    for line in srt(&c).lines().filter(|l| l.chars().any(unspaced)) {
      assert!(width(line) <= LINE, "{line}");
    }
    // A break falls after the comma, never before punctuation, and Latin words keep their spaces.
    assert_eq!(lines(&"一二三四五六七八九十，".repeat(4)).lines().count(), 2);
    assert!(lines(&"一二三四五六七八九十，".repeat(4)).contains("，\n"));
    assert_eq!(lines("東京で Tower を見た"), "東京で Tower を見た");
    let t = tokens(&[("東京", 0), (" ", 0), ("大学", 0), ("で", 0)]);
    assert_eq!(cues(&t, &[], 1000, false)[0].text, "東京 大学で");
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
      "1\n00:00:00,000 --> 00:00:01,280\n我们走吧。\n\n2\n00:00:02,000 --> 00:00:03,000\n好\n\n"
    );
    // A segment that starts before the one before, or too soon after it, joins it.
    let t = tokens(&[
      (" Hi.", 0),
      (" Late.", 9000),
      (" So", 6000),
      (" Thanks.", 12_000),
      (" Go", 12_100),
    ]);
    assert_eq!(
      srt_segments(&t, 14_000),
      "1\n00:00:00,000 --> 00:00:01,000\nHi.\n\n2\n00:00:06,000 --> 00:00:07,420\nLate. So\n\n3\n00:00:12,000 --> 00:00:13,560\nThanks. Go\n\n"
    );
    // Joins chain: D joins C, and then B, which C started before.
    let t = tokens(&[(" A.", 0), (" B.", 5000), (" C.", 9000), (" D", 4500)]);
    let starts: Vec<u32> = cues(&t, &[], 12_000, false).iter().map(|c| c.start).collect();
    assert_eq!(starts, [0, 4500]);
    // A sentence too near the next segment to show joins the one before.
    let pieces = [" Well,", " that's", " the", " thing", " to", " do.", " I...", " Go"];
    let mut t = tokens(&pieces.map(|p| (p, 0)));
    t[7].start_ms = 2300;
    let texts: Vec<String> = cues(&t, &[], 4000, false).into_iter().map(|c| c.text).collect();
    assert_eq!(texts, ["Well, that's the thing to do. I...", "Go"]);
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
