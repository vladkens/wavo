// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Speaker turns and the speaker-attributed transcript.

use std::ops::Range;

use crate::Transcript;

/// The longest a word is taken to last, in milliseconds, when the next one starts later or there
/// is none.
const WORD_MS: u32 = 1000;

/// A stretch of audio where one speaker talks, from
/// [`Diarizer::diarize`](crate::Diarizer::diarize). Two speakers' turns overlap where they talk at
/// once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Turn {
  /// The speaker, from 0: speakers are numbered in the order they first talk in the audio, so the
  /// same voice has the same number throughout one call, but not across calls.
  pub speaker: u32,
  /// Start in milliseconds from the start of the audio, a multiple of the model's 80 ms frame.
  pub start_ms: u32,
  /// End in milliseconds, a multiple of 80 after `start_ms`.
  pub end_ms: u32,
}

/// Consecutive words of one speaker in a [`Transcript`], from [`Transcript::utterances`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Utterance {
  /// The speaker as in [`Turn::speaker`]; `None` only when there were no turns at all.
  pub speaker: Option<u32>,
  /// Where its first word starts, in milliseconds from the start of the audio.
  pub start_ms: u32,
  /// Where its last word ends, in milliseconds: when the next word starts, at most 1 s after the
  /// last word starts.
  pub end_ms: u32,
  /// Its words, as in [`Transcript::text`].
  pub text: String,
  /// Its tokens: a range of [`Transcript::tokens`].
  pub tokens: Range<usize>,
}

impl Transcript {
  /// The transcript cut into utterances by speaker, given the [`Turn`]s of the same audio: each
  /// word goes to the speaker whose turns overlap it most, from the start of its first token that
  /// is more than a space to the next word's start (at most 1 s), or when none does to the speaker
  /// of the nearest turn;
  /// consecutive words of a speaker make an utterance. The utterances hold every token, in order.
  ///
  /// A word starts at a token whose piece starts with `▁` or a space, or that starts more than 1 s
  /// after the token before; words that start together share their span. The attribution is only
  /// as fine as the token times: per word for Parakeet and GigaAM; per segment for Whisper, whose
  /// tokens carry their segment's start; none for Moonshine, whose tokens all start at 0. An
  /// utterance's text is its tokens' pieces, `▁` as a space, trimmed.
  ///
  /// # Examples
  ///
  /// ```no_run
  /// let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
  /// let diarizer = wavo::Diarizer::load("diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf")?;
  /// # let pcm = vec![0.0f32; 16_000];
  /// let transcript = model.transcribe(&pcm)?;
  /// for u in transcript.utterances(&diarizer.diarize(&pcm)?) {
  ///   println!("Speaker {}: {}", u.speaker.map_or(0, |s| s + 1), u.text);
  /// }
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn utterances(&self, turns: &[Turn]) -> Vec<Utterance> {
    let tokens = &self.tokens;
    // A pause starts a word too: between Whisper's segments of text without spaces.
    let firsts: Vec<usize> = (0..tokens.len())
      .filter(|&i| {
        i == 0
          || tokens[i].piece.starts_with(['▁', ' '])
          || tokens[i].start_ms > tokens[i - 1].start_ms.saturating_add(WORD_MS)
      })
      .collect();
    // Per word its tokens and start: that of its first token with more than a space (a space token
    // of a character vocabulary comes right after the word before).
    let words: Vec<(Range<usize>, u32)> = (firsts.iter().enumerate())
      .map(|(k, &first)| {
        let end = firsts.get(k + 1).copied().unwrap_or(tokens.len());
        let blank = |i: &usize| tokens[*i].piece.trim_matches([' ', '▁']).is_empty();
        let shown = (first..end).find(|i| !blank(i)).unwrap_or(first);
        (first..end, tokens[shown].start_ms)
      })
      .collect();

    // Each word lasts until the next start after its own: words that start together (a Whisper
    // segment's) share their span, and so their speaker.
    let mut ends = vec![0; words.len()];
    let mut later = u32::MAX;
    for k in (0..words.len()).rev() {
      let start = words[k].1;
      if let Some(next) = words.get(k + 1).filter(|w| w.1 != start) {
        later = next.1;
      }

      ends[k] = later.min(start.saturating_add(WORD_MS)).max(start + 1);
    }

    let mut out: Vec<Utterance> = Vec::new();
    for ((range, start), end) in words.into_iter().zip(ends) {
      let speaker = speaker(turns, start, end);
      match out.last_mut() {
        Some(u) if u.speaker == speaker => (u.end_ms, u.tokens.end) = (end, range.end),
        _ => out.push(Utterance {
          speaker,
          start_ms: start,
          end_ms: end,
          tokens: range,
          ..Default::default()
        }),
      }
    }

    for u in &mut out {
      let text: String =
        tokens[u.tokens.clone()].iter().map(|t| t.piece.replace('▁', " ")).collect();
      u.text = text.trim().to_string();
    }

    out
  }
}

/// The speaker whose turns overlap `start..end` most (the lower one on a tie), else the speaker of
/// the nearest turn.
fn speaker(turns: &[Turn], start: u32, end: u32) -> Option<u32> {
  let overlap = |t: &Turn| end.min(t.end_ms).saturating_sub(start.max(t.start_ms));
  let gap = |t: &Turn| t.start_ms.saturating_sub(end).max(start.saturating_sub(t.end_ms));
  let mut by: Vec<(u32, u32)> = Vec::new();
  for t in turns {
    match by.iter_mut().find(|b| b.0 == t.speaker) {
      Some(b) => b.1 += overlap(t),
      None => by.push((t.speaker, overlap(t))),
    }
  }

  let best = by.iter().filter(|b| b.1 > 0).min_by_key(|b| (std::cmp::Reverse(b.1), b.0));
  best.map(|b| b.0).or_else(|| turns.iter().min_by_key(|t| (gap(t), t.speaker)).map(|t| t.speaker))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Token;

  fn transcript(pieces: &[(&str, u32)]) -> Transcript {
    let token = |&(piece, start_ms): &(&str, u32)| {
      let mut t = Token::default();
      (t.piece, t.start_ms) = (piece.to_string(), start_ms);
      t
    };
    Transcript { tokens: pieces.iter().map(token).collect(), ..Default::default() }
  }

  fn turn(speaker: u32, start_ms: u32, end_ms: u32) -> Turn {
    Turn { speaker, start_ms, end_ms }
  }

  fn said(t: &Transcript, turns: &[Turn]) -> Vec<(Option<u32>, String, u32, u32)> {
    t.utterances(turns).into_iter().map(|u| (u.speaker, u.text, u.start_ms, u.end_ms)).collect()
  }

  #[test]
  fn words_follow_turns() {
    let t = transcript(&[
      ("▁Hel", 0),
      ("lo", 80),
      ("▁there.", 400),
      ("▁Hi", 2000),
      ("!", 2080),
      ("▁Bye", 5000),
    ]);
    let turns = [turn(0, 0, 1200), turn(1, 1900, 2500), turn(0, 4800, 5600)];
    let u = |s, text: &str, a, b| (Some(s), text.to_string(), a, b);
    assert_eq!(
      said(&t, &turns),
      [u(0, "Hello there.", 0, 1400), u(1, "Hi!", 2000, 3000), u(0, "Bye", 5000, 6000)]
    );
    let all = t.utterances(&turns);
    assert_eq!(all.iter().map(|u| u.tokens.clone()).collect::<Vec<_>>(), [0..3, 3..5, 5..6]);

    // A word goes to the speaker it overlaps longer; in a pause to the nearest turn.
    let turns = [turn(1, 0, 300), turn(0, 300, 2000), turn(1, 1900, 2500)];
    assert_eq!(
      said(&t, &turns),
      [u(1, "Hello", 0, 400), u(0, "there.", 400, 1400), u(1, "Hi! Bye", 2000, 6000)]
    );
    assert_eq!(said(&t, &[]), [(None, "Hello there. Hi! Bye".into(), 0, 6000)]);
    assert!(Transcript::default().utterances(&turns).is_empty());
  }

  #[test]
  fn segments_and_spaces() {
    // Whisper: a segment's tokens share its start, and its words its speaker.
    let t = transcript(&[(" A", 0), (" b", 0), (" c.", 0), (" Next", 3000)]);
    let turns = [turn(0, 0, 700), turn(1, 700, 4000)];
    let got: Vec<_> = said(&t, &turns).into_iter().map(|u| (u.0, u.1)).collect();
    assert_eq!(got, [(Some(0), "A b c.".into()), (Some(1), "Next".into())]);
    // A character vocabulary: a space token starts each word, timed by its first letter.
    let t = transcript(&[("д", 0), ("а", 40), (" ", 80), ("н", 2000), ("е", 2040), ("т", 2080)]);
    let turns = [turn(0, 0, 500), turn(1, 1800, 2500)];
    let got: Vec<_> = said(&t, &turns).into_iter().map(|u| (u.0, u.1)).collect();
    assert_eq!(got, [(Some(0), "да".into()), (Some(1), "нет".into())]);
    assert_eq!(said(&t, &[turn(0, 0, 3000)])[0].1, "да нет");
    // Whisper's segments of text without spaces: a pause starts a word.
    let t = transcript(&[("我们", 0), ("走吧。", 0), ("好", 2000)]);
    let turns = [turn(0, 0, 1500), turn(1, 1900, 3000)];
    let got: Vec<_> = said(&t, &turns).into_iter().map(|u| (u.0, u.1)).collect();
    assert_eq!(got, [(Some(0), "我们走吧。".into()), (Some(1), "好".into())]);
  }
}
