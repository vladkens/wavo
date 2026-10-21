// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The transcripts of long audio's pieces (cut by `wavo::split`) joined back into one.

use wavo::{Token, Transcript};

/// Joins the transcripts of pieces starting at the given samples: texts with a space, token times
/// shifted to the whole audio. Also returns the index of each piece's first token, where
/// subtitles start a new word.
pub fn join(parts: impl IntoIterator<Item = (usize, Transcript)>) -> (Transcript, Vec<usize>) {
  let (mut out, mut starts) = (Transcript::default(), Vec::new());
  for (start, part) in parts {
    starts.push(out.tokens.len());
    if !part.text.is_empty() {
      out.text += if out.text.is_empty() { "" } else { " " };
      out.text += &part.text;
    }
    let offset = (start / 16) as u32; // a multiple of 160 samples: whole ms
    let shift = |mut t: Token| {
      t.start_ms += offset;
      t
    };
    out.tokens.extend(part.tokens.into_iter().map(shift));
  }
  (out, starts)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn join_shifts_tokens_and_skips_empty_texts() {
    let token = |id, start_ms| {
      let mut t = Token::default();
      (t.id, t.piece, t.start_ms) = (id, format!("▁{id}"), start_ms);
      t
    };
    let part = |text: &str, tokens| {
      let mut t = Transcript::default();
      (t.text, t.tokens) = (text.into(), tokens);
      t
    };
    let (t, starts) = join([
      (0, part("a b", vec![token(1, 40), token(2, 80)])),
      (16 * 20_000, part("", vec![])),
      (16 * 30_010, part("c", vec![token(3, 0)])),
    ]);
    assert_eq!(t.text, "a b c");
    let times: Vec<(u32, u32)> = t.tokens.iter().map(|t| (t.id, t.start_ms)).collect();
    assert_eq!(times, [(1, 40), (2, 80), (3, 30_010)]);
    assert_eq!(starts, [0, 2, 2]);
    let one = part("x", vec![token(7, 120)]);
    assert_eq!(join([(0, one.clone())]).0, one, "a single piece from 0 is unchanged");
  }
}
