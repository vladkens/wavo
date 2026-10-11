// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! SentencePiece BPE with byte fallback, decoding only, as transcribe.cpp's `decode_sentencepiece`
//! (`src/transcribe-tokenizer.cpp`): a piece `<0xHH>` is that raw byte, `▁` is a space, and control
//! pieces (`<s>`, `</s>`, `<<ST_N>>`) print as they are spelled.

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::whisper::tokenizer::pieces;

pub struct Tokenizer {
  /// Token bytes, concatenated: the piece's UTF-8, or the one byte of a byte token. Token i is
  /// `bytes[ends[i - 1]..ends[i]]`.
  bytes: Vec<u8>,
  ends: Vec<usize>,
}

impl Tokenizer {
  pub fn new(g: &Gguf) -> Result<Self> {
    g.check("tokenizer.ggml.model", &["bpe"])?;
    let (mut bytes, mut ends) = (Vec::new(), Vec::new());
    for piece in g.array::<&str>("tokenizer.ggml.tokens")? {
      let hex = piece.strip_prefix("<0x").and_then(|h| h.strip_suffix('>'));
      match hex.filter(|h| h.len() == 2).and_then(|h| u8::from_str_radix(h, 16).ok()) {
        Some(b) => bytes.push(b),
        None => bytes.extend_from_slice(piece.as_bytes()),
      }
      ends.push(bytes.len());
    }
    if ends.is_empty() {
      bail!("empty vocabulary");
    }
    Ok(Self { bytes, ends })
  }

  pub fn len(&self) -> usize {
    self.ends.len()
  }

  /// One piece per id: the vocabulary piece (`▁` kept), and for byte tokens the characters they
  /// complete (see `whisper::tokenizer::pieces`).
  pub fn pieces(&self, ids: &[usize]) -> Vec<String> {
    let bytes = |id| &self.bytes[if id == 0 { 0 } else { self.ends[id - 1] }..self.ends[id]];
    pieces(bytes, ids)
  }
}

/// The text of `pieces` without its one leading space, as the reference trims it. A byte token
/// never spells `▁`: SentencePiece has the piece.
pub fn text(pieces: &[String]) -> String {
  let text = pieces.concat().replace('▁', " ");
  text.strip_prefix(' ').map(str::to_string).unwrap_or(text)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn byte_fallback_and_spaces() {
    // "й" is D0 B9 over two byte tokens; a control token prints as spelled.
    let mut t = Tokenizer { bytes: Vec::new(), ends: Vec::new() };
    for piece in [&b"\xe2\x96\x81And"[..], b"\xe2\x96\x81so", b"\xd0", b"\xb9", b"</s>"] {
      t.bytes.extend_from_slice(piece);
      t.ends.push(t.bytes.len());
    }
    let pieces = t.pieces(&[0, 1, 2, 3, 4]);
    assert_eq!(pieces, ["▁And", "▁so", "", "й", "</s>"]);
    assert_eq!(text(&pieces), "And soй</s>");
    // Only one leading space goes.
    assert_eq!(text(&t.pieces(&[1, 1])), "so so");
  }
}
