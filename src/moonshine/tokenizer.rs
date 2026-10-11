// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! SentencePiece BPE with byte fallback, decoding only, as transcribe.cpp's `decode_sentencepiece`
//! (`src/transcribe-tokenizer.cpp`): a piece `<0xHH>` is that raw byte, `▁` in any other piece is a
//! space, and control pieces (`<s>`, `</s>`, `<<ST_N>>`) print as they are spelled.

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::whisper::tokenizer::pieces;

pub struct Tokenizer {
  /// Token bytes, concatenated: the piece's UTF-8, or the one byte of a byte token. Token i is
  /// `bytes[ends[i - 1]..ends[i]]`.
  bytes: Vec<u8>,
  ends: Vec<usize>,
  /// Byte tokens, whose byte is never a space.
  raw: Vec<bool>,
}

impl Tokenizer {
  pub fn new(g: &Gguf) -> Result<Self> {
    g.check("tokenizer.ggml.model", &["bpe"])?;
    let mut t = Self { bytes: Vec::new(), ends: Vec::new(), raw: Vec::new() };
    for piece in g.array::<&str>("tokenizer.ggml.tokens")? {
      let hex = piece.strip_prefix("<0x").and_then(|h| h.strip_suffix('>'));
      let byte = hex.filter(|h| h.len() == 2).and_then(|h| u8::from_str_radix(h, 16).ok());
      match byte {
        Some(b) => t.bytes.push(b),
        None => t.bytes.extend_from_slice(piece.as_bytes()),
      }
      t.ends.push(t.bytes.len());
      t.raw.push(byte.is_some());
    }
    if t.ends.is_empty() {
      bail!("empty vocabulary");
    }
    Ok(t)
  }

  pub fn len(&self) -> usize {
    self.ends.len()
  }

  fn bytes(&self, id: usize) -> &[u8] {
    &self.bytes[if id == 0 { 0 } else { self.ends[id - 1] }..self.ends[id]]
  }

  /// The text of `ids`, without its one leading space, as the reference trims it.
  pub fn text(&self, ids: &[usize]) -> String {
    let mut out = Vec::new();
    for &id in ids {
      let piece = String::from_utf8_lossy(self.bytes(id));
      match self.raw[id] {
        true => out.extend_from_slice(self.bytes(id)),
        false => out.extend_from_slice(piece.replace('▁', " ").as_bytes()),
      }
    }
    let text = String::from_utf8_lossy(&out);
    text.strip_prefix(' ').unwrap_or(&text).to_string()
  }

  /// One piece per id: the vocabulary piece (`▁` kept), and for byte tokens the characters they
  /// complete (see `whisper::tokenizer::pieces`).
  pub fn pieces(&self, ids: &[usize]) -> Vec<String> {
    pieces(|id| self.bytes(id), ids)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn byte_fallback_and_spaces() {
    // "й" is D0 B9 over two byte tokens; a control token prints as spelled.
    let mut t = Tokenizer { bytes: Vec::new(), ends: Vec::new(), raw: Vec::new() };
    for (piece, raw) in [("▁And", false), ("▁so", false), ("\u{d0}", true), ("\u{b9}", true)] {
      let bytes: &[u8] =
        if raw { &[piece.chars().next().unwrap() as u8] } else { piece.as_bytes() };
      t.bytes.extend_from_slice(bytes);
      t.ends.push(t.bytes.len());
      t.raw.push(raw);
    }
    t.bytes.extend_from_slice(b"</s>");
    t.ends.push(t.bytes.len());
    t.raw.push(false);
    assert_eq!(t.text(&[0, 1, 2, 3, 4]), "And soй</s>");
    assert_eq!(t.pieces(&[0, 1, 2, 3, 4]), ["▁And", "▁so", "", "й", "</s>"]);
    // Only one leading space goes.
    assert_eq!(t.text(&[1, 1]), "so so");
  }
}
