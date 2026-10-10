// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GPT-2 byte-level BPE, decoding only: each character of a vocabulary piece stands for one byte
//! (`bytes_to_unicode`), so a token is a byte string that may end inside a UTF-8 character.

use crate::error::{Result, bail};
use crate::gguf::Gguf;

pub struct Tokenizer {
  /// Token bytes, concatenated; token i is `bytes[ends[i - 1]..ends[i]]`.
  bytes: Vec<u8>,
  ends: Vec<usize>,
}

impl Tokenizer {
  pub fn new(g: &Gguf) -> Result<Self> {
    g.check("tokenizer.ggml.model", &["gpt2"])?;
    // GPT-2's bytes_to_unicode: printable bytes stand for themselves, the other 68 for U+0100 on.
    let mut byte = [None; 512];
    let mut next = 256;
    for b in 0..256usize {
      let printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
      let c = if printable { b } else { (next, next += 1).0 };
      byte[c] = Some(b as u8);
    }
    let (mut bytes, mut ends) = (Vec::new(), Vec::new());
    for piece in g.array::<&str>("tokenizer.ggml.tokens")? {
      for c in piece.chars() {
        // Other characters (special tokens) pass through as UTF-8, as the reference decodes them.
        match byte.get(c as usize).copied().flatten() {
          Some(b) => bytes.push(b),
          None => bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
        }
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

  pub fn bytes(&self, id: usize) -> &[u8] {
    &self.bytes[if id == 0 { 0 } else { self.ends[id - 1] }..self.ends[id]]
  }

  /// The id of the piece spelled `s`.
  pub fn find(&self, s: &str) -> Option<usize> {
    (0..self.len()).find(|&i| self.bytes(i) == s.as_bytes())
  }
}

/// Splits the bytes of `ids` into one piece per token: a character goes to the token that completes
/// it, so the pieces are valid UTF-8 (some empty) and concatenate to the decoded text. Bytes that
/// are not UTF-8 become U+FFFD.
pub fn pieces(tok: &Tokenizer, ids: &[usize]) -> Vec<String> {
  let mut pending = Vec::new();
  let mut out: Vec<String> = Vec::with_capacity(ids.len());
  for &id in ids {
    pending.extend_from_slice(tok.bytes(id));
    let done = match std::str::from_utf8(&pending) {
      Ok(_) => pending.len(),
      // An incomplete character at the end waits for the next token; anything else is invalid.
      Err(e) if e.error_len().is_none() => e.valid_up_to(),
      Err(_) => pending.len(),
    };
    out.push(String::from_utf8_lossy(&pending[..done]).into_owned());
    pending.drain(..done);
  }
  if let Some(last) = out.last_mut() {
    *last += &String::from_utf8_lossy(&pending);
  }
  out
}

/// Trims ASCII whitespace, as the reference does.
pub fn trim(s: &str) -> &str {
  s.trim_matches([' ', '\t', '\n', '\r'])
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn split_characters_go_to_the_token_that_completes_them() {
    // "日" is E6 97 A5, split over two tokens; then " a" and an invalid lone byte.
    let raw: [&[u8]; 4] = [b"\xe6\x97", b"\xa5", b" a", b"\xff"];
    let mut t = Tokenizer { bytes: Vec::new(), ends: Vec::new() };
    for r in raw {
      t.bytes.extend_from_slice(r);
      t.ends.push(t.bytes.len());
    }
    assert_eq!(pieces(&t, &[0, 1, 2]), ["", "日", " a"]);
    assert_eq!(pieces(&t, &[2, 0]), [" a", "\u{fffd}"]);
    assert_eq!(pieces(&t, &[3, 2]), ["\u{fffd}", " a"]);
    assert_eq!(t.find(" a"), Some(2));
    assert_eq!(trim("\t x \n"), "x");
  }
}
