// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The protobuf wire format, encode only: what Core ML's `Model.proto` and `MIL.proto` need
//! (from coremltools, see `mil.rs`).

/// A message being written. Nested messages are written into their own `Msg` and then appended
/// with their length.
#[derive(Default)]
pub struct Msg(pub Vec<u8>);

impl Msg {
  fn varint(&mut self, mut v: u64) {
    while v >= 0x80 {
      self.0.push(v as u8 | 0x80);
      v >>= 7;
    }

    self.0.push(v as u8);
  }

  fn key(&mut self, field: u32, wire: u8) {
    self.varint(((field as u64) << 3) | wire as u64);
  }

  /// An integer or enum field; negative values take 10 bytes, as protobuf sign-extends them.
  pub fn int(&mut self, field: u32, v: i64) -> &mut Self {
    self.key(field, 0);
    self.varint(v as u64);
    self
  }

  pub fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
    self.key(field, 2);
    self.varint(b.len() as u64);
    self.0.extend_from_slice(b);
    self
  }

  pub fn str(&mut self, field: u32, s: &str) -> &mut Self {
    self.bytes(field, s.as_bytes())
  }

  pub fn msg(&mut self, field: u32, f: impl FnOnce(&mut Msg)) -> &mut Self {
    let mut m = Msg::default();
    f(&mut m);
    self.bytes(field, &m.0)
  }

  /// A map entry: `key` 1, the message `value` 2.
  pub fn entry(&mut self, field: u32, key: &str, value: impl FnOnce(&mut Msg)) -> &mut Self {
    self.msg(field, |e| _ = e.str(1, key).msg(2, value))
  }

  /// A packed repeated integer field.
  pub fn ints(&mut self, field: u32, v: impl IntoIterator<Item = i64>) -> &mut Self {
    let mut m = Msg::default();
    v.into_iter().for_each(|v| m.varint(v as u64));
    self.bytes(field, &m.0)
  }
}

#[cfg(test)]
mod tests {
  use super::Msg;

  #[test]
  fn wire_format() {
    // Examples from the protobuf encoding guide: 150 in field 1, "testing" in field 2.
    assert_eq!(Msg::default().int(1, 150).0, [0x08, 0x96, 0x01]);
    assert_eq!(Msg::default().str(2, "testing").0, b"\x12\x07testing");
    assert_eq!(Msg::default().msg(3, |m| _ = m.int(1, 150)).0, [0x1a, 0x03, 0x08, 0x96, 0x01]);
    let packed = Msg::default().ints(4, [3, 270, 86942]).0.clone();
    assert_eq!(packed, [0x22, 0x06, 0x03, 0x8e, 0x02, 0x9e, 0xa7, 0x05]);
    let mut neg = vec![0x08];
    neg.extend([0xff; 9]);
    neg.push(0x01);
    assert_eq!(Msg::default().int(1, -1).0, neg);
    assert_eq!(
      Msg::default().entry(2, "k", |v| _ = v.int(1, 1)).0,
      b"\x12\x07\x0a\x01k\x12\x02\x08\x01"
    );
    // A field number past 15 takes a two-byte key.
    assert_eq!(Msg::default().int(502, 0).0, [0xb0, 0x1f, 0x00]);
  }
}
