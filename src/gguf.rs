// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GGUF v3 reader: metadata from the header, F32/F16/Q8_0/Q5_0 tensors streamed from the file.

use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::f16;

use crate::error::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
  F32,
  F16,
  /// Blocks of 32 weights: f16 scale + 32 × i8 (34 bytes).
  Q8_0,
  /// Blocks of 32 weights: f16 scale, 4 high-bit bytes, 16 low nibbles (22 bytes). A weight is
  /// `(code - 16) · scale`, with `code` in 0..31.
  Q5_0,
}

impl Type {
  fn from_ggml(id: u32) -> Option<Self> {
    match id {
      0 => Some(Self::F32),
      1 => Some(Self::F16),
      6 => Some(Self::Q5_0),
      8 => Some(Self::Q8_0),
      _ => None,
    }
  }

  fn size(self, n: usize) -> usize {
    match self {
      Self::F32 => n * 4,
      Self::F16 => n * 2,
      Self::Q8_0 => n / 32 * 34,
      Self::Q5_0 => n / 32 * 22,
    }
  }
}

/// Codes 0..31 of one ggml Q5_0 block, in weight order. The block is an f16 scale, four high-bit
/// bytes, then sixteen low nibbles.
pub(crate) fn q5_codes(block: &[u8; 22]) -> [u8; 32] {
  let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
  let mut codes = [0u8; 32];
  for j in 0..16 {
    let byte = block[6 + j];
    codes[j] = (byte & 0x0f) | (((qh >> j) & 1) << 4) as u8;
    codes[j + 16] = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4) as u8;
  }
  codes
}

/// The block's codes packed little-endian, five bits each, then four zero bytes. A group of four
/// starts at bit `group * 20`, and the padding keeps that group's load inside the 24 bytes.
pub(crate) fn q5_packed(block: &[u8; 22]) -> [u8; 24] {
  let mut out = [0u8; 24];
  let mut acc = 0u64;
  let mut bits = 0;
  let mut at = 0;
  for code in q5_codes(block) {
    acc |= u64::from(code) << bits;
    bits += 5;
    while bits >= 8 {
      out[at] = acc as u8;
      acc >>= 8;
      bits -= 8;
      at += 1;
    }
  }
  out
}

/// ggml Q5_0 over `w`, whose length is a multiple of 32. The scale is the largest-magnitude
/// element divided by −16 (the first one on a tie), stored as f16.
#[cfg(test)]
pub(crate) fn quantize_q5_0(w: &[f32]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(32));
  let mut out = Vec::with_capacity(w.len() / 32 * 22);
  for block in w.as_chunks::<32>().0 {
    let mut max = block[0];
    for &x in &block[1..] {
      if x.abs() > max.abs() {
        max = x;
      }
    }
    let d = max / -16.0;
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    let mut codes = [0u8; 32];
    for (code, &x) in codes.iter_mut().zip(block) {
      *code = (x * id + 16.5).trunc().clamp(0.0, 31.0) as u8;
    }
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
      qs[j] = (codes[j] & 0x0f) | (codes[j + 16] << 4);
      qh |= ((codes[j] >> 4) as u32) << j;
      qh |= ((codes[j + 16] >> 4) as u32) << (j + 16);
    }
    out.extend(f16::from_f32(d).to_le_bytes());
    out.extend(qh.to_le_bytes());
    out.extend(qs);
  }
  out
}

/// A metadata value, decoded from the header when a model reads it.
#[derive(Debug, Clone, Copy)]
pub enum Value<'a> {
  /// Any integer type: no model reads a u64 above `i64::MAX`.
  Int(i64),
  Float(f64),
  Bool(bool),
  /// `None` when not UTF-8.
  Str(Option<&'a str>),
  /// Items of GGUF type `item`, read by `Gguf::array`.
  Array {
    item: u32,
    len: usize,
  },
}

/// A metadata type that models read: `u32`, `i32`, `f32`, `bool` or `&str`.
pub trait Meta<'a>: Sized {
  fn from(v: Value<'a>) -> Option<Self>;
}

impl Meta<'_> for u32 {
  fn from(v: Value) -> Option<Self> {
    if let Value::Int(v) = v { v.try_into().ok() } else { None }
  }
}

impl Meta<'_> for i32 {
  fn from(v: Value) -> Option<Self> {
    if let Value::Int(v) = v { v.try_into().ok() } else { None }
  }
}

impl Meta<'_> for f32 {
  fn from(v: Value) -> Option<Self> {
    if let Value::Float(v) = v { Some(v as f32) } else { None }
  }
}

impl Meta<'_> for bool {
  fn from(v: Value) -> Option<Self> {
    if let Value::Bool(v) = v { Some(v) } else { None }
  }
}

impl<'a> Meta<'a> for &'a str {
  fn from(v: Value<'a>) -> Option<Self> {
    if let Value::Str(s) = v { s } else { None }
  }
}

struct Info {
  dims: Vec<usize>,
  ty: Type,
  offset: usize,
}

pub struct Gguf {
  path: PathBuf,
  /// The start of the file, at least the header; tensors inside it are read from here.
  head: Vec<u8>,
  /// Per key the value's GGUF type and its offset in `head`. Values are decoded when read, so a
  /// vocabulary nobody reads costs nothing.
  meta: HashMap<String, (u32, usize)>,
  tensors: HashMap<String, Info>,
  data: usize,
}

/// A tensor in the file. `dims` are in ggml order: `dims[0]` is the contiguous axis, so a linear
/// weight `[in, out]` is row-major `[out][in]`.
pub struct Tensor<'a> {
  pub dims: &'a [usize],
  pub ty: Type,
  gguf: &'a Gguf,
  start: usize,
  size: usize,
}

impl Gguf {
  /// Reads the header only; tensor data is read when used.
  pub fn open(path: &Path) -> Result<Self> {
    let len = std::fs::metadata(path)?.len() as usize;
    // The header is usually far below 1 MiB; read more of the file if it doesn't fit.
    let mut n = len.min(1 << 20);
    loop {
      let mut head = vec![0; n];
      File::open(path)?.read_exact(&mut head)?;
      match Self::parse(head, len, path.into()) {
        Err(_) if n < len => n = len.min(n * 4),
        r => return r,
      }
    }
  }

  fn parse(head: Vec<u8>, len: usize, path: PathBuf) -> Result<Self> {
    let mut r = Reader { bytes: &head, pos: 0 };
    if r.take(4)? != b"GGUF" {
      bail!("not a GGUF file");
    }
    let version = r.u32()?;
    if version != 3 {
      bail!("unsupported GGUF version {version}");
    }
    let n_tensors = r.u64()? as usize;
    let n_meta = r.u64()? as usize;

    let mut meta = HashMap::new();
    for _ in 0..n_meta {
      let key = r.string()?;
      let ty = r.u32()?;
      meta.insert(key, (ty, r.pos));
      r.skip(ty)?;
    }

    let mut tensors = HashMap::new();
    for _ in 0..n_tensors {
      let name = r.string()?;
      let n_dims = r.u32()? as usize;
      let dims = (0..n_dims).map(|_| r.u64().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
      let id = r.u32()?;
      let Some(ty) = Type::from_ggml(id) else {
        bail!("tensor {name}: unsupported ggml type {id}")
      };
      let offset = r.u64()? as usize;
      tensors.insert(name, Info { dims, ty, offset });
    }

    let end = r.pos;
    let mut g = Self { path, head, meta, tensors, data: 0 };
    let align = match g.value("general.alignment") {
      Ok((_, Value::Int(a))) => a as usize,
      _ => 32,
    };
    g.data = end.next_multiple_of(align);
    for (name, t) in &g.tensors {
      if g.data + t.offset + t.ty.size(t.dims.iter().product()) > len {
        bail!("tensor {name} is out of file bounds");
      }
    }
    Ok(g)
  }

  /// The value of `key`, and a reader past it (at an array's items).
  fn value(&self, key: &str) -> Result<(Reader<'_>, Value<'_>)> {
    let Some(&(ty, pos)) = self.meta.get(key) else { bail!("missing metadata {key}") };
    let mut r = Reader { bytes: &self.head, pos };
    let v = r.value(ty)?;
    Ok((r, v))
  }

  /// The metadata value `key` as a `T`.
  pub fn get<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<T> {
    let (_, v) = self.value(key)?;
    match T::from(v) {
      Some(v) => Ok(v),
      None => bail!("metadata {key}: expected {}, got {v:?}", std::any::type_name::<T>()),
    }
  }

  /// The metadata value `key` as a `T`, or `None` when the file has no `key`.
  pub fn optional<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<Option<T>> {
    if !self.meta.contains_key(key) {
      return Ok(None);
    }
    self.get(key).map(Some)
  }

  /// Fails unless the metadata value `key` is one of `allowed`.
  pub fn check<'a, T: Meta<'a> + PartialEq + Debug>(
    &'a self,
    key: &str,
    allowed: &[T],
  ) -> Result<()> {
    let got = self.get::<T>(key)?;
    if !allowed.contains(&got) {
      bail!("{key} is {got:?}, only {allowed:?} supported");
    }
    Ok(())
  }

  /// The metadata array `key` as `T`s.
  pub fn array<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<Vec<T>> {
    let (mut r, Value::Array { item, len }) = self.value(key)? else {
      bail!("metadata {key}: expected an array")
    };
    let items = (0..len).map(|_| r.value(item).map(T::from)).collect::<Result<Option<_>>>()?;
    let Some(items) = items else {
      bail!("metadata {key}: expected an array of {}", std::any::type_name::<T>())
    };
    Ok(items)
  }

  /// The metadata array `key` as `T`s, empty when the file has no `key`.
  pub fn optional_array<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<Vec<T>> {
    if !self.meta.contains_key(key) {
      return Ok(Vec::new());
    }
    self.array(key)
  }

  /// The tensor `name`, which must have exactly the ggml-order `dims`.
  pub fn tensor(&self, name: &str, dims: &[usize]) -> Result<Tensor<'_>> {
    let Some(t) = self.tensors.get(name) else { bail!("missing tensor {name}") };
    if t.dims != dims {
      bail!("tensor {name}: shape {:?}, expected {dims:?}", t.dims);
    }
    let (start, size) = (self.data + t.offset, t.ty.size(dims.iter().product()));
    Ok(Tensor { dims: &t.dims, ty: t.ty, gguf: self, start, size })
  }
}

impl Tensor<'_> {
  pub fn len(&self) -> usize {
    self.dims.iter().product()
  }

  /// Calls `f(offset, bytes)` on consecutive pieces of the tensor's data of up to `chunk` bytes.
  pub fn read(&self, chunk: usize, mut f: impl FnMut(usize, &[u8])) -> Result<()> {
    if let Some(bytes) = self.gguf.head.get(self.start..self.start + self.size) {
      bytes.chunks(chunk).enumerate().for_each(|(i, c)| f(i * chunk, c));
      return Ok(());
    }
    let mut file = File::open(&self.gguf.path)?;
    file.seek(SeekFrom::Start(self.start as u64))?;
    let mut buf = vec![0; chunk.min(self.size)];
    for offset in (0..self.size).step_by(chunk) {
      let piece = &mut buf[..chunk.min(self.size - offset)];
      file.read_exact(piece)?;
      f(offset, piece);
    }
    Ok(())
  }

  pub fn to_f32(&self) -> Result<Vec<f32>> {
    let mut b = Vec::with_capacity(self.size);
    self.read(self.size.max(1), |_, piece| b.extend_from_slice(piece))?;
    Ok(match self.ty {
      Type::F32 => b.as_chunks().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
      Type::F16 => b.as_chunks().0.iter().map(|&c| f16::from_le_bytes(c).to_f32()).collect(),
      Type::Q8_0 => (b.as_chunks::<34>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          block[2..].iter().map(move |&q| q as i8 as f32 * d)
        })
        .collect(),
      Type::Q5_0 => (b.as_chunks::<22>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          q5_codes(block).into_iter().map(move |c| (c as i32 - 16) as f32 * d)
        })
        .collect(),
    })
  }
}

#[cfg(test)]
impl Gguf {
  /// An in-memory file holding `(name, ggml dims, type, data)` tensors, for tests.
  pub(crate) fn with_tensors(tensors: &[(&str, &[usize], Type, &[u8])]) -> Self {
    let mut head = Vec::new();
    let mut infos = HashMap::new();
    for &(name, dims, ty, data) in tensors {
      infos.insert(name.to_string(), Info { dims: dims.to_vec(), ty, offset: head.len() });
      head.extend_from_slice(data);
    }
    Self { path: PathBuf::new(), head, meta: HashMap::new(), tensors: infos, data: 0 }
  }
}

struct Reader<'a> {
  bytes: &'a [u8],
  pos: usize,
}

impl<'a> Reader<'a> {
  fn take(&mut self, n: usize) -> Result<&'a [u8]> {
    let Some(s) = self.bytes.get(self.pos..self.pos + n) else { bail!("truncated GGUF header") };
    self.pos += n;
    Ok(s)
  }

  fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
    Ok(self.take(N)?.try_into().unwrap())
  }

  fn u32(&mut self) -> Result<u32> {
    Ok(u32::from_le_bytes(self.array()?))
  }

  fn u64(&mut self) -> Result<u64> {
    Ok(u64::from_le_bytes(self.array()?))
  }

  fn bytes(&mut self) -> Result<&'a [u8]> {
    let n = self.u64()? as usize;
    self.take(n)
  }

  fn string(&mut self) -> Result<String> {
    Ok(String::from_utf8_lossy(self.bytes()?).into_owned())
  }

  /// Moves past a value of type `ty`.
  fn skip(&mut self, ty: u32) -> Result<()> {
    let size = |ty| match ty {
      0 | 1 | 7 => Some(1),
      2 | 3 => Some(2),
      4..=6 => Some(4),
      10..=12 => Some(8),
      _ => None,
    };
    match ty {
      8 => _ = self.bytes()?,
      9 => {
        let item = self.u32()?;
        let n = self.u64()? as usize;
        match size(item) {
          Some(size) => _ = self.take(n * size)?,
          None => (0..n).try_for_each(|_| self.skip(item))?,
        }
      }
      _ => match size(ty) {
        Some(size) => _ = self.take(size)?,
        None => bail!("unknown GGUF value type {ty}"),
      },
    }
    Ok(())
  }

  /// Reads a value of type `ty`; for an array its item type and length, the items follow.
  fn value(&mut self, ty: u32) -> Result<Value<'a>> {
    Ok(match ty {
      0 => Value::Int(self.array::<1>()?[0] as i64),
      1 => Value::Int(self.array::<1>()?[0] as i8 as i64),
      2 => Value::Int(u16::from_le_bytes(self.array()?) as i64),
      3 => Value::Int(i16::from_le_bytes(self.array()?) as i64),
      4 => Value::Int(self.u32()? as i64),
      5 => Value::Int(i32::from_le_bytes(self.array()?) as i64),
      6 => Value::Float(f32::from_le_bytes(self.array()?) as f64),
      7 => Value::Bool(self.array::<1>()?[0] != 0),
      8 => Value::Str(std::str::from_utf8(self.bytes()?).ok()),
      9 => Value::Array { item: self.u32()?, len: self.u64()? as usize },
      10 => Value::Int(self.u64()? as i64),
      11 => Value::Int(i64::from_le_bytes(self.array()?)),
      12 => Value::Float(f64::from_le_bytes(self.array()?)),
      _ => bail!("unknown GGUF value type {ty}"),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
  }

  /// A file with a u32, a string array, a bool, a negative i32, an i32 array and an f32, and an
  /// F32 + a Q8_0 tensor.
  fn sample() -> Vec<u8> {
    let mut f = b"GGUF".to_vec();
    f.extend(3u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    f.extend(6u64.to_le_bytes());
    string(&mut f, "a.n");
    f.extend(4u32.to_le_bytes());
    f.extend(7u32.to_le_bytes());
    string(&mut f, "a.tokens");
    f.extend(9u32.to_le_bytes());
    f.extend(8u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    string(&mut f, "▁x");
    string(&mut f, "y");
    string(&mut f, "a.flag");
    f.extend(7u32.to_le_bytes());
    f.push(1);
    string(&mut f, "a.left");
    f.extend(5u32.to_le_bytes());
    f.extend((-1i32).to_le_bytes());
    string(&mut f, "a.durations");
    f.extend(9u32.to_le_bytes());
    f.extend(5u32.to_le_bytes());
    f.extend(3u64.to_le_bytes());
    [0i32, 2, -4].iter().for_each(|v| f.extend(v.to_le_bytes()));
    string(&mut f, "a.alpha");
    f.extend(6u32.to_le_bytes());
    f.extend(0.97f32.to_le_bytes());

    string(&mut f, "w");
    f.extend(2u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    f.extend(1u64.to_le_bytes());
    f.extend(0u32.to_le_bytes());
    f.extend(0u64.to_le_bytes());
    string(&mut f, "q");
    f.extend(1u32.to_le_bytes());
    f.extend(32u64.to_le_bytes());
    f.extend(8u32.to_le_bytes());
    f.extend(32u64.to_le_bytes());

    f.resize(f.len().next_multiple_of(32), 0);
    f.extend(1.5f32.to_le_bytes());
    f.extend((-2.0f32).to_le_bytes());
    f.resize(f.len().next_multiple_of(32), 0);
    f.extend(f16::from_f32(0.5).to_le_bytes());
    f.extend((0..32).map(|i| (i as i8 - 16) as u8));
    f
  }

  fn parse(f: Vec<u8>) -> Result<Gguf> {
    let len = f.len();
    Gguf::parse(f, len, PathBuf::new())
  }

  #[test]
  fn reads_metadata_and_tensors() {
    let g = parse(sample()).unwrap();
    assert_eq!(g.get::<u32>("a.n").unwrap(), 7);
    assert_eq!(g.get::<i32>("a.n").unwrap(), 7);
    assert!(g.get::<u32>("missing").is_err());
    assert!(g.get::<&str>("a.n").is_err());
    assert_eq!(g.array::<&str>("a.tokens").unwrap(), ["▁x", "y"]);
    assert!(g.get::<bool>("a.flag").unwrap());
    assert_eq!(g.optional::<bool>("a.flag").unwrap(), Some(true));
    assert_eq!(g.optional::<bool>("missing").unwrap(), None);
    assert!(g.optional::<&str>("a.flag").is_err());
    assert_eq!(g.get::<i32>("a.left").unwrap(), -1);
    assert!(g.get::<u32>("a.left").is_err());
    assert_eq!(g.array::<i32>("a.durations").unwrap(), [0, 2, -4]);
    assert!(g.array::<u32>("a.durations").is_err());
    assert_eq!(g.optional_array::<i32>("a.durations").unwrap(), [0, 2, -4]);
    assert!(g.optional_array::<i32>("missing").unwrap().is_empty());
    assert!(g.optional_array::<u32>("a.durations").is_err());
    assert_eq!(g.get::<f32>("a.alpha").unwrap(), 0.97);

    assert_eq!(g.tensor("w", &[2, 1]).unwrap().to_f32().unwrap(), [1.5, -2.0]);
    assert!(g.tensor("w", &[1, 2]).is_err());

    let q = g.tensor("q", &[32]).unwrap();
    assert_eq!(q.ty, Type::Q8_0);
    let expected: Vec<f32> = (0..32).map(|i| (i - 16) as f32 * 0.5).collect();
    assert_eq!(q.to_f32().unwrap(), expected);

    let mut pieces = Vec::new();
    q.read(16, |offset, piece| pieces.push((offset, piece.len()))).unwrap();
    assert_eq!(pieces, [(0, 16), (16, 16), (32, 2)]);
  }

  /// One Q5_0 block: f16 scale, high bits of `codes`, then the low nibbles. `codes` are 0..31.
  fn q5_block(codes: [u8; 32], scale: f32) -> [u8; 22] {
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
      qs[j] = (codes[j] & 0x0f) | (codes[j + 16] << 4);
      qh |= ((codes[j] >> 4) as u32) << j;
      qh |= ((codes[j + 16] >> 4) as u32) << (j + 16);
    }
    let mut block = [0u8; 22];
    block[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    block[2..6].copy_from_slice(&qh.to_le_bytes());
    block[6..22].copy_from_slice(&qs);
    block
  }

  /// The shader's group load: four codes at `group`, from the 24-byte pack.
  fn q5_group(packed: &[u8; 24], group: usize) -> [u8; 4] {
    let bit = group * 20;
    let byte = bit >> 3;
    let i = byte >> 2;
    let s = (byte & 3) * 8;
    let mut word = u32::from_le_bytes(packed[i * 4..][..4].try_into().unwrap());
    if s != 0 {
      let hi = u32::from_le_bytes(packed[(i + 1) * 4..][..4].try_into().unwrap());
      word = (word >> s) | (hi << (32 - s));
    }
    word >>= bit & 7;
    [
      (word & 31) as u8,
      ((word >> 5) & 31) as u8,
      ((word >> 10) & 31) as u8,
      ((word >> 15) & 31) as u8,
    ]
  }

  #[test]
  fn q5_0_round_trips() {
    let codes = std::array::from_fn(|i| i as u8);
    let block = q5_block(codes, 0.5);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q5_0, &block)]);
    let expected: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);

    // −16..15 is an exact grid: the scale is 1 and the codes are 0..31.
    let grid: Vec<f32> = (0i32..32).map(|i| (i - 16) as f32).collect();
    assert_eq!(quantize_q5_0(&grid), q5_block(codes, 1.0));

    let packed = q5_packed(&block);
    for group in 0..8 {
      assert_eq!(&q5_group(&packed, group), &codes[group * 4..group * 4 + 4]);
    }

    let zeros = quantize_q5_0(&[0.0; 32]);
    let z: &[u8; 22] = zeros.as_slice().try_into().unwrap();
    assert_eq!(q5_codes(z), [16; 32]);
    assert_eq!(f16::from_le_bytes([zeros[0], zeros[1]]).to_f32(), 0.0);
    let g = Gguf::with_tensors(&[("z", &[32], Type::Q5_0, &zeros)]);
    assert_eq!(g.tensor("z", &[32]).unwrap().to_f32().unwrap(), vec![0.0; 32]);

    // The first of two equal magnitudes wins, so the scale is negative and −8 clips to −7.5.
    let mut tied = [0.0; 32];
    tied[0] = 8.0;
    tied[1] = -8.0;
    let q = quantize_q5_0(&tied);
    let g = Gguf::with_tensors(&[("t", &[32], Type::Q5_0, &q)]);
    let got = g.tensor("t", &[32]).unwrap().to_f32().unwrap();
    assert_eq!(got[0], 8.0);
    assert_eq!(got[1], -7.5);
  }

  #[test]
  fn rejects_truncated_files() {
    let mut f = sample();
    f.truncate(f.len() - 1);
    assert!(parse(f).is_err());
  }
}
