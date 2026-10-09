// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GGUF v3 reader: metadata plus F32/F16/Q8_0 tensor views over the file bytes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use half::f16;

use crate::error::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
  F32,
  F16,
  /// Blocks of 32 weights: f16 scale + 32 × i8 (34 bytes).
  Q8_0,
}

impl Type {
  fn from_ggml(id: u32) -> Option<Self> {
    match id {
      0 => Some(Self::F32),
      1 => Some(Self::F16),
      8 => Some(Self::Q8_0),
      _ => None,
    }
  }

  fn size(self, n: usize) -> usize {
    match self {
      Self::F32 => n * 4,
      Self::F16 => n * 2,
      Self::Q8_0 => n / 32 * 34,
    }
  }
}

#[derive(Debug)]
enum Value {
  Uint(u64),
  Int(i64),
  Str(String),
  Array(Vec<Value>),
  /// Floats and bools, which no model reads.
  Other,
}

struct Info {
  dims: Vec<usize>,
  ty: Type,
  offset: usize,
}

pub struct Gguf {
  bytes: Vec<u8>,
  meta: HashMap<String, Value>,
  tensors: HashMap<String, Info>,
  data: usize,
}

/// A tensor inside the file. `dims` are in ggml order: `dims[0]` is the contiguous axis, so a
/// linear weight `[in, out]` is row-major `[out][in]`.
pub struct Tensor<'a> {
  pub dims: &'a [usize],
  pub ty: Type,
  pub bytes: &'a [u8],
}

impl Gguf {
  /// Reads the file in 8 parallel chunks, which is several times faster from the page cache.
  pub fn open(path: &Path) -> Result<Self> {
    let len = std::fs::metadata(path)?.len() as usize;
    let mut bytes = vec![0; len];
    std::thread::scope(|s| {
      let chunks = bytes.chunks_mut(len.div_ceil(8).max(1)).enumerate();
      let threads: Vec<_> = (chunks.map(|(i, chunk)| {
        s.spawn(move || -> std::io::Result<()> {
          let mut file = File::open(path)?;
          file.seek(SeekFrom::Start((i * len.div_ceil(8)) as u64))?;
          file.read_exact(chunk)
        })
      }))
      .collect();
      threads.into_iter().try_for_each(|t| t.join().unwrap())
    })?;
    Self::parse(bytes)
  }

  pub fn parse(bytes: Vec<u8>) -> Result<Self> {
    let mut r = Reader { bytes: &bytes, pos: 0 };
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
      meta.insert(key, r.value(ty)?);
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

    let align = match meta.get("general.alignment") {
      Some(Value::Uint(a)) => *a as usize,
      _ => 32,
    };
    let data = r.pos.next_multiple_of(align);
    for (name, t) in &tensors {
      let end = data + t.offset + t.ty.size(t.dims.iter().product());
      if end > bytes.len() {
        bail!("tensor {name} is out of file bounds");
      }
    }
    Ok(Self { bytes, meta, tensors, data })
  }

  fn get(&self, key: &str) -> Result<&Value> {
    match self.meta.get(key) {
      Some(v) => Ok(v),
      None => bail!("missing metadata {key}"),
    }
  }

  pub fn u32(&self, key: &str) -> Result<u32> {
    match self.get(key)? {
      Value::Uint(v) => Ok(*v as u32),
      Value::Int(v) => Ok(*v as u32),
      v => bail!("metadata {key}: expected an integer, got {v:?}"),
    }
  }

  pub fn str(&self, key: &str) -> Result<&str> {
    match self.get(key)? {
      Value::Str(s) => Ok(s),
      v => bail!("metadata {key}: expected a string, got {v:?}"),
    }
  }

  pub fn strs(&self, key: &str) -> Result<Vec<&str>> {
    let Value::Array(items) = self.get(key)? else { bail!("metadata {key}: expected an array") };
    items
      .iter()
      .map(|v| match v {
        Value::Str(s) => Ok(s.as_str()),
        v => bail!("metadata {key}: expected strings, got {v:?}"),
      })
      .collect()
  }

  /// The tensor `name`, which must have exactly the ggml-order `dims`.
  pub fn tensor(&self, name: &str, dims: &[usize]) -> Result<Tensor<'_>> {
    let Some(t) = self.tensors.get(name) else { bail!("missing tensor {name}") };
    if t.dims != dims {
      bail!("tensor {name}: shape {:?}, expected {dims:?}", t.dims);
    }
    let start = self.data + t.offset;
    let bytes = &self.bytes[start..start + t.ty.size(dims.iter().product())];
    Ok(Tensor { dims: &t.dims, ty: t.ty, bytes })
  }
}

impl Tensor<'_> {
  pub fn len(&self) -> usize {
    self.dims.iter().product()
  }

  pub fn to_f32(&self) -> Vec<f32> {
    let b = self.bytes;
    match self.ty {
      Type::F32 => b.as_chunks().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
      Type::F16 => b.as_chunks().0.iter().map(|&c| f16::from_le_bytes(c).to_f32()).collect(),
      Type::Q8_0 => (b.as_chunks::<34>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          block[2..].iter().map(move |&q| q as i8 as f32 * d)
        })
        .collect(),
    }
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

  fn string(&mut self) -> Result<String> {
    let n = self.u64()? as usize;
    Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
  }

  fn value(&mut self, ty: u32) -> Result<Value> {
    Ok(match ty {
      0 => Value::Uint(self.array::<1>()?[0] as u64),
      1 => Value::Int(self.array::<1>()?[0] as i8 as i64),
      2 => Value::Uint(u16::from_le_bytes(self.array()?) as u64),
      3 => Value::Int(i16::from_le_bytes(self.array()?) as i64),
      4 => Value::Uint(self.u32()? as u64),
      5 => Value::Int(i32::from_le_bytes(self.array()?) as i64),
      6 => self.take(4).map(|_| Value::Other)?,
      7 => self.take(1).map(|_| Value::Other)?,
      8 => Value::Str(self.string()?),
      9 => {
        let item = self.u32()?;
        let n = self.u64()? as usize;
        Value::Array((0..n).map(|_| self.value(item)).collect::<Result<_>>()?)
      }
      10 => Value::Uint(self.u64()?),
      11 => Value::Int(i64::from_le_bytes(self.array()?)),
      12 => self.take(8).map(|_| Value::Other)?,
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

  /// A file with one u32, one string array and an F32 + a Q8_0 tensor.
  fn sample() -> Vec<u8> {
    let mut f = b"GGUF".to_vec();
    f.extend(3u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    string(&mut f, "a.n");
    f.extend(4u32.to_le_bytes());
    f.extend(7u32.to_le_bytes());
    string(&mut f, "a.tokens");
    f.extend(9u32.to_le_bytes());
    f.extend(8u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    string(&mut f, "▁x");
    string(&mut f, "y");

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

  #[test]
  fn reads_metadata_and_tensors() {
    let g = Gguf::parse(sample()).unwrap();
    assert_eq!(g.u32("a.n").unwrap(), 7);
    assert_eq!(g.strs("a.tokens").unwrap(), ["▁x", "y"]);
    assert!(g.u32("missing").is_err());

    assert_eq!(g.tensor("w", &[2, 1]).unwrap().to_f32(), [1.5, -2.0]);
    assert!(g.tensor("w", &[1, 2]).is_err());

    let q = g.tensor("q", &[32]).unwrap();
    assert_eq!(q.ty, Type::Q8_0);
    let expected: Vec<f32> = (0..32).map(|i| (i - 16) as f32 * 0.5).collect();
    assert_eq!(q.to_f32(), expected);
  }

  #[test]
  fn rejects_truncated_files() {
    let mut f = sample();
    f.truncate(f.len() - 1);
    assert!(Gguf::parse(f).is_err());
  }
}
