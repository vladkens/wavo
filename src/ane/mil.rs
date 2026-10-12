// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! A Core ML ML Program written as an `.mlpackage`: `Manifest.json`, the `Model.proto` message in
//! `Data/com.apple.CoreML/model.mlmodel`, and the weights in `weights/weight.bin`. Every tensor
//! has a fixed shape, so each op's output shape is computed here, where it is built.
//!
//! The weight file is coremltools' MIL blob storage: a 64-byte header (blob count, version 2),
//! then per blob a 64-byte record (0xDEADBEEF, data type, size, data offset) followed by its data,
//! each record and data at a multiple of 64 bytes. A tensor refers to its record's offset.
//!
//! Adapted from coremltools (BSD-3-Clause, Apple; the notice is in LICENSE): the message field
//! numbers and data type codes from `mlmodel/format/Model.proto`, `MIL.proto` and
//! `FeatureTypes.proto`; the weight file layout from `mlmodel/src/MILBlob/Blob/StorageFormat.hpp`;
//! the package layout and `Manifest.json` from `modelpackage/src/ModelPackage.cpp`; op names and
//! their inputs from the iOS 15–18 opsets in `coremltools/converters/mil/mil/ops/defs/`.

use std::fs::{self, File};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use half::f16;

use super::proto::Msg;

/// MIL's opset for macOS 15 / iOS 18, which has `constexpr_blockwise_shift_scale`.
const OPSET: &str = "CoreML8";
/// `Model.specificationVersion` of that opset.
const SPEC_VERSION: i64 = 9;
const WEIGHTS: &str = "@model_path/weights/weight.bin";
/// Constants with more elements than this go into the weight file, as coremltools does.
const INLINE: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dtype {
  F16,
  I8,
  I32,
  Bool,
  Str,
}

impl Dtype {
  /// `MILSpec.DataType`.
  fn mil(self) -> i64 {
    match self {
      Self::F16 => 10,
      Self::I8 => 21,
      Self::I32 => 23,
      Self::Bool => 1,
      Self::Str => 2,
    }
  }
}

/// A value of the program: its name, type and shape (empty for a scalar).
#[derive(Clone, Debug)]
pub struct Var {
  pub name: String,
  pub dtype: Dtype,
  pub shape: Vec<usize>,
}

/// A tensor stored in the weight file, by the offset of its record.
struct Blob {
  offset: u64,
  dtype: Dtype,
  shape: Vec<usize>,
}

pub struct Program {
  inputs: Vec<Var>,
  /// The block's operations, serialized as they are added.
  ops: Msg,
  weights: BufWriter<File>,
  weights_len: u64,
  blobs: u32,
  /// The first failed write to the weight file, returned by `finish`.
  error: Option<io::Error>,
  names: usize,
  /// The name of the next op's output, set by `name_next`.
  next: Option<String>,
}

impl Program {
  /// A program whose weights go to `weights` (`weights/weight.bin` of the package).
  pub fn new(weights: &Path) -> io::Result<Self> {
    fs::create_dir_all(weights.parent().unwrap())?;
    let mut p = Self {
      inputs: Vec::new(),
      ops: Msg::default(),
      weights: BufWriter::with_capacity(1 << 20, File::create(weights)?),
      weights_len: 0,
      blobs: 0,
      error: None,
      names: 0,
      next: None,
    };
    p.write(&[0; 64]); // the header, completed by `finish`
    Ok(p)
  }

  fn name(&mut self, prefix: &str) -> String {
    self.names += 1;
    format!("{prefix}_{}", self.names)
  }

  /// Names the output of the next op (not constant): a program output's name is its feature name.
  pub fn name_next(&mut self, name: &str) {
    self.next = Some(name.into());
  }

  /// A function input of F16 values.
  pub fn input(&mut self, name: &str, shape: &[usize]) -> Var {
    let v = Var { name: name.into(), dtype: Dtype::F16, shape: shape.to_vec() };
    self.inputs.push(v.clone());
    v
  }

  fn write(&mut self, b: &[u8]) {
    if self.error.is_none() {
      self.error = self.weights.write_all(b).err();
    }
    self.weights_len += b.len() as u64;
  }

  /// Appends a blob of `data` bytes of `dtype` (F16 or I8) to the weight file.
  fn blob(&mut self, dtype: Dtype, shape: &[usize], data: &[u8]) -> Blob {
    let offset = self.weights_len;
    let code: u32 = if dtype == Dtype::F16 { 1 } else { 4 };
    let mut record = [0u8; 64];
    record[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    record[4..8].copy_from_slice(&code.to_le_bytes());
    record[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
    record[16..24].copy_from_slice(&(offset + 64).to_le_bytes());
    self.write(&record);
    self.write(data);
    let pad = (data.len() as u64).next_multiple_of(64) - data.len() as u64;
    self.write(&vec![0; pad as usize]);
    self.blobs += 1;
    Blob { offset, dtype, shape: shape.to_vec() }
  }

  /// Adds the op `ty` with named inputs, each one value or a list (`concat`'s `values`), and
  /// returns its F16 output of `shape`.
  pub fn op(&mut self, ty: &str, inputs: &[(&str, &[&Var])], shape: Vec<usize>) -> Var {
    let name = self.next.take().unwrap_or_else(|| self.name(ty));
    let out = Var { name, dtype: Dtype::F16, shape };
    self.ops.msg(3, |op| {
      op.str(1, ty);
      for (key, vars) in inputs {
        op.entry(2, key, |arg| vars.iter().for_each(|v| _ = arg.msg(1, |b| _ = b.str(1, &v.name))));
      }
      op.msg(3, |o| named(o, &out));
      name_attribute(op, &out.name);
    });
    out
  }

  /// A `const` op holding `val`, an immediate value or a blob.
  fn constant(&mut self, dtype: Dtype, shape: &[usize], val: impl FnOnce(&mut Msg)) -> Var {
    let out = Var { name: self.name("const"), dtype, shape: shape.to_vec() };
    self.ops.msg(3, |op| {
      op.str(1, "const");
      op.msg(3, |o| named(o, &out));
      op.entry(5, "val", |v| {
        v.msg(2, |t| value_type(t, dtype, shape));
        val(v);
      });
      name_attribute(op, &out.name);
    });
    out
  }

  /// An I32 tensor; `shape` empty for a scalar.
  pub fn ints(&mut self, shape: &[usize], v: &[i64]) -> Var {
    self.constant(Dtype::I32, shape, |m| {
      immediate(m, |t| _ = t.msg(2, |r| _ = r.ints(1, v.iter().copied())))
    })
  }

  pub fn int(&mut self, v: i64) -> Var {
    self.ints(&[], &[v])
  }

  /// A 1-D I32 tensor such as a shape or a permutation.
  pub fn list(&mut self, v: &[i64]) -> Var {
    self.ints(&[v.len()], v)
  }

  pub fn boolean(&mut self, v: bool) -> Var {
    self
      .constant(Dtype::Bool, &[], |m| immediate(m, |t| _ = t.msg(3, |r| _ = r.ints(1, [v as i64]))))
  }

  pub fn string(&mut self, v: &str) -> Var {
    self.constant(Dtype::Str, &[], |m| immediate(m, |t| _ = t.msg(4, |r| _ = r.str(1, v))))
  }

  /// An F16 tensor of `data` rounded from f32; `shape` empty for a scalar.
  pub fn tensor(&mut self, shape: &[usize], data: &[f32]) -> Var {
    debug_assert_eq!(shape.iter().product::<usize>(), data.len());
    let bytes: Vec<u8> = data.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect();
    if data.len() <= INLINE {
      return self.constant(Dtype::F16, shape, |m| {
        immediate(m, |t| _ = t.msg(7, |r| _ = r.bytes(1, &bytes)))
      });
    }
    let blob = self.blob(Dtype::F16, shape, &bytes);
    self.constant(Dtype::F16, shape, |m| blob_file(m, &blob))
  }

  pub fn scalar(&mut self, v: f32) -> Var {
    self.tensor(&[], &[v])
  }

  /// A weight `[rows][cols]` stored as int8 with one F16 scale per row, dequantized when the
  /// model loads (`constexpr_blockwise_shift_scale`). The Neural Engine runs linear layers on
  /// such weights, but not on Q8_0's 32-wide blocks.
  pub fn int8(&mut self, rows: usize, cols: usize, w: &[f32]) -> Var {
    let (q, scales) = quantize(w, cols);
    let data = self.blob(Dtype::I8, &[rows, cols], &q.iter().map(|&q| q as u8).collect::<Vec<_>>());
    let scales: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scale = self.blob(Dtype::F16, &[rows, 1], &scales);
    let out = Var { name: self.name("weight"), dtype: Dtype::F16, shape: vec![rows, cols] };
    self.ops.msg(3, |op| {
      op.str(1, "constexpr_blockwise_shift_scale");
      for (key, blob) in [("data", &data), ("scale", &scale)] {
        op.entry(2, key, |arg| _ = arg.msg(1, |b| _ = b.msg(2, |v| blob_value(v, blob))));
      }
      op.msg(3, |o| named(o, &out));
      name_attribute(op, &out.name);
    });
    out
  }

  /// Writes the model with the given outputs and completes the weight file. `dir` is the
  /// `.mlpackage` folder, whose weight file `new` was given.
  pub fn finish(mut self, outputs: &[&Var], dir: &Path) -> io::Result<()> {
    if let Some(e) = self.error.take() {
      return Err(e);
    }
    let mut file = self.weights.into_inner().map_err(|e| e.into_error())?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&self.blobs.to_le_bytes())?;
    file.write_all(&2u32.to_le_bytes())?;
    file.sync_all()?;

    let feature = |m: &mut Msg, v: &Var| {
      m.str(1, &v.name);
      // FeatureType.multiArrayType: shape, FLOAT16
      m.msg(3, |t| {
        _ = t.msg(5, |a| _ = a.ints(1, v.shape.iter().map(|&d| d as i64)).int(2, 65552))
      });
    };
    let mut model = Msg::default();
    model.int(1, SPEC_VERSION);
    model.msg(2, |d| {
      self.inputs.iter().for_each(|v| _ = d.msg(1, |f| feature(f, v)));
      outputs.iter().for_each(|v| _ = d.msg(10, |f| feature(f, v)));
    });
    let mut block = self.ops;
    outputs.iter().for_each(|v| _ = block.str(2, &v.name));
    model.msg(502, |p| {
      p.int(1, 1);
      p.entry(2, "main", |f| {
        self.inputs.iter().for_each(|v| _ = f.msg(1, |n| named(n, v)));
        f.str(2, OPSET);
        f.entry(3, OPSET, |b| b.0 = std::mem::take(&mut block.0));
      });
    });
    let data = dir.join("Data/com.apple.CoreML");
    fs::write(data.join("model.mlmodel"), &model.0)?;
    fs::write(dir.join("Manifest.json"), MANIFEST)
  }
}

/// Fixed item ids: a package holds one model and one weight folder.
const MANIFEST: &str = r#"{
  "fileFormatVersion": "1.0.0",
  "itemInfoEntries": {
    "5A8E2D1C-0B6F-4E3A-9C7D-1F2E3A4B5C6D": {
      "author": "com.apple.CoreML",
      "description": "CoreML Model Specification",
      "name": "model.mlmodel",
      "path": "com.apple.CoreML/model.mlmodel"
    },
    "6B9F3E2D-1C7A-4F4B-8D8E-2A3B4C5D6E7F": {
      "author": "com.apple.CoreML",
      "description": "CoreML Model Weights",
      "name": "weights",
      "path": "com.apple.CoreML/weights"
    }
  },
  "rootModelIdentifier": "5A8E2D1C-0B6F-4E3A-9C7D-1F2E3A4B5C6D"
}
"#;

/// Per row of `cols` weights: an F16 scale (the largest magnitude / 127) and the weights divided
/// by it, rounded to int8.
pub fn quantize(w: &[f32], cols: usize) -> (Vec<i8>, Vec<f16>) {
  let mut q = Vec::with_capacity(w.len());
  let scales = (w.chunks_exact(cols))
    .map(|row| {
      let scale = f16::from_f32(row.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0);
      let inv = if scale.to_f32() > 0.0 { 1.0 / scale.to_f32() } else { 0.0 };
      q.extend(row.iter().map(|x| (x * inv).round().clamp(-127.0, 127.0) as i8));
      scale
    })
    .collect();
  (q, scales)
}

/// `ValueType` of a tensor.
fn value_type(m: &mut Msg, dtype: Dtype, shape: &[usize]) {
  m.msg(1, |t| {
    t.int(1, dtype.mil()).int(2, shape.len() as i64);
    shape.iter().for_each(|&d| _ = t.msg(3, |dim| _ = dim.msg(1, |c| _ = c.int(1, d as i64))));
  });
}

/// `NamedValueType` of a var.
fn named(m: &mut Msg, v: &Var) {
  m.str(1, &v.name).msg(2, |t| value_type(t, v.dtype, &v.shape));
}

/// `Value.immediateValue.tensor`, filled by `tensor`.
fn immediate(m: &mut Msg, tensor: impl FnOnce(&mut Msg)) {
  m.msg(3, |i| _ = i.msg(1, tensor));
}

/// `Value.blobFileValue`, without the value's type.
fn blob_file(m: &mut Msg, b: &Blob) {
  m.msg(5, |f| _ = f.str(1, WEIGHTS).int(2, b.offset as i64));
}

/// A `Value` in the weight file.
fn blob_value(m: &mut Msg, b: &Blob) {
  m.msg(2, |t| value_type(t, b.dtype, &b.shape));
  blob_file(m, b);
}

/// The `name` attribute every op carries.
fn name_attribute(op: &mut Msg, name: &str) {
  op.entry(5, "name", |v| {
    v.msg(2, |t| value_type(t, Dtype::Str, &[]));
    immediate(v, |t| _ = t.msg(4, |r| _ = r.str(1, name)));
  });
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn per_row_int8() {
    let w = [0.5, -1.0, 0.25, 0.0, 0.0, 0.0, 2.0, 1.0, -0.5];
    let (q, s) = quantize(&w, 3);
    assert_eq!(s[1], f16::ZERO, "an all-zero row");
    assert_eq!(&q[3..6], &[0, 0, 0]);
    assert_eq!((q[1], q[6]), (-127, 127), "the largest magnitude maps to ±127");
    for (row, (q, s)) in w.chunks(3).zip(q.chunks(3).zip(&s)) {
      for (x, q) in row.iter().zip(q) {
        assert!((*q as f32 * s.to_f32() - x).abs() <= s.to_f32() / 2.0 + 1e-6, "{x} {q} {s}");
      }
    }
  }

  #[test]
  fn weight_file_layout() {
    let dir = std::env::temp_dir().join(format!("wavo-mil-{}", std::process::id()));
    let weights = dir.join("Data/com.apple.CoreML/weights/weight.bin");
    let mut p = Program::new(&weights).unwrap();
    let x = p.input("x", &[2, 16]);
    let g = p.tensor(&[16], &[1.0; 16]);
    let s = p.scalar(0.5);
    let w = p.int8(3, 16, &[1.0; 48]);
    let y = p.op("mul", &[("x", &[&x]), ("y", &[&g])], vec![2, 16]);
    let y = p.op("mul", &[("x", &[&y]), ("y", &[&s])], vec![2, 16]);
    let y = p.op("linear", &[("x", &[&y]), ("weight", &[&w])], vec![2, 3]);
    p.finish(&[&y], &dir).unwrap();

    let b = fs::read(&weights).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    assert_eq!((u32_at(0), u32_at(4)), (3, 2), "three blobs, version 2");
    // F16 [16] at 64, its data at 128; int8 [3][16] at 192 (32 bytes padded to 64); its F16
    // scales [3][1] at 320. The scalar stays inline.
    for (record, dtype, size) in [(64, 1, 32), (192, 4, 48), (320, 1, 6)] {
      assert_eq!(u32_at(record), 0xDEAD_BEEF);
      assert_eq!((u32_at(record + 4), u64_at(record + 8)), (dtype, size));
      assert_eq!(u64_at(record + 16), record as u64 + 64);
    }
    assert_eq!(&b[128..130], &f16::ONE.to_le_bytes());
    assert_eq!(b[256] as i8, 127);
    assert_eq!(b.len(), 448);

    let model = fs::read(dir.join("Data/com.apple.CoreML/model.mlmodel")).unwrap();
    assert_eq!(&model[..2], &[0x08, 9], "specificationVersion 9");
    let has = |s: &[u8]| model.windows(s.len()).any(|w| w == s);
    assert!(has(b"constexpr_blockwise_shift_scale") && has(b"CoreML8") && has(WEIGHTS.as_bytes()));
    assert!(fs::read_to_string(dir.join("Manifest.json")).unwrap().contains("model.mlmodel"));
    fs::remove_dir_all(&dir).unwrap();
  }
}
