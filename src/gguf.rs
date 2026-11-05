// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GGUF v3 reader: metadata from the header, F32/F16/Q8_0/Q5_0/Q4_0/Q4_1/Q5_1/Q4_K tensors
//! streamed from the file.

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
  /// Blocks of 32 weights: f16 scale and 16 bytes of nibbles (18 bytes). A weight is
  /// `(code - 8) · scale`, with `code` in 0..15.
  Q4_0,
  /// Blocks of 32 weights: f16 scale, f16 minimum, 16 bytes of nibbles (20 bytes). A weight is
  /// `code · scale + minimum`, with `code` in 0..15.
  Q4_1,
  /// Blocks of 32 weights: f16 scale, f16 minimum, 4 high-bit bytes, 16 low nibbles (24 bytes). A
  /// weight is `code · scale + minimum`, with `code` in 0..31.
  Q5_1,
  /// Blocks of 256 weights: f16 superblock scale, f16 superblock minimum, 12 bytes of 6-bit
  /// scales and minimums, 128 bytes of nibbles (144 bytes). A weight is
  /// `code · d · scale - dmin · min`, with `code` in 0..15 and `scale` and `min` in 0..63. The
  /// element count is a multiple of 256. Blocks follow weight order, so a row whose length is not
  /// a multiple of 256 shares a block with the next row.
  #[allow(non_camel_case_types)]
  Q4_K,
}

impl Type {
  fn from_ggml(id: u32) -> Option<Self> {
    match id {
      0 => Some(Self::F32),
      1 => Some(Self::F16),
      2 => Some(Self::Q4_0),
      3 => Some(Self::Q4_1),
      6 => Some(Self::Q5_0),
      7 => Some(Self::Q5_1),
      8 => Some(Self::Q8_0),
      12 => Some(Self::Q4_K),
      _ => None,
    }
  }

  fn size(self, n: usize) -> usize {
    match self {
      Self::F32 => n * 4,
      Self::F16 => n * 2,
      Self::Q8_0 => n / 32 * 34,
      Self::Q5_0 => n / 32 * 22,
      Self::Q4_0 => n / 32 * 18,
      Self::Q4_1 => n / 32 * 20,
      Self::Q5_1 => n / 32 * 24,
      Self::Q4_K => n / 256 * 144,
    }
  }
}

/// Codes 0..31 from a Q5 high-bit word and the sixteen low nibbles that follow it.
fn q5_codes_from(qh_qs: &[u8]) -> [u8; 32] {
  let qh = u32::from_le_bytes([qh_qs[0], qh_qs[1], qh_qs[2], qh_qs[3]]);
  let mut codes = [0u8; 32];
  for j in 0..16 {
    let byte = qh_qs[4 + j];
    codes[j] = (byte & 0x0f) | (((qh >> j) & 1) << 4) as u8;
    codes[j + 16] = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4) as u8;
  }
  codes
}

/// Codes 0..31 of one ggml Q5_0 block, in weight order. The block is an f16 scale, four high-bit
/// bytes, then sixteen low nibbles.
pub(crate) fn q5_codes(block: &[u8; 22]) -> [u8; 32] {
  q5_codes_from(&block[2..])
}

/// Codes 0..31 of one ggml Q5_1 block. The block is an f16 scale, an f16 minimum, then the same
/// high bits and low nibbles as Q5_0.
pub(crate) fn q5_1_codes(block: &[u8; 24]) -> [u8; 32] {
  q5_codes_from(&block[4..])
}

/// `codes` packed little-endian, five bits each, then four zero bytes. A group of four starts at
/// bit `group * 20`, and the padding keeps that group's load inside the 24 bytes.
fn pack5(codes: [u8; 32]) -> [u8; 24] {
  let mut out = [0u8; 24];
  let mut acc = 0u64;
  let mut bits = 0;
  let mut at = 0;
  for code in codes {
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

/// The Q5_0 block's codes in the packed layout `pack5` describes.
pub(crate) fn q5_packed(block: &[u8; 22]) -> [u8; 24] {
  pack5(q5_codes(block))
}

/// The Q5_1 block's codes in the same packed layout.
pub(crate) fn q5_1_packed(block: &[u8; 24]) -> [u8; 24] {
  pack5(q5_1_codes(block))
}

/// Nibbles of one Q4 block, in weight order. `qs` stores weight `j` in the low nibble and weight
/// `j + 16` in the high nibble.
pub(crate) fn q4_codes(qs: &[u8; 16]) -> [u8; 32] {
  let mut codes = [0u8; 32];
  for j in 0..16 {
    codes[j] = qs[j] & 0x0f;
    codes[j + 16] = qs[j] >> 4;
  }
  codes
}

/// `qs` repacked so weight `i` is the nibble at bit `i * 4`. Sixteen bytes, no padding.
pub(crate) fn q4_packed(qs: &[u8; 16]) -> [u8; 16] {
  let mut out = [0u8; 16];
  for (i, code) in q4_codes(qs).into_iter().enumerate() {
    out[i / 2] |= code << ((i % 2) * 4);
  }
  out
}

/// Q4_0 codes in the `q4_packed` layout. The block is an f16 scale, then the nibbles.
pub(crate) fn q4_0_packed(block: &[u8; 18]) -> [u8; 16] {
  let qs: &[u8; 16] = block[2..].try_into().unwrap();
  q4_packed(qs)
}

/// Q4_1 codes in the `q4_packed` layout. The block is an f16 scale, an f16 minimum, then the
/// nibbles.
pub(crate) fn q4_1_packed(block: &[u8; 20]) -> [u8; 16] {
  let qs: &[u8; 16] = block[4..].try_into().unwrap();
  q4_packed(qs)
}

/// 6-bit scale and minimum of sub-block `j` (0..8) from a Q4_K block's 12 scale bytes.
pub(crate) fn scale_min_k4(scales: &[u8; 12], j: usize) -> (u8, u8) {
  if j < 4 {
    (scales[j] & 63, scales[j + 4] & 63)
  } else {
    let scale = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
    let min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    (scale, min)
  }
}

/// Eight 6-bit scales and eight 6-bit minimums packed into the 12 byte Q4_K scale field.
#[cfg(test)]
fn pack_scales_k4(scale: &[u8; 8], min: &[u8; 8]) -> [u8; 12] {
  let mut s = [0u8; 12];
  for j in 0..8 {
    if j < 4 {
      s[j] = scale[j];
      s[j + 4] = min[j];
    } else {
      s[j + 4] = (scale[j] & 0x0f) | ((min[j] & 0x0f) << 4);
      s[j - 4] |= (scale[j] >> 4) << 6;
      s[j] |= (min[j] >> 4) << 6;
    }
  }
  s
}

/// Codes 0..15 of one Q4_K block, in weight order. Each 32 bytes hold one group of 32 in the low
/// nibbles and the next group in the high nibbles.
pub(crate) fn q4k_codes(qs: &[u8; 128]) -> [u8; 256] {
  let mut codes = [0u8; 256];
  for j in (0..256).step_by(64) {
    for l in 0..32 {
      let byte = qs[j / 2 + l];
      codes[j + l] = byte & 0x0f;
      codes[j + l + 32] = byte >> 4;
    }
  }
  codes
}

#[cfg(test)]
fn pack_qs_k4(codes: &[u8; 256]) -> [u8; 128] {
  let mut qs = [0u8; 128];
  for j in (0..256).step_by(64) {
    for l in 0..32 {
      qs[j / 2 + l] = (codes[j + l] & 0x0f) | (codes[j + l + 32] << 4);
    }
  }
  qs
}

/// Q4_K nibbles in weight order, 128 bytes per 256 weights.
pub(crate) fn q4k_packed(block: &[u8; 144]) -> [u8; 128] {
  let qs: &[u8; 128] = block[16..].try_into().unwrap();
  let codes = q4k_codes(qs);
  let mut out = [0u8; 128];
  for (i, code) in codes.into_iter().enumerate() {
    out[i / 2] |= code << ((i % 2) * 4);
  }
  out
}

/// Superblock f16 scale and minimum, then the eight decoded scales and eight decoded minimums.
pub(crate) fn q4k_scales(block: &[u8; 144]) -> [u8; 20] {
  let mut out = [0u8; 20];
  out[..4].copy_from_slice(&block[..4]);
  let scales: &[u8; 12] = block[4..16].try_into().unwrap();
  for j in 0..8 {
    let (scale, min) = scale_min_k4(scales, j);
    out[4 + j] = scale;
    out[12 + j] = min;
  }
  out
}

/// Largest-magnitude element of a block, the first one when two tie.
#[cfg(test)]
fn first_max_abs(block: &[f32; 32]) -> f32 {
  let mut max = block[0];
  for &x in &block[1..] {
    if x.abs() > max.abs() {
      max = x;
    }
  }
  max
}

/// ggml's symmetric quantizer: scale is `max / -half` (Q4_0 uses 8, Q5_0 uses 16), and a code is
/// `trunc(x / scale + half + 0.5)` clipped to the code range.
#[cfg(test)]
fn symmetric_codes(block: &[f32; 32], half: f32) -> (f32, [u8; 32]) {
  let d = first_max_abs(block) / -half;
  let id = if d == 0.0 { 0.0 } else { 1.0 / d };
  let mut codes = [0u8; 32];
  for (code, &x) in codes.iter_mut().zip(block) {
    *code = (x * id + half + 0.5).trunc().clamp(0.0, half * 2.0 - 1.0) as u8;
  }
  (d, codes)
}

/// ggml's affine quantizer: scale is `(max - min) / steps` and the minimum is stored beside it.
/// A code is `trunc((x - min) / scale + 0.5)` clipped to `0..=steps`.
#[cfg(test)]
fn affine_codes(block: &[f32; 32], steps: f32) -> (f32, f32, [u8; 32]) {
  let (mut min, mut max) = (block[0], block[0]);
  for &x in &block[1..] {
    if x < min {
      min = x;
    }
    if x > max {
      max = x;
    }
  }
  let d = (max - min) / steps;
  let id = if d == 0.0 { 0.0 } else { 1.0 / d };
  let mut codes = [0u8; 32];
  for (code, &x) in codes.iter_mut().zip(block) {
    *code = ((x - min) * id + 0.5).trunc().clamp(0.0, steps) as u8;
  }
  (d, min, codes)
}

/// Low nibble of code `j` beside the low nibble of code `j + 16`.
#[cfg(test)]
fn nibble_pairs(codes: &[u8; 32]) -> [u8; 16] {
  let mut qs = [0u8; 16];
  for j in 0..16 {
    qs[j] = (codes[j] & 0x0f) | (codes[j + 16] << 4);
  }
  qs
}

/// High bit of each Q5 code, packed little-endian, and the low nibbles.
#[cfg(test)]
fn q5_file_bits(codes: &[u8; 32]) -> (u32, [u8; 16]) {
  let mut qh = 0u32;
  for j in 0..16 {
    qh |= ((codes[j] >> 4) as u32) << j;
    qh |= ((codes[j + 16] >> 4) as u32) << (j + 16);
  }
  (qh, nibble_pairs(codes))
}

/// ggml Q4_0 over `w`, whose length is a multiple of 32. The scale is stored as f16.
#[cfg(test)]
pub(crate) fn quantize_q4_0(w: &[f32]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(32));
  let mut out = Vec::with_capacity(w.len() / 32 * 18);
  for block in w.as_chunks::<32>().0 {
    let (d, codes) = symmetric_codes(block, 8.0);
    out.extend(f16::from_f32(d).to_le_bytes());
    out.extend(nibble_pairs(&codes));
  }
  out
}

/// ggml Q4_1 over `w`, whose length is a multiple of 32. The scale and the minimum are stored as
/// f16.
#[cfg(test)]
pub(crate) fn quantize_q4_1(w: &[f32]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(32));
  let mut out = Vec::with_capacity(w.len() / 32 * 20);
  for block in w.as_chunks::<32>().0 {
    let (d, min, codes) = affine_codes(block, 15.0);
    out.extend(f16::from_f32(d).to_le_bytes());
    out.extend(f16::from_f32(min).to_le_bytes());
    out.extend(nibble_pairs(&codes));
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
    let (d, codes) = symmetric_codes(block, 16.0);
    let (qh, qs) = q5_file_bits(&codes);
    out.extend(f16::from_f32(d).to_le_bytes());
    out.extend(qh.to_le_bytes());
    out.extend(qs);
  }
  out
}

/// ggml Q5_1 over `w`, whose length is a multiple of 32. The scale and the minimum are stored as
/// f16.
#[cfg(test)]
pub(crate) fn quantize_q5_1(w: &[f32]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(32));
  let mut out = Vec::with_capacity(w.len() / 32 * 24);
  for block in w.as_chunks::<32>().0 {
    let (d, min, codes) = affine_codes(block, 31.0);
    let (qh, qs) = q5_file_bits(&codes);
    out.extend(f16::from_f32(d).to_le_bytes());
    out.extend(f16::from_f32(min).to_le_bytes());
    out.extend(qh.to_le_bytes());
    out.extend(qs);
  }
  out
}

/// ggml Q4_K over `w`, whose length is a multiple of 256. Each group of 32 is an affine 4-bit
/// grid whose scale and minimum are then quantized to 6 bits against the largest in the block of
/// 256. A positive group minimum is pulled down to 0, so the stored minimum stays non-negative.
#[cfg(test)]
pub(crate) fn quantize_q4_k(w: &[f32]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(256));
  let mut out = Vec::with_capacity(w.len() / 256 * 144);
  for block in w.as_chunks::<256>().0 {
    let mut group_scale = [0f32; 8];
    let mut group_min = [0f32; 8];
    for j in 0..8 {
      let (scale, stored, _) = rtn_group(&block[j * 32..j * 32 + 32]);
      group_scale[j] = scale;
      group_min[j] = stored;
    }
    out.extend(pack_q4k(&group_scale, &group_min, block));
  }
  out
}

/// Q4_K with a per-input-channel importance. `channel_w` has one non-negative weight per column
/// of a row (`w.len() / row` rows). A group of 32 keeps the min/max grid, then tries nearby code
/// scales and keeps the one with the smaller weighted squared error. The 6-bit superblock scales
/// are the same as [`quantize_q4_k`].
#[cfg(test)]
pub(crate) fn quantize_q4_k_weighted(w: &[f32], row: usize, channel_w: &[f32]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(256) && row.is_multiple_of(32));
  debug_assert_eq!(channel_w.len(), row);
  let mut out = Vec::with_capacity(w.len() / 256 * 144);
  for (b, block) in w.as_chunks::<256>().0.iter().enumerate() {
    let origin = b * 256;
    let mut group_scale = [0f32; 8];
    let mut group_min = [0f32; 8];
    for j in 0..8 {
      let mut xs = [0f32; 32];
      let mut ws = [0f32; 32];
      for l in 0..32 {
        let i = origin + j * 32 + l;
        xs[l] = block[j * 32 + l];
        ws[l] = channel_w[i % row];
      }
      let (scale, stored, _) = weighted_group(&xs, &ws);
      group_scale[j] = scale;
      group_min[j] = stored;
    }
    out.extend(pack_q4k(&group_scale, &group_min, block));
  }
  out
}

/// Q4_K after a group-wise Hessian update along each row. `hinv` is the inverse of the damped
/// input Gram matrix, row-major, `row * row`. Each group of 32 is quantized with the round-to-
/// nearest grid. The remaining columns of that row move by `Hinv` so the update cancels that
/// group's error in the least-squares sense. Columns already visited stay as they were. A diagonal
/// `hinv` leaves the weights unchanged, so the bytes match [`quantize_q4_k`].
#[cfg(test)]
pub(crate) fn quantize_q4_k_gptq(w: &[f32], row: usize, hinv: &[f64]) -> Vec<u8> {
  debug_assert!(w.len().is_multiple_of(row) && row.is_multiple_of(32));
  debug_assert_eq!(hinv.len(), row * row);
  let mut data: Vec<f64> = w.iter().map(|&v| v as f64).collect();
  let rows = w.len() / row;
  for r in 0..rows {
    let values = &mut data[r * row..(r + 1) * row];
    for g in 0..row / 32 {
      let i0 = g * 32;
      let mut block = [0f32; 32];
      for t in 0..32 {
        block[t] = values[i0 + t] as f32;
      }
      let (scale, stored, codes) = rtn_group(&block);
      let mut delta = [0f64; 32];
      for t in 0..32 {
        let rec = codes[t] as f64 * scale as f64 - stored as f64;
        let pivot = hinv[(i0 + t) * row + i0 + t];
        delta[t] = if pivot.abs() < 1e-18 { 0.0 } else { (values[i0 + t] - rec) / pivot };
      }
      for t in 0..32 {
        let d = delta[t];
        if d == 0.0 {
          continue;
        }
        let hrow = &hinv[(i0 + t) * row + i0 + 32..(i0 + t) * row + row];
        for (slot, &h) in values[i0 + 32..].iter_mut().zip(hrow) {
          *slot -= d * h;
        }
      }
    }
  }
  quantize_q4_k(&data.iter().map(|&v| v as f32).collect::<Vec<_>>())
}

/// Round-to-nearest group: scale `(max - min) / 15`, stored minimum `-min` (0 when `min > 0`).
#[cfg(test)]
fn rtn_group(sub: &[f32]) -> (f32, f32, [u8; 32]) {
  let (mut min, mut max) = (sub[0], sub[0]);
  for &x in &sub[1..] {
    if x < min {
      min = x;
    }
    if x > max {
      max = x;
    }
  }
  if min > 0.0 {
    min = 0.0;
  }
  let scale = if max == min { 0.0 } else { (max - min) / 15.0 };
  let stored = -min;
  let mut codes = [0u8; 32];
  if scale != 0.0 {
    let id = 1.0 / scale;
    for (code, &x) in codes.iter_mut().zip(sub) {
      *code = ((x + stored) * id + 0.5).trunc().clamp(0.0, 15.0) as u8;
    }
  }
  (scale, stored, codes)
}

/// Weighted squared error over candidate scales. The code assignment still uses the group's
/// min and max; the weights choose the scale and the minimum among those candidates.
#[cfg(test)]
fn weighted_group(x: &[f32; 32], w: &[f32; 32]) -> (f32, f32, [u8; 32]) {
  let (scale, stored, codes) = rtn_group(x);
  let min = -stored;
  let max = x.iter().copied().fold(x[0], f32::max);
  if max == min {
    return (scale, stored, codes);
  }
  let mut best_scale = scale;
  let mut best_min = min;
  let mut best_codes = codes;
  let mut best_err = f64::MAX;
  let mut consider = |iscale: f32| {
    let mut codes = [0u8; 32];
    let (mut sum_w, mut sum_x, mut sum_l, mut sum_l2, mut sum_xl) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..32 {
      // Half up, so the min/max scale assigns the same codes as round-to-nearest.
      let l = (iscale * (x[i] - min) + 0.5).trunc().clamp(0.0, 15.0);
      codes[i] = l as u8;
      let wi = w[i].max(0.0) as f64;
      let li = l as f64;
      sum_w += wi;
      sum_x += wi * x[i] as f64;
      sum_l += wi * li;
      sum_l2 += wi * li * li;
      sum_xl += wi * li * x[i] as f64;
    }
    let det = sum_w * sum_l2 - sum_l * sum_l;
    let (scale, signed) = if det > 1e-12 {
      let mut scale = (sum_w * sum_xl - sum_x * sum_l) / det;
      let mut signed = (sum_l2 * sum_x - sum_l * sum_xl) / det;
      if signed > 0.0 {
        signed = 0.0;
        scale = if sum_l2 > 0.0 { sum_xl / sum_l2 } else { 0.0 };
      }
      (scale.max(0.0), signed)
    } else {
      (best_scale as f64, best_min as f64)
    };
    let mut err = 0.0;
    for i in 0..32 {
      let diff = scale * codes[i] as f64 + signed - x[i] as f64;
      err += w[i].max(0.0) as f64 * diff * diff;
    }
    if err < best_err {
      best_err = err;
      best_scale = scale as f32;
      best_min = signed as f32;
      best_codes = codes;
    }
  };
  let span = max - min;
  consider(15.0 / span);
  for step in 0..=20 {
    consider((-1.0 + 0.1 * step as f32 + 15.0) / span);
  }
  (best_scale, -best_min, best_codes)
}

/// One Q4_K block from the eight group scales and stored minimums.
#[cfg(test)]
fn pack_q4k(group_scale: &[f32; 8], group_min: &[f32; 8], block: &[f32; 256]) -> [u8; 144] {
  let max_scale = group_scale.iter().copied().fold(0.0, f32::max);
  let max_min = group_min.iter().copied().fold(0.0, f32::max);
  let d = f16::from_f32(if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 });
  let dmin = f16::from_f32(if max_min > 0.0 { max_min / 63.0 } else { 0.0 });
  let (d_s, m_s) = (d.to_f32(), dmin.to_f32());
  let q6 = |x: f32| (x + 0.5).trunc().clamp(0.0, 63.0) as u8;
  let mut scale = [0u8; 8];
  let mut min = [0u8; 8];
  for j in 0..8 {
    scale[j] = if d_s == 0.0 { 0 } else { q6(group_scale[j] / d_s) };
    min[j] = if m_s == 0.0 { 0 } else { q6(group_min[j] / m_s) };
  }
  let mut codes = [0u8; 256];
  for j in 0..8 {
    let dq = d_s * scale[j] as f32;
    if dq == 0.0 {
      continue;
    }
    let dm = m_s * min[j] as f32;
    let id = 1.0 / dq;
    for l in 0..32 {
      let x = block[j * 32 + l];
      codes[j * 32 + l] = ((x + dm) * id + 0.5).trunc().clamp(0.0, 15.0) as u8;
    }
  }
  let mut out = [0u8; 144];
  out[..2].copy_from_slice(&d.to_le_bytes());
  out[2..4].copy_from_slice(&dmin.to_le_bytes());
  out[4..16].copy_from_slice(&pack_scales_k4(&scale, &min));
  out[16..].copy_from_slice(&pack_qs_k4(&codes));
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
      let n = t.dims.iter().product::<usize>();
      if t.ty == Type::Q4_K && !n.is_multiple_of(256) {
        bail!("tensor {name}: Q4_K has {n} weights, not a multiple of 256");
      }
      if g.data + t.offset + t.ty.size(n) > len {
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
      Type::Q4_0 => (b.as_chunks::<18>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          let qs: &[u8; 16] = block[2..].try_into().unwrap();
          q4_codes(qs).into_iter().map(move |c| (c as i32 - 8) as f32 * d)
        })
        .collect(),
      Type::Q4_1 => (b.as_chunks::<20>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
          let qs: &[u8; 16] = block[4..].try_into().unwrap();
          q4_codes(qs).into_iter().map(move |c| c as f32 * d + m)
        })
        .collect(),
      Type::Q5_1 => (b.as_chunks::<24>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
          q5_1_codes(block).into_iter().map(move |c| c as f32 * d + m)
        })
        .collect(),
      Type::Q4_K => (b.as_chunks::<144>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          let min = f16::from_le_bytes([block[2], block[3]]).to_f32();
          let scales: &[u8; 12] = block[4..16].try_into().unwrap();
          let qs: &[u8; 128] = block[16..].try_into().unwrap();
          let codes = q4k_codes(qs);
          (0..8).flat_map(move |j| {
            let (scale, m) = scale_min_k4(scales, j);
            let (d1, m1) = (d * scale as f32, min * m as f32);
            (0..32).map(move |l| d1 * codes[j * 32 + l] as f32 - m1)
          })
        })
        .collect(),
    })
  }
}

/// Q8_0 tensors rewritten as Q5_0 by [`write_q5_0`].
#[cfg(test)]
pub(crate) struct Q5Write {
  /// Tensors whose row length is a multiple of 32.
  pub converted: usize,
  /// Weights in those tensors.
  pub elements: usize,
  /// Bytes of the source file.
  pub src_bytes: u64,
  /// Bytes of the written file.
  pub dst_bytes: u64,
}

/// Q8_0 tensors rewritten by [`write_q4_k`].
#[cfg(test)]
pub(crate) struct Q4Write {
  /// Tensors stored as Q4_K.
  pub q4: usize,
  /// Weights in those tensors.
  pub q4_elements: usize,
  /// Tensors stored as Q5_0.
  pub q5: usize,
  /// Weights in those tensors.
  pub q5_elements: usize,
  /// Bytes of the source file.
  pub src_bytes: u64,
  /// Bytes of the written file.
  pub dst_bytes: u64,
}

/// Writes `src` to `dst`, re-quantizing each Q8_0 tensor to Q5_0 when its row length
/// (`len / dims.last()`) is a multiple of 32. The codes are round-to-nearest on the dequantized
/// Q8_0 values. Other tensors are copied. Metadata is copied; a `general.file_type` of 7 (mostly
/// Q8_0) becomes 8 (mostly Q5_0).
#[cfg(test)]
pub(crate) fn write_q5_0(src: &Path, dst: &Path) -> Result<Q5Write> {
  let wrote = rewrite(
    src,
    dst,
    |_, dims| {
      let row = row_len(dims);
      (row >= 32 && row.is_multiple_of(32)).then_some(Type::Q5_0)
    },
    |_, _, w| quantize_q4_k(w),
    Label::Q5,
  )?;
  Ok(Q5Write {
    converted: wrote.converted,
    elements: wrote.elements,
    src_bytes: wrote.src_bytes,
    dst_bytes: wrote.dst_bytes,
  })
}

/// Writes `src` to `dst`. A Q8_0 tensor whose row length is a multiple of 32 becomes Q4_K when
/// it holds a multiple of 256 weights and `q5` returns false. It becomes Q5_0 when `q5` returns
/// true, and when the count is not a multiple of 256. Blocks cover the flat tensor. The
/// contiguous axis does not have to be a multiple of 256. The codes are round-to-nearest on the
/// dequantized Q8_0 values. Other tensors are copied. A `general.file_type` of 7 becomes 14 when
/// every converted tensor is Q4_K, and 15 when any is Q5_0.
#[cfg(test)]
pub(crate) fn write_q4_k(src: &Path, dst: &Path, q5: impl Fn(&str) -> bool) -> Result<Q4Write> {
  write_q4_k_with(src, dst, q5, |_, _, w| quantize_q4_k(w))
}

/// [`write_q4_k`], with `quantize` producing the Q4_K bytes of one tensor. `row` is that tensor's
/// row length. The type choice is unchanged: Q4_K when the tensor holds a multiple of 256 weights
/// and `q5` returns false, otherwise Q5_0.
#[cfg(test)]
pub(crate) fn write_q4_k_with(
  src: &Path,
  dst: &Path,
  q5: impl Fn(&str) -> bool,
  quantize: impl Fn(&str, usize, &[f32]) -> Vec<u8>,
) -> Result<Q4Write> {
  let wrote = rewrite(
    src,
    dst,
    |name, dims| {
      let row = row_len(dims);
      if row < 32 || !row.is_multiple_of(32) {
        return None;
      }
      let n = dims.iter().product::<usize>();
      if q5(name) || !n.is_multiple_of(256) { Some(Type::Q5_0) } else { Some(Type::Q4_K) }
    },
    quantize,
    Label::Q4,
  )?;
  Ok(Q4Write {
    q4: wrote.q4,
    q4_elements: wrote.q4_elements,
    q5: wrote.q5,
    q5_elements: wrote.q5_elements,
    src_bytes: wrote.src_bytes,
    dst_bytes: wrote.dst_bytes,
  })
}

#[cfg(test)]
fn row_len(dims: &[usize]) -> usize {
  let n = dims.iter().product::<usize>();
  match dims.last() {
    Some(&d) if d > 0 => n / d,
    _ => 0,
  }
}

/// What a `general.file_type` of 7 is rewritten to.
#[cfg(test)]
enum Label {
  /// Mostly Q5_0, file type 8.
  Q5,
  /// Mostly Q4_K. File type 14 when every converted tensor is Q4_K, 15 when any is Q5_0.
  Q4,
}

#[cfg(test)]
struct Rewritten {
  converted: usize,
  elements: usize,
  q4: usize,
  q4_elements: usize,
  q5: usize,
  q5_elements: usize,
  src_bytes: u64,
  dst_bytes: u64,
}

/// Copies `src` to `dst`. `pick` chooses a new type for each Q8_0 tensor, or `None` to copy it.
#[cfg(test)]
fn rewrite(
  src: &Path,
  dst: &Path,
  pick: impl Fn(&str, &[usize]) -> Option<Type>,
  quantize_q4: impl Fn(&str, usize, &[f32]) -> Vec<u8>,
  label: Label,
) -> Result<Rewritten> {
  use std::io::Write;

  let src_bytes = std::fs::metadata(src)?.len();
  let g = Gguf::open(src)?;
  let align = match g.value("general.alignment") {
    Ok((_, Value::Int(a))) if a > 0 => a as usize,
    _ => 32,
  };

  let mut r = Reader { bytes: &g.head, pos: 0 };
  if r.take(4)? != b"GGUF" {
    bail!("not a GGUF file");
  }
  let version = r.u32()?;
  if version != 3 {
    bail!("unsupported GGUF version {version}");
  }
  let n_tensors = r.u64()? as usize;
  let n_meta = r.u64()? as usize;
  let mut file_type = None;
  for _ in 0..n_meta {
    let n = r.u64()? as usize;
    let key = std::str::from_utf8(r.take(n)?).unwrap_or("");
    let ty = r.u32()?;
    if key == "general.file_type" && (ty == 4 || ty == 5 || ty == 10 || ty == 11) {
      file_type = Some((r.pos, ty));
    }
    r.skip(ty)?;
  }
  let meta_end = r.pos;

  let mut converted = 0;
  let mut elements = 0;
  let mut q4 = 0;
  let mut q4_elements = 0;
  let mut q5 = 0;
  let mut q5_elements = 0;
  let mut tensors = Vec::with_capacity(n_tensors);
  for _ in 0..n_tensors {
    let name = r.string()?;
    let n_dims = r.u32()? as usize;
    let dims = (0..n_dims).map(|_| r.u64().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
    let id = r.u32()?;
    let Some(ty) = Type::from_ggml(id) else { bail!("tensor {name}: unsupported ggml type {id}") };
    _ = r.u64()?;
    let n = dims.iter().product::<usize>();
    let row = row_len(&dims);
    let tensor = g.tensor(&name, &dims)?;
    let dest = if ty == Type::Q8_0 { pick(&name, &dims) } else { None };
    let (ty, bytes) = if let Some(dest) = dest {
      let values = tensor.to_f32()?;
      let row_ok = row >= 32 && row.is_multiple_of(32);
      let bytes = match dest {
        Type::Q5_0 if row_ok => {
          q5 += 1;
          q5_elements += n;
          quantize_q5_0(&values)
        }
        Type::Q4_K if row_ok && n.is_multiple_of(256) => {
          q4 += 1;
          q4_elements += n;
          quantize_q4(&name, row, &values)
        }
        other => bail!("tensor {name}: {n} weights cannot be stored as {other:?}"),
      };
      converted += 1;
      elements += n;
      (dest, bytes)
    } else {
      let mut bytes = vec![0u8; ty.size(n)];
      if !bytes.is_empty() {
        tensor.read(bytes.len(), |off, piece| {
          bytes[off..off + piece.len()].copy_from_slice(piece);
        })?;
      }
      (ty, bytes)
    };
    if bytes.len() != ty.size(n) {
      bail!("tensor {name}: {} bytes for {n} {ty:?} weights", bytes.len());
    }
    tensors.push((name, dims, ty, bytes));
  }

  let mut offset = 0usize;
  let placed: Vec<_> = tensors
    .into_iter()
    .map(|(name, dims, ty, bytes)| {
      offset = offset.next_multiple_of(align);
      let at = offset;
      offset += bytes.len();
      (name, dims, ty, at, bytes)
    })
    .collect();

  let mut header = g.head[..meta_end].to_vec();
  let replacement = match label {
    Label::Q5 => Some(8u64),
    Label::Q4 if q5 > 0 => Some(15),
    Label::Q4 if q4 > 0 => Some(14),
    Label::Q4 => None,
  };
  if let Some(id) = replacement
    && let Some((pos, ty)) = file_type
  {
    let width = if ty == 10 || ty == 11 { 8 } else { 4 };
    let cur = if width == 8 {
      u64::from_le_bytes(header[pos..pos + 8].try_into().unwrap())
    } else {
      u32::from_le_bytes(header[pos..pos + 4].try_into().unwrap()) as u64
    };
    if cur == 7 {
      header[pos..pos + width].copy_from_slice(&id.to_le_bytes()[..width]);
    }
  }
  for (name, dims, ty, at, _) in &placed {
    let name = name.as_bytes();
    header.extend((name.len() as u64).to_le_bytes());
    header.extend(name);
    header.extend((dims.len() as u32).to_le_bytes());
    for d in dims {
      header.extend((*d as u64).to_le_bytes());
    }
    header.extend(ggml_id(*ty).to_le_bytes());
    header.extend((*at as u64).to_le_bytes());
  }
  header.resize(header.len().next_multiple_of(align), 0);

  let mut file = std::io::BufWriter::new(File::create(dst)?);
  file.write_all(&header)?;
  let mut cursor = 0usize;
  for (_, _, _, at, bytes) in &placed {
    if *at > cursor {
      file.write_all(&vec![0u8; at - cursor])?;
    }
    file.write_all(bytes)?;
    cursor = at + bytes.len();
  }
  file.flush()?;
  Ok(Rewritten {
    converted,
    elements,
    q4,
    q4_elements,
    q5,
    q5_elements,
    src_bytes,
    dst_bytes: (header.len() + cursor) as u64,
  })
}

/// ggml type id of `ty`.
#[cfg(test)]
fn ggml_id(ty: Type) -> u32 {
  match ty {
    Type::F32 => 0,
    Type::F16 => 1,
    Type::Q4_0 => 2,
    Type::Q4_1 => 3,
    Type::Q5_0 => 6,
    Type::Q5_1 => 7,
    Type::Q8_0 => 8,
    Type::Q4_K => 12,
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
    let mut off_grid = grid.clone();
    off_grid[1] = 2.7;
    let q = quantize_q5_0(&off_grid);
    let g = Gguf::with_tensors(&[("o", &[32], Type::Q5_0, &q)]);
    assert_eq!(g.tensor("o", &[32]).unwrap().to_f32().unwrap()[1], 3.0);

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
  fn q5_0_second_half_high_bits() {
    let codes: [u8; 32] = std::array::from_fn(|i| ((i * 7) % 32) as u8);
    let block = q5_block(codes, 0.5);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q5_0, &block)]);
    let expected: Vec<f32> = codes.iter().map(|&c| (c as f32 - 16.0) * 0.5).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);
  }

  /// One Q4_0 block: f16 scale, then weight `j` in the low nibble and `j + 16` in the high one.
  fn q4_0_block(codes: [u8; 32], scale: f32) -> [u8; 18] {
    let mut block = [0u8; 18];
    block[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    for j in 0..16 {
      block[2 + j] = (codes[j] & 0x0f) | (codes[j + 16] << 4);
    }
    block
  }

  /// One Q4_1 block: f16 scale, f16 minimum, then the same nibbles.
  fn q4_1_block(codes: [u8; 32], scale: f32, min: f32) -> [u8; 20] {
    let mut block = [0u8; 20];
    block[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(min).to_le_bytes());
    for j in 0..16 {
      block[4 + j] = (codes[j] & 0x0f) | (codes[j + 16] << 4);
    }
    block
  }

  /// One Q5_1 block: f16 scale, f16 minimum, high bits, then the low nibbles.
  fn q5_1_block(codes: [u8; 32], scale: f32, min: f32) -> [u8; 24] {
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
      qs[j] = (codes[j] & 0x0f) | (codes[j + 16] << 4);
      qh |= ((codes[j] >> 4) as u32) << j;
      qh |= ((codes[j + 16] >> 4) as u32) << (j + 16);
    }
    let mut block = [0u8; 24];
    block[0..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(min).to_le_bytes());
    block[4..8].copy_from_slice(&qh.to_le_bytes());
    block[8..24].copy_from_slice(&qs);
    block
  }

  /// The shader's Q4 group load: four codes at `group`, from the 16-byte pack.
  fn q4_group(packed: &[u8; 16], group: usize) -> [u8; 4] {
    let local = group * 4;
    let word = u32::from_le_bytes(packed[local / 8 * 4..][..4].try_into().unwrap());
    let bits = (word >> ((local % 8) * 4)) & 0xffff;
    [
      (bits & 15) as u8,
      ((bits >> 4) & 15) as u8,
      ((bits >> 8) & 15) as u8,
      ((bits >> 12) & 15) as u8,
    ]
  }

  #[test]
  fn q4_0_round_trips() {
    let codes = std::array::from_fn(|i| (i % 16) as u8);
    let block = q4_0_block(codes, 0.5);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q4_0, &block)]);
    let expected: Vec<f32> = (0..32).map(|i| ((i % 16) as f32 - 8.0) * 0.5).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);
    let distinct: [u8; 32] = std::array::from_fn(|i| if i < 16 { i as u8 } else { (31 - i) as u8 });
    let distinct_block = q4_0_block(distinct, 0.5);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q4_0, &distinct_block)]);
    let expected: Vec<f32> = distinct.iter().map(|&c| (c as f32 - 8.0) * 0.5).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);

    // −8..7 twice is an exact grid: the scale is 1 and each half is codes 0..15.
    let grid: Vec<f32> = (0..32).map(|i| (i % 16) as f32 - 8.0).collect();
    assert_eq!(quantize_q4_0(&grid), q4_0_block(codes, 1.0));
    let packed = q4_0_packed(quantize_q4_0(&grid).as_slice().try_into().unwrap());
    for group in 0..8 {
      assert_eq!(&q4_group(&packed, group), &codes[group * 4..group * 4 + 4]);
    }

    let zeros = quantize_q4_0(&[0.0; 32]);
    let z: &[u8; 18] = zeros.as_slice().try_into().unwrap();
    assert_eq!(q4_codes(z[2..].try_into().unwrap()), [8; 32]);
    assert_eq!(f16::from_le_bytes([zeros[0], zeros[1]]).to_f32(), 0.0);
    let g = Gguf::with_tensors(&[("z", &[32], Type::Q4_0, &zeros)]);
    assert_eq!(g.tensor("z", &[32]).unwrap().to_f32().unwrap(), vec![0.0; 32]);

    // The first of two equal magnitudes wins, so the scale is negative and −8 clips to −7.
    let mut tied = [0.0; 32];
    tied[0] = 8.0;
    tied[1] = -8.0;
    let got = Gguf::with_tensors(&[("t", &[32], Type::Q4_0, &quantize_q4_0(&tied))])
      .tensor("t", &[32])
      .unwrap()
      .to_f32()
      .unwrap();
    assert_eq!(got[0], 8.0);
    assert_eq!(got[1], -7.0);
  }

  #[test]
  fn q4_1_round_trips() {
    let codes = std::array::from_fn(|i| (i % 16) as u8);
    let block = q4_1_block(codes, 1.0, -1.0);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q4_1, &block)]);
    let expected: Vec<f32> = (0..32).map(|i| (i % 16) as f32 - 1.0).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);
    let distinct: [u8; 32] = std::array::from_fn(|i| if i < 16 { i as u8 } else { (31 - i) as u8 });
    let distinct_block = q4_1_block(distinct, 0.5, -2.0);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q4_1, &distinct_block)]);
    let expected: Vec<f32> = distinct.iter().map(|&c| c as f32 * 0.5 - 2.0).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);

    let grid: Vec<f32> = (0..32).map(|i| (i % 16) as f32 - 1.0).collect();
    assert_eq!(quantize_q4_1(&grid), block);
    let packed = q4_1_packed(quantize_q4_1(&grid).as_slice().try_into().unwrap());
    for group in 0..8 {
      assert_eq!(&q4_group(&packed, group), &codes[group * 4..group * 4 + 4]);
    }

    // A constant block stores that value as the minimum and a zero scale.
    let flat = quantize_q4_1(&[2.5; 32]);
    let g = Gguf::with_tensors(&[("f", &[32], Type::Q4_1, &flat)]);
    assert_eq!(g.tensor("f", &[32]).unwrap().to_f32().unwrap(), vec![2.5; 32]);
  }

  #[test]
  fn q5_1_round_trips() {
    let codes = std::array::from_fn(|i| i as u8);
    let block = q5_1_block(codes, 0.5, -2.0);
    let g = Gguf::with_tensors(&[("q", &[32], Type::Q5_1, &block)]);
    let expected: Vec<f32> = (0..32).map(|i| i as f32 * 0.5 - 2.0).collect();
    assert_eq!(g.tensor("q", &[32]).unwrap().to_f32().unwrap(), expected);

    // 0..31 is an exact grid at scale 1 and minimum 0. −2 + 0.5·code is the other exact grid.
    let grid: Vec<f32> = (0..32).map(|i| i as f32).collect();
    assert_eq!(quantize_q5_1(&grid), q5_1_block(codes, 1.0, 0.0));
    let shifted: Vec<f32> = (0..32).map(|i| i as f32 * 0.5 - 2.0).collect();
    assert_eq!(quantize_q5_1(&shifted), block);
    let packed = q5_1_packed(block.as_slice().try_into().unwrap());
    for group in 0..8 {
      assert_eq!(&q5_group(&packed, group), &codes[group * 4..group * 4 + 4]);
    }

    let flat = quantize_q5_1(&[2.5; 32]);
    let g = Gguf::with_tensors(&[("f", &[32], Type::Q5_1, &flat)]);
    assert_eq!(g.tensor("f", &[32]).unwrap().to_f32().unwrap(), vec![2.5; 32]);
  }

  /// Two Q8_0 rows on the exact −16..15 grid become Q5_0. A one-dimensional Q8_0 block stays,
  /// because its row length is 1, and an F32 tensor is copied. `general.file_type` 7 becomes 8.
  #[test]
  fn write_q5_0_repacks_rows() {
    let grid: [i8; 32] = std::array::from_fn(|i| i as i8 - 16);
    let mut q8 = [0u8; 34];
    q8[..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
    for (i, v) in grid.iter().enumerate() {
      q8[2 + i] = *v as u8;
    }
    let weights = q8.repeat(2);
    let floats = [1.5f32, -2.0];
    let mut data = Vec::new();
    let place = |data: &mut Vec<u8>| {
      let at = data.len().next_multiple_of(32);
      data.resize(at, 0);
      at as u64
    };
    let w_at = place(&mut data);
    data.extend(&weights);
    let kept_at = place(&mut data);
    data.extend(q8);
    let bias_at = place(&mut data);
    data.extend(floats.iter().flat_map(|v| v.to_le_bytes()));

    let mut f = b"GGUF".to_vec();
    f.extend(3u32.to_le_bytes());
    f.extend(3u64.to_le_bytes());
    f.extend(1u64.to_le_bytes());
    string(&mut f, "general.file_type");
    f.extend(4u32.to_le_bytes());
    f.extend(7u32.to_le_bytes());
    let info = |f: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, at: u64| {
      string(f, name);
      f.extend((dims.len() as u32).to_le_bytes());
      for d in dims {
        f.extend(d.to_le_bytes());
      }
      f.extend(ty.to_le_bytes());
      f.extend(at.to_le_bytes());
    };
    info(&mut f, "w", &[32, 2], 8, w_at);
    info(&mut f, "kept", &[32], 8, kept_at);
    info(&mut f, "bias", &[2], 0, bias_at);
    f.resize(f.len().next_multiple_of(32), 0);
    f.extend(data);

    let dir = std::env::temp_dir();
    let src = dir.join(format!("wavo-q5-src-{}", std::process::id()));
    let dst = dir.join(format!("wavo-q5-dst-{}", std::process::id()));
    std::fs::write(&src, &f).unwrap();
    let wrote = write_q5_0(&src, &dst).unwrap();
    let g = Gguf::open(&dst).unwrap();
    assert_eq!((wrote.converted, wrote.elements), (1, 64));
    assert_eq!(wrote.src_bytes, std::fs::metadata(&src).unwrap().len());
    assert_eq!(wrote.dst_bytes, std::fs::metadata(&dst).unwrap().len());
    assert!(wrote.dst_bytes < wrote.src_bytes);
    assert_eq!(g.get::<u32>("general.file_type").unwrap(), 8);
    let expected: Vec<f32> = grid.iter().map(|&v| v as f32).cycle().take(64).collect();
    assert_eq!(g.tensor("w", &[32, 2]).unwrap().ty, Type::Q5_0);
    assert_eq!(g.tensor("w", &[32, 2]).unwrap().to_f32().unwrap(), expected);
    assert_eq!(g.tensor("kept", &[32]).unwrap().ty, Type::Q8_0);
    assert_eq!(g.tensor("kept", &[32]).unwrap().to_f32().unwrap(), &expected[..32]);
    assert_eq!(g.tensor("bias", &[2]).unwrap().to_f32().unwrap(), floats);
    std::fs::remove_file(src).unwrap();
    std::fs::remove_file(dst).unwrap();
  }

  /// Eight Q8_0 rows on the −16..15 grid become one Q4_K block. A tensor named in `q5` stays
  /// Q5_0, a 64-weight tensor is too short for Q4_K and stays Q5_0, a one-dimensional block is
  /// copied, and an F32 tensor is copied. `general.file_type` 7 becomes 15 for that mix and 14
  /// when every converted tensor is Q4_K.
  #[test]
  fn write_q4_k_repacks_blocks() {
    let grid: [i8; 32] = std::array::from_fn(|i| i as i8 - 16);
    let mut q8 = [0u8; 34];
    q8[..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
    for (i, v) in grid.iter().enumerate() {
      q8[2 + i] = *v as u8;
    }
    let weights = q8.repeat(8);
    let tail = q8.repeat(2);
    let floats = [1.5f32, -2.0];
    let values = |n: usize| grid.iter().map(|&v| v as f32).cycle().take(n).collect::<Vec<_>>();
    let dequant = |ty: Type, dims: &[usize], bytes: &[u8]| {
      Gguf::with_tensors(&[("t", dims, ty, bytes)]).tensor("t", dims).unwrap().to_f32().unwrap()
    };
    let q4 = dequant(Type::Q4_K, &[32, 8], &quantize_q4_k(&values(256)));
    let q5_wide = dequant(Type::Q5_0, &[32, 8], &quantize_q5_0(&values(256)));
    let q5_tail = dequant(Type::Q5_0, &[32, 2], &quantize_q5_0(&values(64)));

    let build = |parts: &[(&str, &[u64], u32, &[u8])]| {
      let mut data = Vec::new();
      let mut placed = Vec::new();
      for (name, dims, ty, bytes) in parts {
        let at = data.len().next_multiple_of(32);
        data.resize(at, 0);
        placed.push((*name, *dims, *ty, at as u64));
        data.extend(*bytes);
      }
      let mut f = b"GGUF".to_vec();
      f.extend(3u32.to_le_bytes());
      f.extend((parts.len() as u64).to_le_bytes());
      f.extend(1u64.to_le_bytes());
      string(&mut f, "general.file_type");
      f.extend(4u32.to_le_bytes());
      f.extend(7u32.to_le_bytes());
      for (name, dims, ty, at) in placed {
        string(&mut f, name);
        f.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
          f.extend(d.to_le_bytes());
        }
        f.extend(ty.to_le_bytes());
        f.extend(at.to_le_bytes());
      }
      f.resize(f.len().next_multiple_of(32), 0);
      f.extend(data);
      f
    };
    let open = |bytes: &[u8], tag: &str, q5: fn(&str) -> bool| {
      let dir = std::env::temp_dir();
      let src = dir.join(format!("wavo-q4-src-{tag}-{}", std::process::id()));
      let dst = dir.join(format!("wavo-q4-dst-{tag}-{}", std::process::id()));
      std::fs::write(&src, bytes).unwrap();
      let wrote = write_q4_k(&src, &dst, q5).unwrap();
      let g = Gguf::open(&dst).unwrap();
      assert_eq!(wrote.src_bytes, std::fs::metadata(&src).unwrap().len());
      assert_eq!(wrote.dst_bytes, std::fs::metadata(&dst).unwrap().len());
      assert!(wrote.dst_bytes < wrote.src_bytes);
      std::fs::remove_file(src).unwrap();
      std::fs::remove_file(dst).unwrap();
      (wrote, g)
    };

    let mixed = build(&[
      ("narrow", &[32, 8], 8, &weights),
      ("wide", &[32, 8], 8, &weights),
      ("tail", &[32, 2], 8, &tail),
      ("kept", &[32], 8, &q8),
      ("bias", &[2], 0, &floats.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
    ]);
    let (wrote, g) = open(&mixed, "mix", |name| name == "wide");
    assert_eq!((wrote.q4, wrote.q4_elements), (1, 256));
    assert_eq!((wrote.q5, wrote.q5_elements), (2, 256 + 64));
    assert_eq!(g.get::<u32>("general.file_type").unwrap(), 15);
    assert_eq!(g.tensor("narrow", &[32, 8]).unwrap().ty, Type::Q4_K);
    assert_eq!(g.tensor("narrow", &[32, 8]).unwrap().to_f32().unwrap(), q4);
    assert_eq!(g.tensor("wide", &[32, 8]).unwrap().ty, Type::Q5_0);
    assert_eq!(g.tensor("wide", &[32, 8]).unwrap().to_f32().unwrap(), q5_wide);
    assert_eq!(g.tensor("tail", &[32, 2]).unwrap().ty, Type::Q5_0);
    assert_eq!(g.tensor("tail", &[32, 2]).unwrap().to_f32().unwrap(), q5_tail);
    assert_eq!(g.tensor("kept", &[32]).unwrap().ty, Type::Q8_0);
    assert_eq!(g.tensor("kept", &[32]).unwrap().to_f32().unwrap(), values(32));
    assert_eq!(g.tensor("bias", &[2]).unwrap().to_f32().unwrap(), floats);

    let only = build(&[("narrow", &[32, 8], 8, &weights)]);
    let (wrote, g) = open(&only, "q4", |_| false);
    assert_eq!((wrote.q4, wrote.q5), (1, 0));
    assert_eq!(g.get::<u32>("general.file_type").unwrap(), 14);
    assert_eq!(g.tensor("narrow", &[32, 8]).unwrap().ty, Type::Q4_K);
  }

  #[test]
  fn q4_k_round_trips() {
    let scale = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let min = [0u8, 1, 0, 1, 0, 1, 0, 1];
    let codes = std::array::from_fn(|i| (i % 16) as u8);
    let mut block = [0u8; 144];
    block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
    block[4..16].copy_from_slice(&pack_scales_k4(&scale, &min));
    block[16..].copy_from_slice(&pack_qs_k4(&codes));
    for j in 0..8 {
      assert_eq!(scale_min_k4(block[4..16].try_into().unwrap(), j), (scale[j], min[j]));
    }
    let scales = q4k_scales(&block);
    assert_eq!(&scales[4..12], &scale);
    assert_eq!(&scales[12..20], &min);
    let packed = q4k_packed(&block);
    for group in 0..64 {
      let local = group * 4;
      let word = u32::from_le_bytes(packed[local / 8 * 4..][..4].try_into().unwrap());
      let bits = (word >> ((local % 8) * 4)) & 0xffff;
      let got = [
        (bits & 15) as u8,
        ((bits >> 4) & 15) as u8,
        ((bits >> 8) & 15) as u8,
        ((bits >> 12) & 15) as u8,
      ];
      assert_eq!(got, codes[local..local + 4]);
    }
    let g = Gguf::with_tensors(&[("q", &[256], Type::Q4_K, &block)]);
    let expected: Vec<f32> = (0..256)
      .map(|i| 0.5 * scale[i / 32] as f32 * codes[i] as f32 - 0.25 * min[i / 32] as f32)
      .collect();
    assert_eq!(g.tensor("q", &[256]).unwrap().to_f32().unwrap(), expected);

    let zeros = quantize_q4_k(&[0.0; 256]);
    assert_eq!(zeros, vec![0u8; 144]);
    let g = Gguf::with_tensors(&[("z", &[256], Type::Q4_K, &zeros)]);
    assert_eq!(g.tensor("z", &[256]).unwrap().to_f32().unwrap(), vec![0.0; 256]);
  }

  /// A diagonal inverse leaves every weight where it was, so the bytes match round-to-nearest.
  #[test]
  fn q4_k_gptq_identity_matches_round_to_nearest() {
    let row = 64;
    let w: Vec<f32> = (0..256).map(|i| ((i * 3) % 19) as f32 - 9.0).collect();
    let mut hinv = vec![0.0; row * row];
    for i in 0..row {
      hinv[i * row + i] = 1.0;
    }
    assert_eq!(quantize_q4_k_gptq(&w, row, &hinv), quantize_q4_k(&w));
  }

  /// A non-uniform channel weight changes the packed block. The 6-bit snap can still raise the
  /// error of one column, so this checks the bytes, not a single-column score.
  #[test]
  fn q4_k_weighted_differs_from_round_to_nearest() {
    let row = 32;
    let w: Vec<f32> =
      (0..256).map(|i| if i % row == 0 { 7.0 } else { ((i % 15) as f32 - 7.0) * 0.1 }).collect();
    let mut channel_w = vec![0.01; row];
    channel_w[0] = 100.0;
    let weighted = quantize_q4_k_weighted(&w, row, &channel_w);
    assert_eq!(weighted.len(), 144);
    assert_ne!(weighted, quantize_q4_k(&w));
  }

  #[test]
  fn rejects_truncated_files() {
    let mut f = sample();
    f.truncate(f.len() - 1);
    assert!(parse(f).is_err());
  }
}
