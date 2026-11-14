// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Matrix products of the encoders on the CPU: `c = epilogue(a·Wᵀ + b)` with W kept in its GGUF
//! type. W is unpacked 16 rows at a time into f32 panels `[k][16]`, and a tile of 6 rows × 16
//! columns of fused multiply-adds (AVX2+FMA, NEON, else portable) runs over the rows of `a`. A job
//! unpacks a group of panels that fits the L2 cache once and runs them over a range of rows,
//! 6 rows of `a` at a time while they are in L1; threads take jobs as they come free.
//!
//! With AVX-VNNI, Q8_0 weights meet the rows of `a` rounded per 32 values to i16 instead: products
//! summed exactly in i32, then scaled. On an N100 that is 7–10% faster than f32 FMA; the GigaAM
//! fixtures hold. (i8 rows, twice as fast again, change a comma or a space now and then.)

use half::f16;
use half::slice::HalfFloatSliceExt;

use super::{Q8, par_each};
use crate::error::{Result, bail};
use crate::gguf::{Tensor, Type};
use crate::gpu::Epilogue;

/// Panel width: output columns per tile.
pub const NR: usize = 16;
/// Rows of `a` per tile.
const MR: usize = 6;
/// Bytes of the panels a job unpacks.
const GROUP: usize = 384 << 10;

enum Weights {
  Q8(Q8),
  F16(Vec<f16>),
  F32(Vec<f32>),
}

/// A linear layer `y = x·Wᵀ + b`, W `[n][k]` in its GGUF type (Q8_0, F16 or F32).
pub struct Linear {
  pub n: usize,
  pub k: usize,
  w: Weights,
  /// `n` values, zeros without a bias.
  bias: Vec<f32>,
}

impl Linear {
  /// The weights `ws` (ggml dims `[k, …, rows]`) stacked along the output, and their biases (empty
  /// for none).
  pub fn new(ws: &[Tensor], bias: Vec<f32>) -> Result<Self> {
    let k = ws[0].len() / ws[0].dims.last().unwrap();
    let n = ws.iter().map(|w| w.dims.last().unwrap()).sum::<usize>();
    if ws.iter().any(|w| w.ty != ws[0].ty || w.len() / w.dims.last().unwrap() != k) {
      bail!("stacked weights differ in type or shape");
    }
    let w = match ws[0].ty {
      Type::Q8_0 => Weights::Q8(Q8::new(&ws.iter().collect::<Vec<_>>())?),
      Type::F16 => {
        let mut w = Vec::with_capacity(n * k);
        for t in ws {
          t.read(2 << 20, |_, piece| {
            w.extend(piece.as_chunks::<2>().0.iter().map(|b| f16::from_le_bytes(*b)));
          })?;
        }
        Weights::F16(w)
      }
      _ => Weights::F32(ws.iter().map(|w| w.to_f32()).collect::<Result<Vec<_>>>()?.concat()),
    };
    if !bias.is_empty() && bias.len() != n {
      bail!("linear layer {k}→{n}: bias has {} values", bias.len());
    }
    let bias = if bias.is_empty() { vec![0.0; n] } else { bias };
    Ok(Self { n, k, w, bias })
  }

  /// Panel `p` into `out` `[k][NR]`: rows 16p.. of W, zeros past the last; `wide` is scratch of
  /// `k` floats.
  fn pack(&self, p: usize, out: &mut [f32], wide: &mut [f32]) {
    let k = self.k;
    for j in 0..NR {
      let row = NR * p + j;
      if row >= self.n {
        out.iter_mut().skip(j).step_by(NR).for_each(|v| *v = 0.0);
        continue;
      }
      match &self.w {
        Weights::Q8(q8) => {
          let (q, d) = (&q8.q[row * k..][..k], &q8.d[row * k / 32..][..k / 32]);
          for ((q, d), out) in
            q.as_chunks::<32>().0.iter().zip(d).zip(out.as_chunks_mut::<{ 32 * NR }>().0)
          {
            for (q, out) in q.iter().zip(out.as_chunks_mut::<NR>().0) {
              out[j] = *q as f32 * d;
            }
          }
        }
        Weights::F16(w) => {
          w[row * k..][..k].convert_to_f32_slice(wide);
          for (w, out) in wide.iter().zip(out.as_chunks_mut::<NR>().0) {
            out[j] = *w;
          }
        }
        Weights::F32(w) => {
          for (w, out) in w[row * k..][..k].iter().zip(out.as_chunks_mut::<NR>().0) {
            out[j] = *w;
          }
        }
      }
    }
  }
}

/// `c = epilogue(a·Wᵀ + b)` for `m` rows of `a` `[m][k]` into `c` `[m][n]`.
pub fn gemm(a: &[f32], l: &Linear, c: &mut [f32], m: usize, epilogue: Epilogue) {
  let k = l.k;
  #[cfg(target_arch = "x86_64")]
  if let Weights::Q8(q8) = &l.w
    && super::x86::vnni()
  {
    let (q, d) = (&mut vec![0; m * k], &mut vec![0.0; m * k / 32]);
    let rows = m.div_ceil(4 * super::threads());
    let parts =
      a[..m * k].chunks(rows * k).zip(q.chunks_mut(rows * k)).zip(d.chunks_mut(rows * k / 32));
    // SAFETY: the CPU has AVX-VNNI.
    par_each(parts, || (), |_, ((a, q), d)| unsafe { x86::quantize(a, q, d) });
    let (aq, ad) = (&*q, &*d);
    let (kq, kd) = (k * NR, k / 32 * NR);
    return jobs(
      l,
      c,
      m,
      epilogue,
      2 * kq + 4 * kd,
      |panels| (vec![0; panels * kq], vec![0.0; panels * kd]),
      |(q, d), i, p| x86::pack(q8, l.n, p, &mut q[i * kq..][..kq], &mut d[i * kd..][..kd]),
      |(q, d), i, row, last| {
        let at = |r: usize| (row + r).min(last - 1);
        let a = std::array::from_fn(|r| &aq[at(r) * k..][..k]);
        let ad = std::array::from_fn(|r| &ad[at(r) * k / 32..][..k / 32]);
        // SAFETY: the CPU has AVX-VNNI.
        unsafe { x86::tile_i16(a, ad, &q[i * kq..][..kq], &d[i * kd..][..kd], k) }
      },
    );
  }
  let kb = k * NR;
  jobs(
    l,
    c,
    m,
    epilogue,
    4 * kb,
    |panels| (vec![0.0; panels * kb], vec![0.0; k]),
    |(b, wide), i, p| l.pack(p, &mut b[i * kb..][..kb], wide),
    |(b, _), i, row, last| {
      tile(std::array::from_fn(|r| &a[(row + r).min(last - 1) * k..][..k]), &b[i * kb..][..kb], k)
    },
  );
}

/// Runs a GEMM's jobs over `l`'s panels of `bytes` each: a job packs a group of panels that fits
/// the L2 cache into its `scratch` (made for so many panels) with `pack(scratch, index in the
/// group, panel)`, then takes `tile(scratch, index in the group, first row, end of its rows)` over
/// a range of rows, `MR` rows at a time.
#[allow(clippy::too_many_arguments)]
fn jobs<S>(
  l: &Linear,
  c: &mut [f32],
  m: usize,
  epilogue: Epilogue,
  bytes: usize,
  scratch: impl Fn(usize) -> S + Sync,
  pack: impl Fn(&mut S, usize, usize) + Sync,
  tile: impl Fn(&S, usize, usize, usize) -> [[f32; NR]; MR] + Sync,
) {
  let n = l.n;
  let panels = n.div_ceil(NR);
  let per_group = (GROUP / bytes).clamp(1, panels);
  let groups = panels.div_ceil(per_group);
  // Enough jobs to keep the threads busy to the end, each over at least 8 tiles of rows.
  let splits = (4 * super::threads()).div_ceil(groups).min(m.div_ceil(8 * MR)).max(1);
  let rows = m.div_ceil(splits).next_multiple_of(MR);
  let c = Out(c.as_mut_ptr());
  par_each(
    0..groups * splits,
    || scratch(per_group),
    |buf, job| {
      let (g, s) = (job / splits, job % splits);
      let ps = g * per_group..((g + 1) * per_group).min(panels);
      for (i, p) in ps.clone().enumerate() {
        pack(buf, i, p);
      }
      let (r0, r1) = (s * rows, ((s + 1) * rows).min(m));
      for row in (r0..r1).step_by(MR) {
        for (i, p) in ps.clone().enumerate() {
          let acc = tile(buf, i, row, r1);
          let cols = NR * p..(NR * p + NR).min(n);
          // SAFETY: jobs write disjoint tiles of `c`, rows r0..r1 of their panels' columns.
          unsafe {
            c.store(&acc, row..(row + MR).min(r1), cols.clone(), n, &l.bias[cols], epilogue)
          };
        }
      }
    },
  );
}

/// The panels `[k][NR]` of `b`, a matrix `[n][k]` (row stride `ld`) or with `transposed`
/// `[k][n]`, for `product`: zeros past n, and past the rows of `b` up to `k`.
pub fn panels(b: &[f32], ld: usize, n: usize, k: usize, rows: usize, transposed: bool) -> Vec<f32> {
  let mut out = vec![0.0; n.div_ceil(NR) * k * NR];
  for (p, out) in out.chunks_exact_mut(k * NR).enumerate() {
    for j in 0..NR.min(n - NR * p) {
      for (kk, out) in out.as_chunks_mut::<NR>().0.iter_mut().take(rows).enumerate() {
        let col = NR * p + j;
        out[j] = if transposed { b[kk * ld + col] } else { b[col * ld + kk] };
      }
    }
  }
  out
}

/// `c = a·b` for `m` rows of `a` (row stride `lda`, `k` long) and `b` as `panels`, on this
/// thread: all the panels' columns of `c` (row stride `ldc`, at least as wide).
pub fn product(a: &[f32], lda: usize, m: usize, b: &[f32], k: usize, c: &mut [f32], ldc: usize) {
  assert!(b.len() / k <= ldc && c.len() >= (m - 1) * ldc + b.len() / k);
  let out = Out(c.as_mut_ptr());
  for i in (0..m).step_by(MR) {
    let a = std::array::from_fn(|r| &a[(i + r).min(m - 1) * lda..][..k]);
    for (p, b) in b.chunks_exact(k * NR).enumerate() {
      let acc = tile(a, b, k);
      // SAFETY: rows below m of `c`, columns below the panels' width ≤ ldc (asserted).
      unsafe {
        out.store(&acc, i..(i + MR).min(m), NR * p..NR * p + NR, ldc, &ZEROS, Epilogue::Bias)
      };
    }
  }
}

const ZEROS: [f32; NR] = [0.0; NR];

/// `c` for threads writing disjoint tiles.
#[derive(Clone, Copy)]
struct Out(*mut f32);

// SAFETY: the writers' tiles don't overlap.
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
  /// `epilogue(acc + bias)` into the rows `rows` and columns `cols` of `c` (row stride `ld`),
  /// `bias` from the first of `cols`.
  ///
  /// # Safety
  ///
  /// The tile is inside `c` and no other thread touches it.
  unsafe fn store(
    self,
    acc: &[[f32; NR]; MR],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    ld: usize,
    bias: &[f32],
    epilogue: Epilogue,
  ) {
    for (r, acc) in rows.zip(acc) {
      // SAFETY: as the caller promises.
      let c =
        unsafe { std::slice::from_raw_parts_mut(self.0.add(r * ld + cols.start), cols.len()) };
      let (acc, bias) = (&acc[..cols.len()], &bias[..cols.len()]);
      match epilogue {
        Epilogue::Bias => c.iter_mut().zip(acc).zip(bias).for_each(|((c, a), b)| *c = a + b),
        Epilogue::Silu => {
          c.iter_mut().zip(acc).zip(bias).for_each(|((c, a), b)| *c = super::ops::silu(a + b))
        }
        Epilogue::Relu => {
          c.iter_mut().zip(acc).zip(bias).for_each(|((c, a), b)| *c = (a + b).max(0.0))
        }
        Epilogue::Residual(alpha) => {
          c.iter_mut().zip(acc).zip(bias).for_each(|((c, a), b)| *c += alpha * (a + b))
        }
        Epilogue::Gelu | Epilogue::GeluResidual | Epilogue::Tanh => {
          unreachable!("no CPU encoder uses GELU or tanh")
        }
      }
    }
  }
}

/// `MR` rows of `a` times a panel `b` `[k][NR]`.
fn tile(a: [&[f32]; MR], b: &[f32], k: usize) -> [[f32; NR]; MR] {
  #[cfg(target_arch = "x86_64")]
  if super::x86::avx2() {
    // SAFETY: the CPU has AVX2 and FMA.
    return unsafe { x86::tile(a, b, k) };
  }
  #[cfg(target_arch = "aarch64")]
  return neon::tile(a, b, k);
  #[allow(unreachable_code)]
  {
    let mut acc = [[0.0; NR]; MR];
    for (kk, b) in b.as_chunks::<NR>().0.iter().take(k).enumerate() {
      for (acc, a) in acc.iter_mut().zip(a) {
        for (c, b) in acc.iter_mut().zip(b) {
          *c += a[kk] * b;
        }
      }
    }
    acc
  }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
  use std::arch::x86_64::*;

  use super::{MR, NR, Q8};

  /// Rows `x` per 32 values as i16 `q` and a scale `d`: q = round(x / d), d = max|x| / 32767.
  #[target_feature(enable = "avx2,fma")]
  pub fn quantize(x: &[f32], q: &mut [i16], d: &mut [f32]) {
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    for ((x, q), d) in x.as_chunks::<32>().0.iter().zip(q.as_chunks_mut::<32>().0).zip(d) {
      // SAFETY: four loads of 8 of the 32 floats; two stores of 16 of the 32 i16.
      unsafe {
        let v: [__m256; 4] = std::array::from_fn(|i| _mm256_loadu_ps(x.as_ptr().add(8 * i)));
        let m = v.iter().fold(_mm256_setzero_ps(), |m, v| _mm256_max_ps(m, _mm256_and_ps(*v, abs)));
        let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let max = _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps::<1>(m, m)));
        *d = max / 32767.0;
        let id = _mm256_set1_ps(if max > 0.0 { 32767.0 / max } else { 0.0 });
        let i = v.map(|v| _mm256_cvtps_epi32(_mm256_mul_ps(v, id)));
        // packs interleaves 128-bit lanes: 0..4 8..12 4..8 12..16, back in order by permute.
        let lo = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(i[0], i[1]));
        let hi = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(i[2], i[3]));
        _mm256_storeu_si256(q.as_mut_ptr().cast(), lo);
        _mm256_storeu_si256(q.as_mut_ptr().add(16).cast(), hi);
      }
    }
  }

  /// Panel `p` of `w` (`n` rows) for `tile_i16`: values `[k / 2][NR][2]` (two of k per lane) and
  /// scales `[k / 32][NR]`, zeros past the last row.
  pub fn pack(w: &Q8, n: usize, p: usize, q: &mut [i16], d: &mut [f32]) {
    let k = w.k;
    for j in 0..NR {
      let row = NR * p + j;
      let lanes = q.as_chunks_mut::<{ 2 * NR }>().0.iter_mut().map(|q| &mut q[2 * j..][..2]);
      let scales = d.iter_mut().skip(j).step_by(NR);
      if row >= n {
        lanes.for_each(|l| l.fill(0));
        scales.for_each(|d| *d = 0.0);
        continue;
      }
      for (l, w) in lanes.zip(w.q[row * k..][..k].as_chunks::<2>().0) {
        l.copy_from_slice(&[w[0] as i16, w[1] as i16]);
      }
      for (d, w) in scales.zip(&w.d[row * k / 32..][..k / 32]) {
        *d = *w;
      }
    }
  }

  /// `MR` rows of `quantize`d `a` with their scales `ad` times a `pack`ed panel: per 32 of k, i16
  /// products summed exactly in i32 (AVX-VNNI), then scaled into f32.
  #[target_feature(enable = "avx2,fma,avxvnni")]
  pub fn tile_i16(
    a: [&[i16]; MR],
    ad: [&[f32]; MR],
    q: &[i16],
    d: &[f32],
    k: usize,
  ) -> [[f32; NR]; MR] {
    let (q, d) = (&q[..k * NR], &d[..k / 32 * NR]);
    let mut acc = [[_mm256_setzero_ps(); 2]; MR];
    for blk in 0..k / 32 {
      let mut sum = [[_mm256_setzero_si256(); 2]; MR];
      for pair in 16 * blk..16 * blk + 16 {
        // SAFETY: 32 i16 of the panel's row pair, and two i16 of each row of `a`.
        unsafe {
          let b0 = _mm256_loadu_si256(q.as_ptr().add(pair * 2 * NR).cast());
          let b1 = _mm256_loadu_si256(q.as_ptr().add(pair * 2 * NR + NR).cast());
          for (s, a) in sum.iter_mut().zip(a) {
            let v = _mm256_set1_epi32(a.as_ptr().add(2 * pair).cast::<i32>().read_unaligned());
            s[0] = _mm256_dpwssd_avx_epi32(s[0], v, b0);
            s[1] = _mm256_dpwssd_avx_epi32(s[1], v, b1);
          }
        }
      }
      // SAFETY: the block's 16 scales.
      let (d0, d1) = unsafe {
        (_mm256_loadu_ps(d.as_ptr().add(blk * NR)), _mm256_loadu_ps(d.as_ptr().add(blk * NR + 8)))
      };
      for ((acc, s), ad) in acc.iter_mut().zip(sum).zip(ad) {
        let da = _mm256_set1_ps(ad[blk]);
        acc[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[0]), _mm256_mul_ps(da, d0), acc[0]);
        acc[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[1]), _mm256_mul_ps(da, d1), acc[1]);
      }
    }
    let mut out = [[0.0; NR]; MR];
    for (out, acc) in out.iter_mut().zip(acc) {
      // SAFETY: 16 floats.
      unsafe {
        _mm256_storeu_ps(out.as_mut_ptr(), acc[0]);
        _mm256_storeu_ps(out.as_mut_ptr().add(8), acc[1]);
      }
    }
    out
  }

  #[target_feature(enable = "avx2,fma")]
  pub fn tile(a: [&[f32]; MR], b: &[f32], k: usize) -> [[f32; NR]; MR] {
    let (a, b) = (a.map(|a| a[..k].as_ptr()), &b[..k * NR]);
    let mut acc = [[_mm256_setzero_ps(); 2]; MR];
    for kk in 0..k {
      // SAFETY: 16 floats of row kk of the panel, and element kk of each row of `a`.
      unsafe {
        let b0 = _mm256_loadu_ps(b.as_ptr().add(kk * NR));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(kk * NR + 8));
        for (acc, a) in acc.iter_mut().zip(a) {
          let v = _mm256_broadcast_ss(&*a.add(kk));
          acc[0] = _mm256_fmadd_ps(v, b0, acc[0]);
          acc[1] = _mm256_fmadd_ps(v, b1, acc[1]);
        }
      }
    }
    let mut out = [[0.0; NR]; MR];
    for (out, acc) in out.iter_mut().zip(acc) {
      // SAFETY: 16 floats.
      unsafe {
        _mm256_storeu_ps(out.as_mut_ptr(), acc[0]);
        _mm256_storeu_ps(out.as_mut_ptr().add(8), acc[1]);
      }
    }
    out
  }
}

#[cfg(target_arch = "aarch64")]
mod neon {
  use std::arch::aarch64::*;

  use super::{MR, NR};

  pub fn tile(a: [&[f32]; MR], b: &[f32], k: usize) -> [[f32; NR]; MR] {
    let (a, b) = (a.map(|a| a[..k].as_ptr()), &b[..k * NR]);
    // SAFETY: NEON is part of aarch64; the loads read 16 floats of row kk of the panel and element
    // kk of each row of `a`.
    unsafe {
      let mut acc = [[vdupq_n_f32(0.0); 4]; MR];
      for kk in 0..k {
        let p = b.as_ptr().add(kk * NR);
        let bv = [vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))];
        for (acc, a) in acc.iter_mut().zip(a) {
          let v = vld1q_dup_f32(a.add(kk));
          for (c, b) in acc.iter_mut().zip(bv) {
            *c = vfmaq_f32(*c, b, v);
          }
        }
      }
      let mut out = [[0.0; NR]; MR];
      for (out, acc) in out.iter_mut().zip(acc) {
        for (i, v) in acc.into_iter().enumerate() {
          vst1q_f32(out.as_mut_ptr().add(4 * i), v);
        }
      }
      out
    }
  }
}

#[cfg(test)]
mod tests {
  use half::f16;

  use crate::gguf::{Gguf, Type};
  use crate::gpu::Epilogue;

  fn naive(a: &[f32], w: &[f32], k: usize) -> Vec<f32> {
    a.chunks_exact(k).flat_map(|a| w.chunks_exact(k).map(|w| crate::cpu::dot(a, w))).collect()
  }

  #[test]
  fn gemm_matches_dot() {
    // 40 rows of 64: two and a half panels; 13 rows of a: two tiles and a short one.
    let (k, n, m) = (64, 40, 13);
    let v: Vec<f32> = (0..n * k).map(|i| (i * 7919 % 31) as f32 / 8.0 - 15.0 / 8.0).collect();
    let mut q8 = Vec::new();
    for b in v.as_chunks::<32>().0 {
      q8.extend(f16::from_f32(0.125).to_le_bytes());
      q8.extend(b.map(|v| (v * 8.0) as i8 as u8));
    }
    let f32s: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16s: Vec<u8> = v.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
    let a: Vec<f32> = (0..m * k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32 / 4.0).collect();
    let want = naive(&a, &v, k);
    for (ty, bytes) in [(Type::F32, f32s), (Type::F16, f16s), (Type::Q8_0, q8)] {
      let g = Gguf::with_tensors(&[("w", &[k, n], ty, &bytes)]);
      let l = super::Linear::new(&[g.tensor("w", &[k, n]).unwrap()], bias.clone()).unwrap();
      let mut c = vec![1.0; m * n];
      super::gemm(&a, &l, &mut c, m, Epilogue::Residual(0.5));
      for (i, (c, w)) in c.iter().zip(&want).enumerate() {
        let want = 1.0 + 0.5 * (w + bias[i % n]);
        assert!((c - want).abs() <= 1e-4 * want.abs().max(1.0), "{ty:?} {i}: {c} vs {want}");
      }
    }
  }

  #[test]
  fn product_of_panels() {
    // b [k][n] with n = 20 (a panel and a part), 7 of its k = 9 rows given.
    let (m, k, n, rows) = (7, 9, 20, 7);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..rows * n).map(|i| (i % 7) as f32 / 2.0).collect();
    let panels = super::panels(&b, n, n, k, rows, true);
    let mut c = vec![0.0; m * 32];
    super::product(&a, k, m, &panels, k, &mut c, 32);
    for i in 0..m {
      for j in 0..n {
        let want: f32 = (0..rows).map(|kk| a[i * k + kk] * b[kk * n + j]).sum();
        assert_eq!(c[i * 32 + j], want, "{i} {j}");
      }
    }
  }
}
