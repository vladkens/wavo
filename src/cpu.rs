// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! CPU dot products for the frontends and decoders, with NEON tiles on aarch64 and AVX2 on x86_64.

use std::sync::{Mutex, PoisonError};

use half::f16;

use crate::error::Result;
use crate::gguf::Tensor;

/// Dot product with 16 independent accumulators (four FMA vectors in flight), summed pairwise so
/// the final reduction also vectorizes.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
  let ((a16, a_tail), (b16, b_tail)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
  let mut acc = [0f32; 16];
  for (x, y) in a16.iter().zip(b16) {
    for ((s, x), y) in acc.iter_mut().zip(x).zip(y) {
      *s = x.mul_add(*y, *s);
    }
  }
  for n in [8, 4, 2, 1] {
    let (lo, hi) = acc.split_at_mut(n);
    for (l, h) in lo.iter_mut().zip(&*hi) {
      *l += h;
    }
  }
  acc[0] + a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum::<f32>()
}

/// `w` (`k`-long rows) padded with zero rows to a multiple of 4, as `matmul` takes it.
pub fn rows4(mut w: Vec<f32>, k: usize) -> Vec<f32> {
  w.resize((w.len() / k).next_multiple_of(4) * k, 0.0);
  w
}

/// `[frame][row]` dots of the `k`-long rows of `w` (a multiple of 4 rows) with the `k`-long frames
/// of `z`, each equal to `dot(row, frame)`. Frames go in pairs, two rows at a time; a last single
/// frame goes four rows at a time. Either way each loaded chunk serves several dots.
pub fn matmul(w: &[f32], z: &[f32], k: usize) -> Vec<f32> {
  let rows = w.len() / k;
  let mut out = vec![0.0; z.len() / k * rows];
  let pairs = z.chunks_exact(2 * k);
  let last = pairs.remainder();
  for (z, out) in pairs.zip(out.chunks_exact_mut(2 * rows)) {
    for (r, w) in w.chunks_exact(2 * k).enumerate() {
      let [d0, d1] = tile::<2, 2>(w, z, k);
      out[2 * r..][..2].copy_from_slice(&d0);
      out[rows + 2 * r..][..2].copy_from_slice(&d1);
    }
  }
  if !last.is_empty() {
    let out = out.rchunks_exact_mut(rows).next().unwrap();
    for (w, out) in w.chunks_exact(4 * k).zip(out.as_chunks_mut::<4>().0) {
      [*out] = tile::<4, 1>(w, last, k);
    }
  }
  out
}

/// `dot` of `R` rows of `w` with `F` frames of `z` (`[frame][row]`), loading each chunk once. It
/// keeps `dot`'s lanes, fused multiply-adds and pairwise sum, so the results are identical.
#[cfg(target_arch = "aarch64")]
fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
  use std::arch::aarch64::*;
  let (w, z, n) = (&w[..R * k], &z[..F * k], k / 16 * 16);
  // SAFETY: NEON is part of aarch64, and every load reads 4 floats below `n` ≤ k of a row of `w`
  // or a frame of `z`.
  let acc = unsafe {
    let mut acc = [[[vdupq_n_f32(0.0); 4]; R]; F];
    for i in (0..n).step_by(16) {
      for c in 0..4 {
        let wv: [_; R] = std::array::from_fn(|r| vld1q_f32(w.as_ptr().add(r * k + i + 4 * c)));
        for (f, acc) in acc.iter_mut().enumerate() {
          let zv = vld1q_f32(z.as_ptr().add(f * k + i + 4 * c));
          for (a, wv) in acc.iter_mut().zip(wv) {
            a[c] = vfmaq_f32(a[c], wv, zv);
          }
        }
      }
    }
    acc.map(|acc| {
      acc.map(|[a0, a1, a2, a3]| {
        let lanes = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
        let pair = vadd_f32(vget_low_f32(lanes), vget_high_f32(lanes));
        vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair)
      })
    })
  };
  std::array::from_fn(|f| {
    std::array::from_fn(|r| {
      let (w, z) = (&w[r * k + n..][..k - n], &z[f * k + n..][..k - n]);
      acc[f][r] + w.iter().zip(z).map(|(x, y)| x * y).sum::<f32>()
    })
  })
}

#[cfg(not(target_arch = "aarch64"))]
fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
  #[cfg(target_arch = "x86_64")]
  if x86::avx2() {
    // SAFETY: the CPU has AVX2 and FMA.
    return unsafe { x86::tile(w, z, k) };
  }
  std::array::from_fn(|f| std::array::from_fn(|r| dot(&w[r * k..][..k], &z[f * k..][..k])))
}

/// Q8_0 weights `[rows][k]` (k a multiple of 32) as int8 values and one f32 scale per 32. A
/// weight is `q · d`, exact in f32, so every product below equals `dot` on the dequantized row.
pub struct Q8 {
  k: usize,
  q: Vec<i8>,
  d: Vec<f32>,
}

impl Q8 {
  pub fn new(t: &Tensor) -> Result<Self> {
    let mut b = Vec::new();
    t.read(1 << 20, |_, piece| b.extend_from_slice(piece))?;
    let blocks = b.as_chunks::<34>().0;
    let d = blocks.iter().map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
    let q = blocks.iter().flat_map(|b| b[2..].iter().map(|&v| v as i8)).collect();
    Ok(Self { k: t.dims[0], q, d })
  }

  pub fn row(&self, r: usize) -> Vec<f32> {
    let (q, d) = (&self.q[r * self.k..][..self.k], &self.d[r * self.k / 32..]);
    q.iter().enumerate().map(|(i, &q)| q as f32 * d[i / 32]).collect()
  }

  /// Rows from `r0` times `x` into `out`, four rows per pass over x.
  #[cfg(target_arch = "aarch64")]
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let k = self.k;
    let (q, d, x) = (&self.q[r0 * k..][..out.len() * k], &self.d[r0 * k / 32..], &x[..k]);
    // SAFETY: NEON is part of aarch64; the loads read 16 int8 of a row of `q` and 4 floats of `x`
    // below k, and a short last group repeats its last row, so every row is inside `q`.
    unsafe {
      for (g, out) in out.chunks_mut(4).enumerate() {
        let row = |j: usize| 4 * g + j.min(out.len() - 1);
        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
        for i in (0..k).step_by(16) {
          let xs = [0, 1, 2, 3].map(|c| vld1q_f32(x.as_ptr().add(i + 4 * c)));
          for (j, acc) in acc.iter_mut().enumerate() {
            let at = row(j) * k + i;
            let v = vld1q_s8(q.as_ptr().add(at));
            let (lo, hi) = (vmovl_s8(vget_low_s8(v)), vmovl_high_s8(v));
            let w = [vget_low_s16(lo), vget_high_s16(lo), vget_low_s16(hi), vget_high_s16(hi)];
            for (a, (w, x)) in acc.iter_mut().zip(w.into_iter().zip(xs)) {
              let w = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(w)), d[at / 32]);
              *a = vfmaq_f32(*a, w, x);
            }
          }
        }
        for (o, [a0, a1, a2, a3]) in out.iter_mut().zip(acc) {
          let lanes = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
          let pair = vadd_f32(vget_low_f32(lanes), vget_high_f32(lanes));
          *o = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);
        }
      }
    }
  }

  #[cfg(not(target_arch = "aarch64"))]
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
      // SAFETY: the CPU has AVX2 and FMA.
      return unsafe { x86::rows(self, r0, x, out) };
    }
    for (r, o) in out.iter_mut().enumerate() {
      *o = dot(&self.row(r0 + r), x);
    }
  }
}

/// `tile` and `Q8::rows` with AVX2 and FMA, chosen at run time. Like the NEON ones they keep
/// `dot`'s 16 lanes (two 8-float vectors), fused multiply-adds and pairwise sum, so the results
/// are identical. No intrinsics in closures: those would not get the target features.
#[cfg(target_arch = "x86_64")]
mod x86 {
  use std::arch::x86_64::*;

  use super::Q8;

  pub fn avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
  }

  /// `dot`'s pairwise sum of 16 lanes, 0..8 in `a` and 8..16 in `b`.
  #[target_feature(enable = "avx2,fma")]
  fn sum(a: __m256, b: __m256) -> f32 {
    let s = _mm256_add_ps(a, b);
    let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(s) + _mm_cvtss_f32(_mm_shuffle_ps::<1>(s, s))
  }

  #[target_feature(enable = "avx2,fma")]
  pub fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
    let (w, z, n) = (&w[..R * k], &z[..F * k], k / 16 * 16);
    let mut acc = [[[_mm256_setzero_ps(); 2]; R]; F];
    for i in (0..n).step_by(16) {
      for c in 0..2 {
        let mut wv = [_mm256_setzero_ps(); R];
        for (r, v) in wv.iter_mut().enumerate() {
          // SAFETY: 8 floats below `n` ≤ k of a row of `w`.
          *v = unsafe { _mm256_loadu_ps(w.as_ptr().add(r * k + i + 8 * c)) };
        }
        for (f, acc) in acc.iter_mut().enumerate() {
          // SAFETY: 8 floats below `n` ≤ k of a frame of `z`.
          let zv = unsafe { _mm256_loadu_ps(z.as_ptr().add(f * k + i + 8 * c)) };
          for (a, wv) in acc.iter_mut().zip(wv) {
            a[c] = _mm256_fmadd_ps(wv, zv, a[c]);
          }
        }
      }
    }
    let mut out = [[0.0; R]; F];
    for (f, out) in out.iter_mut().enumerate() {
      for (r, out) in out.iter_mut().enumerate() {
        let (w, z) = (&w[r * k + n..][..k - n], &z[f * k + n..][..k - n]);
        let tail = w.iter().zip(z).map(|(x, y)| x * y).sum::<f32>();
        *out = sum(acc[f][r][0], acc[f][r][1]) + tail;
      }
    }
    out
  }

  #[target_feature(enable = "avx2,fma")]
  pub fn rows(q8: &Q8, r0: usize, x: &[f32], out: &mut [f32]) {
    let k = q8.k;
    let (q, d, x) = (&q8.q[r0 * k..][..out.len() * k], &q8.d[r0 * k / 32..], &x[..k]);
    for (g, out) in out.chunks_mut(4).enumerate() {
      let mut acc = [[_mm256_setzero_ps(); 2]; 4];
      for i in (0..k).step_by(16) {
        // SAFETY: 16 floats of `x` below k.
        let xs =
          unsafe { [_mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i + 8))] };
        for (j, acc) in acc.iter_mut().enumerate() {
          // A short last group repeats its last row.
          let at = (4 * g + j.min(out.len() - 1)) * k + i;
          // SAFETY: 16 int8 of a row of `q` below k.
          let v = unsafe { _mm_loadu_si128(q.as_ptr().add(at).cast()) };
          let scale = _mm256_set1_ps(d[at / 32]);
          let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
          let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(v)));
          acc[0] = _mm256_fmadd_ps(_mm256_mul_ps(lo, scale), xs[0], acc[0]);
          acc[1] = _mm256_fmadd_ps(_mm256_mul_ps(hi, scale), xs[1], acc[1]);
        }
      }
      for (o, [a, b]) in out.iter_mut().zip(acc) {
        *o = sum(a, b);
      }
    }
  }
}

/// `w · x` for each job `(w, x)`. Blocks of 64 rows go to four scoped threads (an M1/M2 has four
/// performance cores) as they come free, so a thread on an efficiency core takes fewer.
pub fn matvecs<const N: usize>(jobs: [(&Q8, &[f32]); N]) -> [Vec<f32>; N] {
  let mut out = jobs.map(|(w, _)| vec![0.0; w.q.len() / w.k]);
  let blocks = out.iter_mut().zip(jobs).flat_map(|(out, (w, x))| {
    out.chunks_mut(64).enumerate().map(move |(i, out)| (w, x, 64 * i, out))
  });
  let blocks = Mutex::new(blocks);
  let next = || blocks.lock().unwrap_or_else(PoisonError::into_inner).next();
  let work = || {
    while let Some((w, x, r0, out)) = next() {
      w.rows(r0, x, out);
    }
  };
  std::thread::scope(|s| {
    for _ in 1..4 {
      s.spawn(work);
    }
    work();
  });
  out
}

#[cfg(test)]
mod tests {
  #[test]
  fn matmul_matches_dot() {
    let (k, rows, frames) = (333, 8, 5);
    let v: Vec<f32> =
      (0..(rows + frames) * k).map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0).collect();
    let (z, w) = v.split_at(frames * k);
    let want: Vec<u32> = (z.chunks_exact(k))
      .flat_map(|z| w.chunks_exact(k).map(|r| super::dot(r, z).to_bits()))
      .collect();
    let got: Vec<u32> = super::matmul(w, z, k).into_iter().map(f32::to_bits).collect();
    assert_eq!(got, want);
  }

  #[test]
  fn matvecs_match_dot() {
    // 70 rows: a full 64-row block and a short one whose last 4-row group has 2 rows.
    let (k, rows) = (96, 70);
    let q = (0..rows * k).map(|i| (i * 7919 % 255) as u8 as i8).collect();
    let d = (0..rows * k / 32).map(|i| (i % 7 + 1) as f32 * 1e-3).collect();
    let w = super::Q8 { k, q, d };
    let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    let want: Vec<u32> = (0..rows).map(|r| super::dot(&w.row(r), &x).to_bits()).collect();
    let [got, again] = super::matvecs([(&w, &x), (&w, &x)]);
    assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want);
    assert_eq!(again.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want);
  }
}
