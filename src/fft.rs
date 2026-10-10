// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! FFT of real input for the frontends' sizes (320, 400 and 512: factors 2, 3 and 5), adapted from
//! `mixed_radix_fft_f32` in transcribe.cpp's `src/transcribe-mel.cpp` (after whisper.cpp's `fft`):
//! radix-2 decimation in time down to naive DFTs of the odd factor (5, 25 or 1), with cos and sin
//! of 2πi/n from one table. Every output gets the reference's operations in its order, but the
//! data sits transposed (Stockham order), so every loop runs over long contiguous rows and
//! vectorizes. The reference builds with FP contraction on arm64, so every multiply-add is fused;
//! on x86_64 the transform gets the FMA instructions when the CPU has them.

use std::ops::Neg;

pub trait Float: Copy + Default + Neg<Output = Self> {
  fn from_f64(v: f64) -> Self;
  fn mul_add(self, a: Self, b: Self) -> Self;
}

impl Float for f32 {
  fn from_f64(v: f64) -> Self {
    v as f32
  }
  #[inline(always)]
  fn mul_add(self, a: Self, b: Self) -> Self {
    f32::mul_add(self, a, b)
  }
}

impl Float for f64 {
  fn from_f64(v: f64) -> Self {
    v
  }
  #[inline(always)]
  fn mul_add(self, a: Self, b: Self) -> Self {
    f64::mul_add(self, a, b)
  }
}

/// With G blocks of L bins (G·L = n), bin k of block g sits at k·G + g. The leaves are the
/// reference's odd-length DFTs; a stage merges blocks 2b and 2b + 1 (the even and odd halves in
/// the reference's recursion) into block b, so output i < n / 2 reads inputs 2i and 2i + 1 and
/// writes i and i + n / 2. After the last stage G is 1: natural order.
pub struct Fft<T> {
  leaf: usize,
  /// `x[perm[j · G + g]]` is sample j of leaf g.
  perm: Vec<usize>,
  /// The leaf DFT's cos and sin, `[j][k]` for sample j and bin k.
  leaf_trig: (Vec<T>, Vec<T>),
  /// Per stage, the twiddle of each output i < n / 2.
  twiddles: (Vec<T>, Vec<T>),
}

impl<T: Float> Fft<T> {
  pub fn new(n: usize) -> Self {
    let leaf = n >> n.trailing_zeros();
    let blocks = n / leaf;
    // The first sample of each leaf: the recursion puts even samples before odd ones.
    let mut starts = vec![0];
    while starts.len() < blocks {
      starts = starts.iter().map(|s| 2 * s).chain(starts.iter().map(|s| 2 * s + 1)).collect();
    }
    let perm = (0..n).map(|i| starts[i % blocks] + i / blocks * blocks).collect();
    let trig = |i: usize| {
      let phase = std::f64::consts::TAU * i as f64 / n as f64;
      (T::from_f64(phase.cos()), T::from_f64(phase.sin()))
    };
    let leaf_trig = (0..leaf * leaf).map(|i| trig(i / leaf * (i % leaf) * blocks % n)).unzip();
    // In a stage to blocks of `size`, bin k of block b (output i = k · G + b) has twiddle k · G.
    let sizes = std::iter::successors(Some(2 * leaf), |s| Some(2 * s)).take_while(|&s| s <= n);
    let twiddles =
      sizes.flat_map(|size| (0..n / 2).map(move |i| i / (n / size) * (n / size))).map(trig).unzip();
    Self { leaf, perm, leaf_trig, twiddles }
  }

  /// The DFT of the n real values `x`: re and im, the first 2n values of `buf` (4n long).
  pub fn forward<'a>(&self, x: &[T], buf: &'a mut [T]) -> (&'a [T], &'a [T]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") {
      // SAFETY: the CPU has FMA.
      unsafe { self.forward_fma(x, buf) };
      return buf[..2 * self.perm.len()].split_at(self.perm.len());
    }
    self.run(x, buf);
    buf[..2 * self.perm.len()].split_at(self.perm.len())
  }

  #[cfg(target_arch = "x86_64")]
  #[target_feature(enable = "fma")]
  fn forward_fma(&self, x: &[T], buf: &mut [T]) {
    self.run(x, buf);
  }

  /// No closures inside: those would not get `forward_fma`'s target feature.
  #[inline(always)]
  fn run(&self, x: &[T], buf: &mut [T]) {
    let n = self.perm.len();
    let stages = self.twiddles.0.len() / (n / 2);
    let (front, back) = buf[..4 * n].split_at_mut(2 * n);
    // The stages alternate between the halves of `buf`, so the leaves go where the last one ends
    // in front.
    let (mut src, mut dst) = if stages.is_multiple_of(2) { (back, front) } else { (front, back) };
    for (v, &i) in src[..n].iter_mut().zip(&self.perm) {
      *v = x[i];
    }
    let (re, im) = dst.split_at_mut(n);
    leaves(&src[..n], re, im, &self.leaf_trig.0, &self.leaf_trig.1, self.leaf);
    let twiddles = self.twiddles.0.chunks_exact(n / 2).zip(self.twiddles.1.chunks_exact(n / 2));
    for (cos, sin) in twiddles {
      std::mem::swap(&mut src, &mut dst);
      let ((sr, si), (dr, di)) = (src.split_at(n), dst.split_at_mut(n));
      stage(sr, si, dr, di, cos, sin);
    }
  }
}

/// The naive DFTs of the leaves, from the samples `xs` (`[j][g]`) into `re` and `im` (`[k][g]`).
#[inline(always)]
fn leaves<T: Float>(xs: &[T], re: &mut [T], im: &mut [T], cos: &[T], sin: &[T], leaf: usize) {
  let blocks = xs.len() / leaf;
  for (k, (re, im)) in re.chunks_exact_mut(blocks).zip(im.chunks_exact_mut(blocks)).enumerate() {
    // Sample 0 has cos 1 and sin 0: the reference's first multiply-adds give it and 0.
    re.copy_from_slice(&xs[..blocks]);
    im.fill(T::default());
    for j in 1..leaf {
      let (c, s, xs) = (cos[j * leaf + k], sin[j * leaf + k], &xs[j * blocks..][..blocks]);
      for ((re, im), &v) in re.iter_mut().zip(im.iter_mut()).zip(xs) {
        *re = v.mul_add(c, *re);
        *im = (-v).mul_add(s, *im);
      }
    }
  }
}

/// One radix-2 stage: the reference's butterfly on the even (2i) and odd (2i + 1) inputs.
#[inline(always)]
fn stage<T: Float>(sr: &[T], si: &[T], dr: &mut [T], di: &mut [T], cos: &[T], sin: &[T]) {
  let half = cos.len();
  let (sr, si, sin) = (&sr[..2 * half], &si[..2 * half], &sin[..half]);
  let (dr, di) = (&mut dr[..2 * half], &mut di[..2 * half]);
  for i in 0..half {
    let (c, s, a, x, b, y) = (cos[i], sin[i], sr[2 * i], sr[2 * i + 1], si[2 * i], si[2 * i + 1]);
    dr[i] = s.mul_add(y, c.mul_add(x, a));
    di[i] = (-s).mul_add(x, c.mul_add(y, b));
    dr[i + half] = (-s).mul_add(y, (-c).mul_add(x, a));
    di[i + half] = s.mul_add(x, (-c).mul_add(y, b));
  }
}

#[cfg(test)]
mod tests {
  use super::{Fft, Float};

  fn check<T: Float + Into<f64>>(n: usize, tol: f64) {
    let x: Vec<f64> = (0..n).map(|i| ((i * 7919 % 1000) as f64 / 500.0 - 1.0) * 0.3).collect();
    let input: Vec<T> = x.iter().map(|&v| T::from_f64(v)).collect();
    let mut buf = vec![T::default(); 4 * n];
    let (re, im) = Fft::new(n).forward(&input, &mut buf);
    for k in 0..n {
      let (mut want_re, mut want_im) = (0f64, 0f64);
      for (i, &v) in input.iter().enumerate() {
        let phase = std::f64::consts::TAU * (k * i % n) as f64 / n as f64;
        want_re += v.into() * phase.cos();
        want_im -= v.into() * phase.sin();
      }
      let (re, im) = (re[k].into(), im[k].into());
      assert!((re - want_re).abs() < tol && (im - want_im).abs() < tol, "n {n}, bin {k}");
    }
  }

  #[test]
  fn matches_a_dft() {
    // 360 = 8 · 45 has a factor 3; 15 has no radix-2 stage.
    for n in [320, 360, 400, 512, 15] {
      check::<f32>(n, 1e-4);
      check::<f64>(n, 1e-10);
    }
  }
}
