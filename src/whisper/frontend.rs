// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features as transcribe.cpp computes them for Whisper (`src/transcribe-mel.cpp`): audio
//! up to 30 s zero-padded to 30 s, reflect padding of 200, periodic Hann window and Slaney
//! filterbank from the GGUF, an f32 mixed-radix FFT of 400, power, f64 mel sums, `log10(max(·,
//! 1e-10))`, the last frame dropped, then clamped to the global maximum − 8 and scaled as `(x + 4)
//! / 4`.

use std::ops::Range;

use crate::error::Result;
use crate::gguf::Gguf;

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
/// A window: 30 s of samples, 3000 frames.
pub const SAMPLES: usize = 480_000;

pub struct Frontend {
  mels: usize,
  window: Vec<f32>,
  /// `[mels][BINS]`
  filters: Vec<f32>,
  /// Per filter, the 4-bin groups that hold its nonzero weights below bin 200.
  groups: Vec<Range<usize>>,
  fft: Fft400,
}

impl Frontend {
  pub fn new(g: &Gguf, mels: usize) -> Result<Self> {
    let window = g.tensor("frontend.window", &[N_FFT])?.to_f32()?;
    let filters = g.tensor("frontend.mel_filterbank", &[BINS, mels])?.to_f32()?;
    let groups = (filters.as_chunks::<BINS>().0.iter())
      .map(|f| {
        let lo = f.iter().position(|&w| w != 0.0).unwrap_or(0) / 4;
        let hi = f.iter().rposition(|&w| w != 0.0).map_or(0, |i| i / 4 + 1);
        lo..hi.clamp(lo, (BINS - 1) / 4)
      })
      .collect();
    Ok(Self { mels, window, filters, groups, fft: Fft400::new() })
  }

  /// Features `[frames][mels]`: 3000 frames for up to 30 s, else one per 160 samples.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    let len = pcm.len().max(SAMPLES);
    let pad = N_FFT / 2;
    let mut padded = vec![0f32; len + N_FFT];
    padded[pad..pad + pcm.len()].copy_from_slice(pcm);
    for i in 0..pad {
      padded[i] = padded[2 * pad - i];
      padded[pad + len + i] = padded[pad + len - 2 - i];
    }
    let frames = len / HOP;
    let mut mel = vec![0f32; frames * self.mels];
    // Frames on four threads; a frame of zeros (padding, digital silence) is the same constant.
    let silent = 1e-10f64.log10() as f32;
    let per = frames.div_ceil(4) * self.mels;
    std::thread::scope(|s| {
      for (i, out) in mel.chunks_mut(per).enumerate() {
        let padded = &padded;
        s.spawn(move || {
          let (mut buf, mut scratch) = ([0f32; 2 * N_FFT], [0f32; 8 * N_FFT]);
          for (t, row) in out.chunks_exact_mut(self.mels).enumerate() {
            let x = &padded[(i * per / self.mels + t) * HOP..][..N_FFT];
            if x.iter().all(|&v| v == 0.0) {
              row.fill(silent);
              continue;
            }
            for ((b, &v), &w) in buf.iter_mut().zip(x).zip(&self.window) {
              *b = v * w;
            }
            self.fft.transform(&mut buf, &mut scratch, N_FFT);
            self.log_mel(&scratch, row);
          }
        });
      }
    });
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    for v in &mut mel {
      *v = (((*v as f64).max(max - 8.0) + 4.0) / 4.0) as f32;
    }
    mel
  }

  /// `log10(max(Σ w·|X|², 1e-10))` per filter from the FFT `x` (re, im pairs), summed in f64 in
  /// groups of 4 bins and the last bin alone, as the reference: the products are exact in f64, so
  /// only the grouping matters, and zero groups add exact zeros.
  fn log_mel(&self, x: &[f32], row: &mut [f32]) {
    let mut power = [0f64; BINS];
    for (p, re_im) in power.iter_mut().zip(x.as_chunks::<2>().0) {
      let [re, im] = *re_im;
      // The reference is built with FP contraction: `re * re + im * im` is a fused multiply-add.
      *p = re.mul_add(re, im * im) as f64;
    }
    for ((out, f), groups) in
      row.iter_mut().zip(self.filters.as_chunks::<BINS>().0).zip(&self.groups)
    {
      let w = |k: usize| f[k] as f64 * power[k];
      let mut sum = 0f64;
      for g in groups.clone() {
        sum += ((w(4 * g) + w(4 * g + 1)) + w(4 * g + 2)) + w(4 * g + 3);
      }
      sum += w(BINS - 1);
      *out = sum.max(1e-10).log10() as f32;
    }
  }
}

/// `mixed_radix_fft_f32` from transcribe.cpp's `src/transcribe-mel.cpp` (after whisper.cpp):
/// radix-2 levels over naive DFTs of the odd leaves (400 → 25), with an f32 table of cos and sin.
/// The reference compiles with FP contraction on arm64, so its butterflies and DFT sums are fused
/// multiply-adds, written out here.
struct Fft400 {
  cos: [f32; N_FFT],
  sin: [f32; N_FFT],
}

impl Fft400 {
  fn new() -> Self {
    let phase = |i| std::f64::consts::TAU * i as f64 / N_FFT as f64;
    Self {
      cos: std::array::from_fn(|i| phase(i).cos() as f32),
      sin: std::array::from_fn(|i| phase(i).sin() as f32),
    }
  }

  /// The DFT of the n real values at the start of `input` (2n long, the rest scratch) into the
  /// first 2n floats of `output` (8n long, the rest scratch).
  fn transform(&self, input: &mut [f32], output: &mut [f32], n: usize) {
    let step = N_FFT / n;
    if n % 2 == 1 {
      for k in 0..n {
        let (mut re, mut im) = (0f32, 0f32);
        for (i, &v) in input[..n].iter().enumerate() {
          let phase = k * i * step % N_FFT;
          re = v.mul_add(self.cos[phase], re);
          im = (-v).mul_add(self.sin[phase], im);
        }
        output[2 * k] = re;
        output[2 * k + 1] = im;
      }
      return;
    }
    let half = n / 2;
    let (data, scratch) = input.split_at_mut(n);
    for i in 0..half {
      scratch[i] = data[2 * i];
    }
    self.transform(scratch, &mut output[2 * n..], half);
    for i in 0..half {
      scratch[i] = data[2 * i + 1];
    }
    self.transform(scratch, &mut output[3 * n..], half);
    let (out, rest) = output.split_at_mut(2 * n);
    let (even, odd) = rest.split_at(n);
    for k in 0..half {
      let (c, s) = (self.cos[k * step], self.sin[k * step]);
      let (re, im) = (odd[2 * k], odd[2 * k + 1]);
      out[2 * k] = s.mul_add(im, c.mul_add(re, even[2 * k]));
      out[2 * k + 1] = (-s).mul_add(re, c.mul_add(im, even[2 * k + 1]));
      out[2 * (k + half)] = (-s).mul_add(im, (-c).mul_add(re, even[2 * k]));
      out[2 * (k + half) + 1] = s.mul_add(re, (-c).mul_add(im, even[2 * k + 1]));
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fft400_matches_a_dft() {
    let x: Vec<f32> = (0..N_FFT).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.3).collect();
    let mut input = [0f32; 2 * N_FFT];
    input[..N_FFT].copy_from_slice(&x);
    let mut out = [0f32; 8 * N_FFT];
    Fft400::new().transform(&mut input, &mut out, N_FFT);
    for k in 0..BINS {
      let (mut re, mut im) = (0f64, 0f64);
      for (i, &v) in x.iter().enumerate() {
        let a = std::f64::consts::TAU * (k * i) as f64 / N_FFT as f64;
        re += v as f64 * a.cos();
        im -= v as f64 * a.sin();
      }
      assert!((out[2 * k] as f64 - re).abs() < 1e-4 && (out[2 * k + 1] as f64 - im).abs() < 1e-4);
    }
  }
}
