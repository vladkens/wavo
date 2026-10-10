// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features as transcribe.cpp computes them for Parakeet (`src/transcribe-mel.cpp`):
//! pre-emphasis in f64 before 256 zeros of padding on each side, symmetric Hann(400) centred in an
//! f64 FFT of 512, power cast to f32, Slaney filterbank, `ln(mel + 2⁻²⁴)`, then each mel bin
//! normalized over all frames but the last, which is zero. The GGUF's dither is never applied.

use std::f64::consts::PI;
use std::ops::Range;

use crate::fft::Fft;

pub const MELS: usize = 128;
pub const N_FFT: usize = 512;
pub const WINDOW: usize = 400;
pub const HOP: usize = 160;
pub const PRE_EMPHASIS: f32 = 0.97;
pub const F_MAX: f32 = 8000.0;
const BINS: usize = N_FFT / 2 + 1;
const LOG_EPS: f32 = 1.0 / (1 << 24) as f32;

pub struct Frontend {
  window: Vec<f64>,
  /// `[MELS][BINS]`
  filters: Vec<f32>,
  /// Per filter, its nonzero bins.
  spans: Vec<Range<usize>>,
  fft: Fft<f64>,
}

impl Frontend {
  pub fn new() -> Self {
    let pad = (N_FFT - WINDOW) / 2;
    let window = (0..N_FFT)
      .map(|i| match i.wrapping_sub(pad) {
        k if k < WINDOW => 0.5 - 0.5 * (2.0 * PI * k as f64 / (WINDOW - 1) as f64).cos(),
        _ => 0.0,
      })
      .collect();
    let filters = slaney_filterbank();
    let spans = (filters.as_chunks::<BINS>().0.iter())
      .map(|f| {
        let lo = f.iter().position(|&w| w != 0.0).unwrap_or(0);
        lo..f.iter().rposition(|&w| w != 0.0).map_or(lo, |i| i + 1)
      })
      .collect();
    Self { window, filters, spans, fft: Fft::new(N_FFT) }
  }

  /// Features of 16 kHz PCM, time-major `[pcm.len() / 160 + 1][MELS]`; empty below 3 frames, too
  /// few for the normalization.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    let frames = pcm.len() / HOP + 1;
    if frames < 3 {
      return Vec::new();
    }
    let mut padded = vec![0f64; pcm.len() + N_FFT];
    padded[N_FFT / 2] = pcm[0] as f64;
    for (y, x) in padded[N_FFT / 2 + 1..].iter_mut().zip(pcm.windows(2)) {
      *y = x[1] as f64 - PRE_EMPHASIS as f64 * x[0] as f64;
    }
    let mut mel = Vec::with_capacity(frames * MELS);
    let (mut frame, mut buf) = ([0f64; N_FFT], [0f64; 4 * N_FFT]);
    let mut power = [0f32; BINS];
    for t in 0..frames {
      for ((x, s), w) in frame.iter_mut().zip(&padded[t * HOP..]).zip(&self.window) {
        *x = s * w;
      }
      let (re, im) = self.fft.forward(&frame, &mut buf);
      for ((p, re), im) in power.iter_mut().zip(re).zip(im) {
        *p = (re * re + im * im) as f32;
      }
      // Each filter as a chain of f32 fused multiply-adds in bin order: bitwise equal to the
      // reference's sgemm on Apple Silicon. The zero weights outside a span would add exact zeros.
      for (filter, span) in self.filters.as_chunks::<BINS>().0.iter().zip(&self.spans) {
        let e = (filter[span.clone()].iter().zip(&power[span.clone()]))
          .fold(0f32, |s, (w, p)| w.mul_add(*p, s));
        mel.push((e + LOG_EPS).ln());
      }
    }
    // Unbiased statistics over all frames but the last.
    let n = frames - 1;
    for m in 0..MELS {
      let column = || (0..n).map(|t| mel[t * MELS + m] as f64);
      let mean = column().sum::<f64>() / n as f64;
      let var = column().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
      let inv = 1.0 / (var.sqrt() + 1e-5f32 as f64);
      for t in 0..n {
        mel[t * MELS + m] = ((mel[t * MELS + m] as f64 - mean) * inv) as f32;
      }
      mel[n * MELS + m] = 0.0;
    }
    mel
  }
}

/// librosa's `mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm='slaney')` as
/// `build_mel_filterbank_slaney` in transcribe.cpp's `src/transcribe-mel.cpp` computes it.
fn slaney_filterbank() -> Vec<f32> {
  const FSP: f64 = 200.0 / 3.0;
  const MIN_LOG_HZ: f64 = 1000.0;
  let min_log_mel = MIN_LOG_HZ / FSP;
  let log_step = 6.4f64.ln() / 27.0;
  let hz_to_mel = |hz: f64| {
    if hz < MIN_LOG_HZ { hz / FSP } else { min_log_mel + (hz / MIN_LOG_HZ).ln() / log_step }
  };
  let mel_to_hz = |mel: f64| {
    if mel < min_log_mel { mel * FSP } else { MIN_LOG_HZ * (log_step * (mel - min_log_mel)).exp() }
  };
  let mel_max = hz_to_mel(F_MAX as f64);
  let hz: Vec<f64> =
    (0..MELS + 2).map(|m| mel_to_hz(mel_max * m as f64 / (MELS + 1) as f64)).collect();
  let mut out = vec![0f32; MELS * BINS];
  for (m, row) in out.as_chunks_mut::<BINS>().0.iter_mut().enumerate() {
    let enorm = 2.0 / (hz[m + 2] - hz[m]);
    for (k, w) in row.iter_mut().enumerate() {
      let f = 16000.0 * k as f64 / N_FFT as f64;
      let lower = (f - hz[m]) / (hz[m + 1] - hz[m]);
      let upper = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
      *w = (lower.min(upper).max(0.0) * enorm) as f32;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  #[test]
  fn short_input() {
    let frontend = super::Frontend::new();
    assert!(frontend.compute(&[0.1; 319]).is_empty());
    let mel = frontend.compute(&[0.1; 320]);
    assert_eq!(mel.len(), 3 * super::MELS);
    assert!(mel.iter().all(|x| x.is_finite()));
  }
}
