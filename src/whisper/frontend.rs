// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features as transcribe.cpp computes them for Whisper (`src/transcribe-mel.cpp`): audio
//! up to 30 s zero-padded to 30 s, reflect padding of 200, periodic Hann window and Slaney
//! filterbank from the GGUF, an f32 mixed-radix FFT of 400, power, f64 mel sums, `log10(max(·,
//! 1e-10))`, the last frame dropped, then clamped to the global maximum − 8 and scaled as `(x + 4)
//! / 4`.

use std::ops::Range;

use crate::error::Result;
use crate::fft::Fft;
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
  fft: Fft<f32>,
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
    Ok(Self { mels, window, filters, groups, fft: Fft::new(N_FFT) })
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
          let (mut buf, mut fft) = ([0f32; N_FFT], [0f32; 4 * N_FFT]);
          for (t, row) in out.chunks_exact_mut(self.mels).enumerate() {
            let x = &padded[(i * per / self.mels + t) * HOP..][..N_FFT];
            if x.iter().all(|&v| v == 0.0) {
              row.fill(silent);
              continue;
            }
            for ((b, &v), &w) in buf.iter_mut().zip(x).zip(&self.window) {
              *b = v * w;
            }
            let (re, im) = self.fft.forward(&buf, &mut fft);
            self.log_mel(re, im, row);
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

  /// `log10(max(Σ w·|X|², 1e-10))` per filter from the spectrum, summed in f64 in groups of 4
  /// bins and the last bin alone, as the reference: the products are exact in f64, so only the
  /// grouping matters, and zero groups add exact zeros.
  fn log_mel(&self, re: &[f32], im: &[f32], row: &mut [f32]) {
    let mut power = [0f64; BINS];
    for ((p, &re), &im) in power.iter_mut().zip(re).zip(im) {
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
