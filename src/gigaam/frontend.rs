// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features: periodic Hann window and HTK filterbank from the GGUF, FFT 320, hop 160,
//! full frames only (`center=False`), power spectrum, `ln(clamp(mel, 1e-9, 1e9))`.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::Result;
use crate::gguf::Gguf;

pub const N_FFT: usize = 320;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;

pub struct Frontend {
  pub mels: usize,
  window: Vec<f32>,
  /// `[mels][BINS]`
  filters: Vec<f32>,
  fft: Arc<dyn Fft<f32>>,
}

impl Frontend {
  pub fn new(g: &Gguf, mels: usize) -> Result<Self> {
    Ok(Self {
      mels,
      window: g.tensor("frontend.window", &[N_FFT])?.to_f32(),
      filters: g.tensor("frontend.mel_filterbank", &[BINS, mels])?.to_f32(),
      fft: FftPlanner::new().plan_fft_forward(N_FFT),
    })
  }

  /// Features of 16 kHz PCM, time-major `[frames][mels]`.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    if pcm.len() < N_FFT {
      return Vec::new();
    }
    let frames = (pcm.len() - N_FFT) / HOP + 1;
    let mut spectrum: Vec<Complex<f32>> = (0..frames * N_FFT)
      .map(|i| Complex::new(pcm[i / N_FFT * HOP + i % N_FFT] * self.window[i % N_FFT], 0.0))
      .collect();
    self.fft.process(&mut spectrum);

    let mut out = Vec::with_capacity(frames * self.mels);
    let mut power = [0f32; BINS];
    for frame in spectrum.as_chunks::<N_FFT>().0 {
      for (p, x) in power.iter_mut().zip(frame) {
        *p = x.norm_sqr();
      }
      for filter in self.filters.as_chunks::<BINS>().0 {
        out.push(super::dot(filter, &power).clamp(1e-9, 1e9).ln());
      }
    }
    out
  }
}
