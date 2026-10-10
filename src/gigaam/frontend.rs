// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features: periodic Hann window and HTK filterbank from the GGUF, FFT 320, hop 160,
//! full frames only (`center=False`), power spectrum, `ln(clamp(mel, 1e-9, 1e9))`.

use std::ops::Range;

use crate::error::Result;
use crate::fft::Fft;
use crate::gguf::Gguf;

pub const N_FFT: usize = 320;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;

pub struct Frontend {
  mels: usize,
  window: Vec<f32>,
  /// `[mels][BINS]`
  filters: Vec<f32>,
  /// Per filter, its nonzero bins widened to `dot`'s 16-bin chunks, or to the end when they reach
  /// the tail: each bin keeps its lane and the zero weights outside add exact zeros, so the sum
  /// equals `dot` over the whole row.
  spans: Vec<Range<usize>>,
  fft: Fft<f32>,
}

impl Frontend {
  pub fn new(g: &Gguf, mels: usize) -> Result<Self> {
    let filters = g.tensor("frontend.mel_filterbank", &[BINS, mels])?.to_f32()?;
    let spans = (filters.as_chunks::<BINS>().0.iter())
      .map(|f| {
        let lo = f.iter().position(|&w| w != 0.0).unwrap_or(0) / 16 * 16;
        let hi = f.iter().rposition(|&w| w != 0.0).map_or(0, |i| i + 1);
        let hi = if hi > BINS / 16 * 16 { BINS } else { hi.next_multiple_of(16) };
        lo..hi.max(lo)
      })
      .collect();
    Ok(Self {
      mels,
      window: g.tensor("frontend.window", &[N_FFT])?.to_f32()?,
      filters,
      spans,
      fft: Fft::new(N_FFT),
    })
  }

  /// Features of 16 kHz PCM, time-major `[frames][mels]`.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pcm.len() / HOP * self.mels);
    let (mut frame, mut buf) = ([0f32; N_FFT], [0f32; 4 * N_FFT]);
    let mut power = [0f32; BINS];
    for pcm in pcm.windows(N_FFT).step_by(HOP) {
      for ((x, s), w) in frame.iter_mut().zip(pcm).zip(&self.window) {
        *x = s * w;
      }
      let (re, im) = self.fft.forward(&frame, &mut buf);
      for ((p, re), im) in power.iter_mut().zip(re).zip(im) {
        *p = re * re + im * im;
      }
      for (filter, span) in self.filters.as_chunks::<BINS>().0.iter().zip(&self.spans) {
        let mel = crate::cpu::dot(&filter[span.clone()], &power[span.clone()]);
        out.push(mel.clamp(1e-9, 1e9).ln());
      }
    }
    out
  }
}
