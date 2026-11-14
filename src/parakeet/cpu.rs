// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder on the CPU, as `encoder.rs` runs it on the GPU: the 8× conv subsampling, the
//! Conformer blocks with relative positions and the joint's encoder projection.

use super::encoder_frames;
use super::frontend::MELS;
use crate::conformer::{self, rel_positions};
use crate::cpu::conformer::Conformer;
use crate::cpu::gemm::{Linear, gemm};
use crate::cpu::par_each;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Epilogue, half, pos_rows};

/// The subsampling runs in chunks of this many encoder frames, so conv2's and conv3's outputs (4×
/// the blocks' activations) never exist whole.
const CHUNK: usize = 64;

/// A 3×3 conv's taps `[9][ch]` (time tap, then frequency tap) and biases.
type Taps = (Vec<f32>, Vec<f32>);

pub struct Encoder {
  /// Subsampling channels.
  ch: usize,
  conv0: Taps,
  conv2: Taps,
  conv3: Linear,
  conv5: Taps,
  conv6: Linear,
  out: Linear,
  conformer: Conformer,
  joint: Linear,
}

impl Encoder {
  pub fn new(g: &Gguf, cfg: conformer::Config, ch: usize, joint: usize) -> Result<Self> {
    let d = cfg.d;
    let f32s = |name: String, dims: &[usize]| g.tensor(&name, dims)?.to_f32();
    // Weights from [ch][time tap][frequency tap] to [tap][ch].
    let conv = |i: usize| {
      let p = format!("enc.pre_encode.conv.{i}");
      let w = f32s(format!("{p}.weight"), &[3, 3, 1, ch])?;
      let w = (0..9 * ch).map(|e| w[e % ch * 9 + e / ch]).collect();
      Ok::<_, crate::Error>((w, f32s(format!("{p}.bias"), &[ch])?))
    };
    let linear = |name: &str, dims: &[usize]| {
      let bias = f32s(format!("{name}.bias"), &dims[dims.len() - 1..])?;
      Linear::new(&[g.tensor(&format!("{name}.weight"), dims)?], bias)
    };
    Ok(Self {
      ch,
      conv0: conv(0)?,
      conv2: conv(2)?,
      conv3: linear("enc.pre_encode.conv.3", &[1, 1, ch, ch])?,
      conv5: conv(5)?,
      conv6: linear("enc.pre_encode.conv.6", &[1, 1, ch, ch])?,
      out: linear("enc.pre_encode.out", &[ch * half(half(half(MELS))), d])?,
      conformer: Conformer::load(g, cfg)?,
      joint: linear("joint.enc", &[d, joint])?,
    })
  }

  /// Encodes `mel` `[mel frames][MELS]` and returns the joint's encoder projections, one row per
  /// encoder frame (`encoder_frames`), with their width. With `valid` less than the mel frames,
  /// the frames past it are padding, masked as `encoder.rs` masks them.
  pub fn run(&self, mel: &[f32], valid: usize) -> (Vec<f32>, usize) {
    let ch = self.ch;
    let (t2, f2) = (half(half(mel.len() / MELS)), half(half(MELS)));
    let (t, f, d) = (encoder_frames(mel.len() / MELS), half(f2), self.conformer.cfg.d);
    // A chunk of conv2's output rows from 2·t0 − 1 on, then conv3's.
    let mut a = vec![0.0; (2 * CHUNK + 1) * f2 * ch];
    let mut b = vec![0.0; a.len()];
    let mut h = vec![0.0; t * f * ch];
    for t0 in (0..t).step_by(CHUNK) {
      let n = CHUNK.min(t - t0);
      let rows = 2 * n + 1;
      let (a, b) = (&mut a[..rows * f2 * ch], &mut b[..rows * f2 * ch]);
      self.first(mel, valid, a, t0);
      gemm(a, &self.conv3, b, rows * f2, Epilogue::Relu);
      depthwise(b, &self.conv5, [t2, f2, ch], &mut h[t0 * f * ch..][..n * f * ch], t0);
    }

    let mut y = vec![0.0; h.len()];
    gemm(&h, &self.conv6, &mut y, t * f, Epilogue::Relu);
    // [t][f][ch] to [t][ch · f].
    for (h, y) in h.chunks_exact_mut(f * ch).zip(y.chunks_exact(f * ch)) {
      for (e, h) in h.iter_mut().enumerate() {
        *h = y[e % f * ch + e / f];
      }
    }

    let mut x = vec![0.0; t * d];
    gemm(&h, &self.out, &mut x, t, Epilogue::Bias);
    let x = self.conformer.run(x, t, &rel_positions(d, t as i64 + 15, pos_rows(t)));
    let mut out = vec![0.0; t * self.joint.n];
    gemm(&x, &self.joint, &mut out, t, Epilogue::Bias);
    (out, self.joint.n)
  }

  /// conv0 + ReLU over the valid mel frames, then conv2, into `y`: its output rows from
  /// 2·`t0` − 1 on `[rows][MELS / 4][ch]`, those from ⌈valid / 4⌉ on zero, as if masked. conv0's
  /// rows are computed per output row and never stored whole.
  fn first(&self, mel: &[f32], valid: usize, y: &mut [f32], t0: usize) {
    let ch = self.ch;
    let (t1, f1) = (half(valid), half(MELS));
    let f2 = half(f1);
    let ((w0, b0), (w2, b2)) = (&self.conv0, &self.conv2);
    par_each(
      y.chunks_exact_mut(f2 * ch).enumerate(),
      || vec![0.0; 3 * f1 * ch],
      |c0, (k, y)| {
        // Output row 2·t0 + k − 1 reads conv0's rows 4·t0 + 2·k − 3.., zero outside 0..t1.
        for (i, row) in c0.chunks_exact_mut(f1 * ch).enumerate() {
          let r = (4 * t0 + 2 * k + i).wrapping_sub(3);
          if r >= t1 {
            row.fill(0.0);
            continue;
          }

          for (c, px) in row.chunks_exact_mut(ch).enumerate() {
            px.copy_from_slice(b0);
            for (tap, w) in w0.chunks_exact(ch).enumerate() {
              let (tm, fm) = ((2 * r + tap / 3).wrapping_sub(1), (2 * c + tap % 3).wrapping_sub(1));
              let x = if tm < valid && fm < MELS { mel[tm * MELS + fm] } else { 0.0 };
              px.iter_mut().zip(w).for_each(|(p, w)| *p += w * x);
            }

            px.iter_mut().for_each(|p| *p = p.max(0.0));
          }
        }

        let masked = 2 * t0 + k > half(t1);
        for (fo, y) in y.chunks_exact_mut(ch).enumerate() {
          if masked {
            y.fill(0.0);
            continue;
          }

          y.copy_from_slice(b2);
          for (tap, w) in w2.chunks_exact(ch).enumerate() {
            let Some(c) = (2 * fo + tap % 3).checked_sub(1).filter(|&c| c < f1) else { continue };
            let x = &c0[(tap / 3 * f1 + c) * ch..][..ch];
            y.iter_mut().zip(w).zip(x).for_each(|((y, w), x)| *y += w * x);
          }
        }
      },
    );
  }
}

/// Depthwise 3×3 conv, stride 2, padding 1, + bias over the image `[t][f][ch]` into the output
/// rows t0.. that `y` `[rows][⌈f/2⌉][ch]` holds; `x` holds the image's rows from 2·t0 − 1 on.
fn depthwise(x: &[f32], (w, b): &Taps, image: [usize; 3], y: &mut [f32], t0: usize) {
  let [t, f, ch] = image;
  par_each(
    y.chunks_exact_mut(half(f) * ch).enumerate(),
    || (),
    |_, (k, y)| {
      for (fo, y) in y.chunks_exact_mut(ch).enumerate() {
        y.copy_from_slice(b);
        for (tap, w) in w.chunks_exact(ch).enumerate() {
          // Input row 2·(t0 + k) − 1 + tap / 3 is row 2·k + tap / 3 of `x`.
          let (ti, fi) = (2 * (t0 + k) + tap / 3, 2 * fo + tap % 3);
          if ti < 1 || ti > t || fi < 1 || fi > f {
            continue;
          }

          let x = &x[((2 * k + tap / 3) * f + fi - 1) * ch..][..ch];
          y.iter_mut().zip(w).zip(x).for_each(|((y, w), x)| *y += w * x);
        }
      }
    },
  );
}
