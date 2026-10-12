// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder (subsampling, blocks, the head's linear layer) as a Core ML program over a 15 s
//! window, as `encoder.rs` runs it on the GPU: shorter mel is padded with zeros, and the first
//! conv's padded rows are zeroed, as the GPU path's second conv sees zero padding there.

use super::frontend::{HOP, N_FFT};
use crate::ane::{self, Graph, Program, Var};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::half;

/// Mel frames of the window: full frames only.
const MEL_FRAMES: usize = (ane::WINDOW - N_FFT) / HOP + 1;

pub fn build(g: &Gguf, p: &mut Program) -> Result<Vec<Var>> {
  let cfg = super::config(g)?;
  let (d, mels) = (cfg.conformer.d, cfg.mels);
  let t1 = half(MEL_FRAMES);
  let t = half(t1);
  let mel = p.input("mel", &[1, mels, MEL_FRAMES]);
  let m1 = p.input("m1", &[1, 1, t1]);
  let m = p.input("m", &[t, 1]);
  let keys = p.input("keys", &[1, 1, t]);
  let mut gr = Graph { p, g };
  // Conv1d weights `[kernel, in, out]` in ggml order, `[out][in][kernel]` in memory.
  let conv = |gr: &mut Graph, x: &Var, i: usize, ch: usize| -> Result<Var> {
    let name = format!("enc.pre_encode.conv.{i}");
    let w = gr.constant(&format!("{name}.weight"), &[5, ch, d], &[d, ch, 5])?;
    let b = gr.constant(&format!("{name}.bias"), &[d], &[d])?;
    let y = gr.conv(x, &w, Some(&b), 2, 2, 1);
    Ok(gr.unary("relu", &y))
  };
  let x = conv(&mut gr, &mel, 0, mels)?;
  let x = gr.binary("mul", &x, &m1);
  let x = conv(&mut gr, &x, 2, d)?; // [1][d][t]
  let x = gr.reshape(&x, &[d, t]);
  let x = gr.transpose(&x, &[1, 0]);
  let x = ane::conformer::blocks(&mut gr, &cfg.conformer, Some(cfg.theta), x, &m, &keys)?;
  gr.p.name_next("enc");
  let (name, dims) = &cfg.head;
  Ok(vec![gr.linear(&x, name, dims, true)?])
}

/// Runs the program on `mel` `[frames][mels]` (at most the window's) and returns the head's
/// outputs `[frames / 4][outputs]`.
pub fn run(model: &ane::Model, mel: &[f32], mels: usize) -> Result<Vec<f32>> {
  let n = mel.len() / mels;
  let t1 = half(MEL_FRAMES);
  let t = half(t1);
  let n2 = half(half(n));
  // Mel goes in channel-major, `[mels][frames]`, as the conv takes it.
  let mut transposed = vec![0.0; mels * MEL_FRAMES];
  for (f, row) in mel.chunks_exact(mels).enumerate() {
    row.iter().enumerate().for_each(|(c, &v)| transposed[c * MEL_FRAMES + f] = v);
  }
  let (m1, m) = (ane::mask(t1, half(n)), ane::mask(t, n2));
  let keys = ane::key_mask(t, n2);
  let out = model.predict(
    &[
      ("mel", &[1, mels, MEL_FRAMES], &transposed),
      ("m1", &[1, 1, t1], &m1),
      ("m", &[t, 1], &m),
      ("keys", &[1, 1, t], &keys),
    ],
    "enc",
  )?;
  let width = out.len() / t;
  Ok(out[..n2 * width].to_vec())
}
