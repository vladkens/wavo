// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder (subsampling, blocks, the joint's encoder projection) as a Core ML program over a
//! 15 s window, as `encoder.rs` runs it on the GPU: shorter mel is padded with zeros, and each
//! subsampling stage's padded rows are zeroed, as the GPU path's convs see zero padding there.

use super::frontend::{HOP, MELS};
use crate::ane::{self, Graph, Program, Var};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::half;

/// Mel frames of the window.
const MEL_FRAMES: usize = ane::WINDOW / HOP + 1;

pub fn build(g: &Gguf, p: &mut Program) -> Result<Vec<Var>> {
  let (cfg, ch, joint) = super::config(g)?;
  let (t1, t2) = (half(MEL_FRAMES), half(half(MEL_FRAMES)));
  let t = half(t2);
  let mel = p.input("mel", &[1, 1, MEL_FRAMES, MELS]);
  let m1 = p.input("m1", &[1, 1, t1, 1]);
  let m2 = p.input("m2", &[1, 1, t2, 1]);
  let m3 = p.input("m3", &[t, 1]);
  let keys = p.input("keys", &[1, 1, t]);
  let mut gr = Graph { p, g };
  let x = conv0(&mut gr, &mel, ch)?;
  let x = gr.unary("relu", &x);
  let x = gr.binary("mul", &x, &m1);
  let x = depthwise(&mut gr, &x, 2, ch)?;
  let x = pointwise(&mut gr, &x, 3, ch)?;
  let x = gr.unary("relu", &x);
  let x = gr.binary("mul", &x, &m2);
  let x = depthwise(&mut gr, &x, 5, ch)?;
  let x = pointwise(&mut gr, &x, 6, ch)?;
  let x = gr.unary("relu", &x); // [1][ch][t][MELS / 8]
  let f = x.shape[3];
  let x = gr.transpose(&x, &[0, 2, 1, 3]);
  let x = gr.reshape(&x, &[t, ch * f]);
  let x = gr.linear(&x, "enc.pre_encode.out", &[ch * f, cfg.d], true)?;
  let x = ane::conformer::blocks(&mut gr, &cfg, None, x, &m3, &keys)?;
  gr.p.name_next("enc");
  let out = gr.linear(&x, "joint.enc", &[cfg.d, joint], true)?;
  Ok(vec![out])
}

fn conv0(gr: &mut Graph, x: &Var, ch: usize) -> Result<Var> {
  let w = gr.constant("enc.pre_encode.conv.0.weight", &[3, 3, 1, ch], &[ch, 1, 3, 3])?;
  let b = gr.constant("enc.pre_encode.conv.0.bias", &[ch], &[ch])?;
  Ok(gr.conv(x, &w, Some(&b), 2, 1, 1))
}

fn depthwise(gr: &mut Graph, x: &Var, i: usize, ch: usize) -> Result<Var> {
  let w =
    gr.constant(&format!("enc.pre_encode.conv.{i}.weight"), &[3, 3, 1, ch], &[ch, 1, 3, 3])?;
  let b = gr.constant(&format!("enc.pre_encode.conv.{i}.bias"), &[ch], &[ch])?;
  Ok(gr.conv(x, &w, Some(&b), 2, 1, ch))
}

fn pointwise(gr: &mut Graph, x: &Var, i: usize, ch: usize) -> Result<Var> {
  let w =
    gr.constant(&format!("enc.pre_encode.conv.{i}.weight"), &[1, 1, ch, ch], &[ch, ch, 1, 1])?;
  let b = gr.constant(&format!("enc.pre_encode.conv.{i}.bias"), &[ch], &[ch])?;
  Ok(gr.conv(x, &w, Some(&b), 1, 0, 1))
}

/// Runs the program on `mel` `[frames][MELS]` (at most the window's) and returns the joint's
/// encoder projections `[frames / 8][width]` with their width.
pub fn run(model: &ane::Model, mel: &[f32]) -> Result<(Vec<f32>, usize)> {
  let n = mel.len() / MELS;
  let (t1, t2) = (half(MEL_FRAMES), half(half(MEL_FRAMES)));
  let t = half(t2);
  let (n1, n2) = (half(n), half(half(n)));
  let n3 = half(n2);
  let mut padded = mel.to_vec();
  padded.resize(MEL_FRAMES * MELS, 0.0);
  let (m1, m2, m3) = (ane::mask(t1, n1), ane::mask(t2, n2), ane::mask(t, n3));
  let keys = ane::key_mask(t, n3);
  let out = model.predict(
    &[
      ("mel", &[1, 1, MEL_FRAMES, MELS], &padded),
      ("m1", &[1, 1, t1, 1], &m1),
      ("m2", &[1, 1, t2, 1], &m2),
      ("m3", &[t, 1], &m3),
      ("keys", &[1, 1, t], &keys),
    ],
    "enc",
  )?;
  let width = out.len() / t;
  Ok((out[..n3 * width].to_vec(), width))
}
