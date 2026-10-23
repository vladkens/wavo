// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder (subsampling, blocks, the joint's encoder projection) as a Core ML program over a
//! 15 s window, as `encoder.rs` runs it on the GPU: the mel past the valid frames is zero, and
//! each subsampling stage's rows past their valid length are zeroed where the GPU path's convs
//! see zero padding. With kestrel masking the GPU zeroes conv2's rows past its valid length but
//! conv5 still reads conv3's output of them, ReLU(bias), up to conv3's full length (one row more
//! when the mel frames are 1 mod 4), as transcribe.cpp's extent mask does: so `m2` comes after
//! conv2 and the extent mask `e2` after conv3.

use super::frontend::{HOP, MELS};
use crate::ane::{self, Graph, Program, Var};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::half;

/// Mel frames of the window, and the rows of the three stride-2 convs' outputs.
const MEL_FRAMES: usize = ane::WINDOW / HOP + 1;
const T1: usize = half(MEL_FRAMES);
const T2: usize = half(T1);
const T: usize = half(T2);

pub fn build(g: &Gguf, p: &mut Program) -> Result<Vec<Var>> {
  let (cfg, ch, joint) = super::config(g)?;
  let mel = p.input("mel", &[1, 1, MEL_FRAMES, MELS]);
  let m1 = p.input("m1", &[1, 1, T1, 1]);
  let m2 = p.input("m2", &[1, 1, T2, 1]);
  let e2 = p.input("e2", &[1, 1, T2, 1]);
  let m3 = p.input("m3", &[T, 1]);
  let keys = p.input("keys", &[1, 1, T]);
  let mut gr = Graph { p, g };
  // conv 0 (1 → ch channels), then twice depthwise and pointwise, each stride-2 stage masked.
  let conv = |gr: &mut Graph, x: &Var, i: usize, pointwise: bool| {
    let name = format!("enc.pre_encode.conv.{i}");
    match (i, pointwise) {
      (0, _) => gr.conv_layer(x, &name, &[3, 3, 1, ch], &[ch, 1, 3, 3], (2, 1, 1)),
      (_, false) => gr.conv_layer(x, &name, &[3, 3, 1, ch], &[ch, 1, 3, 3], (2, 1, ch)),
      (_, true) => gr.conv_layer(x, &name, &[1, 1, ch, ch], &[ch, ch, 1, 1], (1, 0, 1)),
    }
  };
  let x = conv(&mut gr, &mel, 0, false)?;
  let x = gr.unary("relu", &x);
  let x = gr.binary("mul", &x, &m1);
  let x = conv(&mut gr, &x, 2, false)?;
  let x = gr.binary("mul", &x, &m2);
  let x = conv(&mut gr, &x, 3, true)?;
  let x = gr.unary("relu", &x);
  let x = gr.binary("mul", &x, &e2);
  let x = conv(&mut gr, &x, 5, false)?;
  let x = conv(&mut gr, &x, 6, true)?;
  let x = gr.unary("relu", &x); // [1][ch][T][MELS / 8]
  let f = x.shape[3];
  let x = gr.transpose(&x, &[0, 2, 1, 3]);
  let x = gr.reshape(&x, &[T, ch * f]);
  let x = gr.linear(&x, "enc.pre_encode.out", &[ch * f, cfg.d], true)?;
  let x = ane::conformer::blocks(&mut gr, &cfg, None, x, &m3, &keys)?;
  gr.p.name_next("enc");
  let out = gr.linear(&x, "joint.enc", &[cfg.d, joint], true)?;
  Ok(vec![out])
}

/// Runs the program on `mel` `[frames][MELS]` (at most the window's), of which the first `valid`
/// are speech (all but the last with kestrel masking), and returns the joint's encoder
/// projections `[valid / 8][width]` with their width.
pub fn run(model: &ane::Model, mel: &[f32], valid: usize) -> Result<(Vec<f32>, usize)> {
  let frames = mel.len() / MELS;
  let (n1, n2) = (half(valid), half(half(valid)));
  let n3 = half(n2);
  let mut padded = mel[..valid * MELS].to_vec();
  padded.resize(MEL_FRAMES * MELS, 0.0);
  let (m1, m2, m3) = (ane::mask(T1, n1), ane::mask(T2, n2), ane::mask(T, n3));
  let (e2, keys) = (ane::mask(T2, half(half(frames))), ane::key_mask(T, n3));
  let out = model.predict(
    &[
      ("mel", &[1, 1, MEL_FRAMES, MELS], &padded),
      ("m1", &[1, 1, T1, 1], &m1),
      ("m2", &[1, 1, T2, 1], &m2),
      ("e2", &[1, 1, T2, 1], &e2),
      ("m3", &[T, 1], &m3),
      ("keys", &[1, 1, T], &keys),
    ],
    "enc",
  )?;
  let width = out.len() / T;
  Ok((out[..n3 * width].to_vec(), width))
}
