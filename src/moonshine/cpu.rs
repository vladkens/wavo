// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder on the CPU, as `encoder.rs` runs it on the GPU: the conv stem over raw samples,
//! then the shared blocks with partial RoPE.

use super::encoder::{STEM, frames};
use crate::cpu::gemm::{Linear, gemm};
use crate::cpu::ops::{group_norm, im2col};
use crate::cpu::transformer::Encoder as Blocks;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::Epilogue;
use crate::transformer;

pub struct Encoder {
  conv0: Linear,
  norm: (Vec<f32>, Vec<f32>),
  eps: f32,
  conv1: Linear,
  conv2: Linear,
  blocks: Blocks,
}

impl Encoder {
  pub fn load(g: &Gguf, blocks: transformer::Config, eps: f32) -> Result<Self> {
    let d = blocks.d;
    let f32s = |name: &str, n: usize| g.tensor(name, &[n])?.to_f32();
    // conv0 has no bias.
    let conv = |i: usize, ch: usize, out: usize| {
      let bias = if i == 0 { Vec::new() } else { f32s(&format!("enc.conv.{i}.bias"), out)? };
      Linear::new(&[g.tensor(&format!("enc.conv.{i}.weight"), &[STEM[i].0, ch, out])?], bias)
    };
    Ok(Self {
      conv0: conv(0, 1, d)?,
      norm: (f32s("enc.conv.norm.weight", d)?, f32s("enc.conv.norm.bias", d)?),
      eps,
      conv1: conv(1, d, 2 * d)?,
      conv2: conv(2, 2 * d, d)?,
      blocks: Blocks::load(g, blocks)?,
    })
  }

  /// Encodes `pcm` (at least one frame long); returns `[frames(n)[2]][d]`. `table` is the table
  /// of `ops::rope` from position 0.
  pub fn run(&self, pcm: &[f32], table: &[f32]) -> Vec<f32> {
    let d = self.blocks.cfg.d;
    let [t1, t2, t] = frames(pcm.len());
    let [(k0, s0), (k1, s1), (k2, s2)] = STEM;
    let mut a = vec![0.0; t1 * d];
    gemm(&im2col(pcm, pcm.len(), 1, [k0, s0, 0]), &self.conv0, &mut a, t1, Epilogue::Tanh);
    group_norm(&mut a, (&self.norm.0, &self.norm.1), self.eps);
    let mut b = vec![0.0; t2 * 2 * d];
    gemm(&im2col(&a, t1, d, [k1, s1, 0]), &self.conv1, &mut b, t2, Epilogue::Gelu);
    let mut x = vec![0.0; t * d];
    gemm(&im2col(&b, t2, 2 * d, [k2, s2, 0]), &self.conv2, &mut x, t, Epilogue::Gelu);
    self.blocks.run(x, t, Some(table))
  }
}
