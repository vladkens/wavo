// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder on the CPU, as `encoder.rs` runs it on the GPU: the two conv1d + GELU over the mel
//! frames, onto the stored positions, then the shared blocks.

use super::encoder::Config;
use crate::cpu::gemm::{Linear, gemm};
use crate::cpu::ops::im2col;
use crate::cpu::transformer::Encoder as Blocks;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::Epilogue;
use crate::transformer;

pub struct Encoder {
  cfg: Config,
  conv0: Linear,
  conv1: Linear,
  /// `[frames][d]`, the start of the residual stream.
  positions: Vec<f32>,
  blocks: Blocks,
}

impl Encoder {
  pub fn load(g: &Gguf, cfg: Config, blocks: transformer::Config) -> Result<Self> {
    let d = blocks.d;
    let conv = |name: &str, ch: usize| {
      let bias = g.tensor(&format!("{name}.bias"), &[d])?.to_f32()?;
      Linear::new(&[g.tensor(&format!("{name}.weight"), &[3, ch, d])?], bias)
    };
    Ok(Self {
      conv0: conv("enc.conv.0", cfg.mels)?,
      conv1: conv("enc.conv.1", d)?,
      positions: g.tensor("enc.pos_emb.weight", &[d, cfg.frames])?.to_f32()?,
      blocks: Blocks::load(g, blocks)?,
      cfg,
    })
  }

  /// Encodes a window of mel frames `[2·frames][mels]`; returns `[frames][d]`.
  pub fn run(&self, mel: &[f32]) -> Vec<f32> {
    let (t, d, mels) = (self.cfg.frames, self.blocks.cfg.d, self.cfg.mels);
    let mut h = vec![0.0; 2 * t * d];
    gemm(&im2col(mel, 2 * t, mels, [3, 1, 1]), &self.conv0, &mut h, 2 * t, Epilogue::Gelu);
    let mut x = self.positions.clone();
    gemm(&im2col(&h, 2 * t, d, [3, 2, 1]), &self.conv1, &mut x, t, Epilogue::GeluResidual);
    self.blocks.run(x, t, None)
  }
}
