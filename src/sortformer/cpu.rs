// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The network on the CPU, as `encoder.rs` runs it on the GPU: `embed`, the mel frames stacked by
//! 8 and projected; `predict`, the transformer and the head.

use super::encoder::STACK;
use crate::cpu::gemm::{Linear, gemm};
use crate::cpu::ops::{im2col, layer_norm};
use crate::cpu::transformer::{Block, Encoder as Blocks, Norm, linear, norm, parallel};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::Epilogue;
use crate::parakeet::frontend::MELS;
use crate::transformer::Config;

pub struct Encoder {
  stack: Linear,
  embed_norm: Norm,
  blocks: Blocks,
  /// The rotary table of `ops::rope` from position 0.
  rope: Vec<f32>,
  proj: Linear,
  conv: Linear,
  hidden: Linear,
  head: Linear,
}

impl Encoder {
  /// Loads the network of `cfg` and the head's `width`, with the rotary table `rope`.
  pub fn new(g: &Gguf, cfg: Config, width: usize, speakers: usize, rope: Vec<f32>) -> Result<Self> {
    let (d, f) = (cfg.d, cfg.d_ff);
    let blocks = parallel(cfg.layers, |i| {
      let p = |n: &str| format!("encoder.layers.{i}.{n}");
      Ok(Block {
        norm_attn: norm(g, &p("norm1"), d, true)?,
        qkv: linear(g, &p("attn.w_qkv"), &[d, 3 * d], false)?,
        out: linear(g, &p("attn.out_proj"), &[d, d], true)?,
        norm_ffn: norm(g, &p("norm2"), d, true)?,
        fc1: linear(g, &p("ffn.net.0"), &[d, f], true)?,
        fc2: linear(g, &p("ffn.net.3"), &[f, d], true)?,
      })
    })?;
    let m = |n: &str| format!("sortformer_modules.{n}");
    Ok(Self {
      stack: linear(g, "encoder.pre_encode.proj", &[STACK * MELS, d], false)?,
      embed_norm: norm(g, "encoder.embed_norm", d, true)?,
      blocks: Blocks { norm: norm(g, "encoder.final_norm", d, true)?, cfg, blocks },
      rope,
      proj: linear(g, &m("encoder_proj"), &[d, width], true)?,
      conv: linear(g, &m("subpixel_upsample"), &[3, width, STACK * width], true)?,
      hidden: linear(g, &m("first_hidden_to_hidden"), &[width, width], true)?,
      head: linear(g, &m("single_hidden_to_spks"), &[width, speakers], true)?,
    })
  }

  /// The embeddings `[⌈m / 8⌉][d]` of `m` mel frames `[m][MELS]`, the last 8 padded with zeros.
  pub fn embed(&self, mel: &[f32]) -> Vec<f32> {
    let t = (mel.len() / MELS).div_ceil(STACK);
    let mut padded = mel.to_vec();
    padded.resize(t * STACK * MELS, 0.0);
    let mut out = vec![0.0; t * self.stack.n];
    gemm(&padded, &self.stack, &mut out, t, Epilogue::Bias);
    out
  }

  /// Per output frame, 8 per embedding of `x` `[t][d]`, each speaker's probability:
  /// `[8t][speakers]`.
  pub fn predict(&self, x: &[f32]) -> Vec<f32> {
    let (d, width) = (self.blocks.cfg.d, self.proj.n);
    let t = x.len() / d;
    let mut y = vec![0.0; x.len()];
    layer_norm(x, (&self.embed_norm.0, &self.embed_norm.1), &mut y, None);
    let y = self.blocks.run(y, t, Some(&self.rope));
    let mut h = vec![0.0; t * width];
    gemm(&y, &self.proj, &mut h, t, Epilogue::Bias);
    // Each frame's 8·width outputs are 8 frames of width.
    let mut up = vec![0.0; t * STACK * width];
    gemm(&im2col(&h, t, width, [3, 1, 1]), &self.conv, &mut up, t, Epilogue::Relu);
    let mut hidden = vec![0.0; up.len()];
    gemm(&up, &self.hidden, &mut hidden, STACK * t, Epilogue::Relu);
    let mut logits = vec![0.0; STACK * t * self.head.n];
    gemm(&hidden, &self.head, &mut logits, STACK * t, Epilogue::Bias);
    logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect()
  }
}
