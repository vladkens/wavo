// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Encoder on the GPU: two conv1d + GELU (kernel 3; stride 1, then 2) over the mel frames, the
//! sinusoidal positions (stored in the GGUF), then the shared pre-LN blocks and final LayerNorm.

use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Epilogue, Gpu, Linear, Pass};
use crate::transformer::{self, Encoder as Blocks};

pub struct Config {
  pub mels: usize,
  /// Encoder frames: half the mel frames of a window.
  pub frames: usize,
}

pub struct Encoder {
  pub cfg: Config,
  conv0: Linear,
  conv1: Linear,
  /// `[frames][d]`, the start of the residual stream.
  pub positions: Vec<f32>,
  pub blocks: Blocks,
}

/// Activations. `t.qkv` also holds the convs' im2col columns, `t.h` conv0's output.
pub struct Buffers {
  pub mel: Buffer,
  pub t: transformer::Buffers,
}

impl Encoder {
  pub fn load(gpu: &Gpu, g: &Gguf, cfg: Config, blocks: transformer::Config) -> Result<Self> {
    let d = blocks.d;
    let conv = |name: &str, ch: usize| {
      let bias = g.tensor(&format!("{name}.bias"), &[d])?.to_f32()?;
      gpu.conv(g.tensor(&format!("{name}.weight"), &[3, ch, d])?, &bias)
    };
    Ok(Self {
      conv0: conv("enc.conv.0", cfg.mels)?,
      conv1: conv("enc.conv.1", d)?,
      positions: g.tensor("enc.pos_emb.weight", &[d, cfg.frames])?.to_f32()?,
      blocks: Blocks::load(gpu, g, blocks)?,
      cfg,
    })
  }

  /// Buffers for a window; flash attention reads rows up to a multiple of 64.
  pub fn buffers(&self, gpu: &Gpu) -> Buffers {
    let (c, b) = (&self.cfg, &self.blocks.cfg);
    let rows = c.frames.next_multiple_of(64);
    Buffers {
      mel: gpu.buffer(2 * c.frames * c.mels),
      t: transformer::Buffers {
        x: gpu.buffer(rows * b.d),
        y: gpu.buffer(rows * b.d),
        // Also conv0's columns (2·frames × 3·mels padded to 32) and conv1's (frames × 3·d).
        qkv: gpu.buffer((rows * 3 * b.d).max(2 * c.frames * (3 * c.mels).next_multiple_of(32))),
        // Also conv0's output, 2·frames × d.
        h: gpu.buffer(c.frames * b.d_ff.max(2 * b.d)),
        s: gpu.buffer(gpu.scores_len(c.frames, b.heads, b.head_dim(), false)),
      },
    }
  }

  /// Records the encoder over `t` frames (2·t mel frames in `b.mel`); `b.t.x` must hold the
  /// positions. The output is in `b.t.y`.
  pub fn record(&self, p: &mut Pass, b: &Buffers, t: usize) {
    let (d, mels, a) = (self.blocks.cfg.d, self.cfg.mels, &b.t);
    p.im2col(&b.mel, &a.qkv, 2 * t, mels, [3, 1, 1]);
    p.gemm(&a.qkv, &self.conv0, &a.h, 2 * t, Epilogue::Gelu);
    p.im2col(&a.h, &a.qkv, 2 * t, d, [3, 2, 1]);
    p.gemm(&a.qkv, &self.conv1, &a.x, t, Epilogue::GeluResidual);
    self.blocks.record(p, a, t, None);
  }
}
