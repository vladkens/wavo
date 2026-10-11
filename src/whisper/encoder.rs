// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Encoder on the GPU: two conv1d + GELU (kernel 3; stride 1, then 2) over the mel frames, the
//! sinusoidal positions (stored in the GGUF), pre-LN transformer blocks (attention, then a GELU
//! FFN) and a final LayerNorm.

use crate::conformer;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Linear, Pass};

pub struct Config {
  pub mels: usize,
  pub d: usize,
  pub heads: usize,
  pub layers: usize,
  pub d_ff: usize,
  /// Encoder frames: half the mel frames of a window.
  pub frames: usize,
}

pub struct Encoder {
  pub cfg: Config,
  conv0: Linear,
  conv1: Linear,
  /// `[frames][d]`, the start of the residual stream.
  pub positions: Vec<f32>,
  blocks: Vec<Block>,
  norm: Channels,
}

struct Block {
  norm_attn: Channels,
  /// `[q | k | v]`; k has no bias.
  qkv: Linear,
  out: Linear,
  norm_ffn: Channels,
  fc1: Linear,
  fc2: Linear,
}

/// Activations. `qkv` also holds the convs' im2col columns, `h` conv0's output.
pub struct Buffers {
  pub mel: Buffer,
  /// The residual stream, then scratch.
  pub x: Buffer,
  /// The encoder's output after `record`.
  pub y: Buffer,
  pub qkv: Buffer,
  pub h: Buffer,
  s: Buffer,
}

/// A LayerNorm `{name}.weight` and `.bias` of `d` channels.
pub fn norm(gpu: &Gpu, g: &Gguf, name: &str, d: usize) -> Result<Channels> {
  let t = |p: &str| g.tensor(&format!("{name}.{p}"), &[d])?.to_f32();
  gpu.channels(&t("weight")?, &t("bias")?)
}

/// `{prefix}.{q,k,v}` stacked, k without a bias.
pub fn qkv(gpu: &Gpu, g: &Gguf, prefix: &str, d: usize) -> Result<Linear> {
  let w = ["q", "k", "v"].map(|n| g.tensor(&format!("{prefix}.{n}.weight"), &[d, d]));
  let b = |n: &str| g.tensor(&format!("{prefix}.{n}.bias"), &[d])?.to_f32();
  let bias = [b("q")?, vec![0.0; d], b("v")?].concat();
  gpu.linear(&w.into_iter().collect::<Result<Vec<_>>>()?, &bias)
}

impl Encoder {
  pub fn load(gpu: &Gpu, g: &Gguf, cfg: Config) -> Result<Self> {
    let (d, f) = (cfg.d, cfg.d_ff);
    let linear = |name: String, dims: &[usize]| conformer::linear(gpu, g, &[name], dims, true);
    let block = |i: usize| {
      let p = |n: &str| format!("enc.blocks.{i}.{n}");
      Ok(Block {
        norm_attn: norm(gpu, g, &p("norm_attn"), d)?,
        qkv: qkv(gpu, g, &p("attn"), d)?,
        out: linear(p("attn.out"), &[d, d])?,
        norm_ffn: norm(gpu, g, &p("norm_ffn"), d)?,
        fc1: linear(p("ffn.fc1"), &[d, f])?,
        fc2: linear(p("ffn.fc2"), &[f, d])?,
      })
    };
    // Repacking and uploading dominate load time, so blocks load in parallel.
    let blocks = std::thread::scope(|s| {
      let threads: Vec<_> = (0..cfg.layers).map(|i| s.spawn(move || block(i))).collect();
      threads.into_iter().map(|t| t.join().unwrap()).collect::<Result<Vec<_>>>()
    })?;
    let conv = |name: &str, ch: usize| {
      let bias = g.tensor(&format!("{name}.bias"), &[d])?.to_f32()?;
      gpu.conv(g.tensor(&format!("{name}.weight"), &[3, ch, d])?, &bias)
    };
    Ok(Self {
      conv0: conv("enc.conv.0", cfg.mels)?,
      conv1: conv("enc.conv.1", d)?,
      positions: g.tensor("enc.pos_emb.weight", &[d, cfg.frames])?.to_f32()?,
      blocks,
      norm: norm(gpu, g, "enc.final_norm", d)?,
      cfg,
    })
  }

  /// Buffers for a window; flash attention reads rows up to a multiple of 64.
  pub fn buffers(&self, gpu: &Gpu) -> Buffers {
    let c = &self.cfg;
    let rows = c.frames.next_multiple_of(64);
    Buffers {
      mel: gpu.buffer(2 * c.frames * c.mels),
      x: gpu.buffer(rows * c.d),
      y: gpu.buffer(rows * c.d),
      // Also the columns of conv0 (2·frames × 3·mels, rows padded to 32) and conv1 (frames × 3·d).
      qkv: gpu.buffer((rows * 3 * c.d).max(2 * c.frames * (3 * c.mels).next_multiple_of(32))),
      // Also conv0's output, 2·frames × d.
      h: gpu.buffer(c.frames * c.d_ff.max(2 * c.d)),
      s: gpu.buffer(gpu.scores_len(c.frames, c.heads, c.d / c.heads, false)),
    }
  }

  /// Records the encoder over `t` frames (2·t mel frames in `b.mel`); `b.x` must hold the
  /// positions. The output is in `b.y`.
  pub fn record(&self, p: &mut Pass, b: &Buffers, t: usize) {
    let (d, mels) = (self.cfg.d, self.cfg.mels);
    p.im2col(&b.mel, &b.qkv, 2 * t, mels, [3, 1, 1]);
    p.gemm(&b.qkv, &self.conv0, &b.h, 2 * t, Epilogue::Gelu);
    p.im2col(&b.h, &b.qkv, 2 * t, d, [3, 2, 1]);
    p.gemm(&b.qkv, &self.conv1, &b.x, t, Epilogue::GeluResidual);
    let (heads, hd) = (self.cfg.heads, d / self.cfg.heads);
    p.layers(&self.blocks, |p, k| {
      p.layer_norm(&b.x, &k.norm_attn, &b.y, t, d, None);
      p.gemm(&b.y, &k.qkv, &b.qkv, t, Epilogue::Bias);
      p.attention(&b.qkv, None, None, &b.s, &b.y, t, heads, hd);
      p.gemm(&b.y, &k.out, &b.x, t, Epilogue::Residual(1.0));
      p.layer_norm(&b.x, &k.norm_ffn, &b.y, t, d, None);
      p.gemm(&b.y, &k.fc1, &b.h, t, Epilogue::Gelu);
      p.gemm(&b.h, &k.fc2, &b.x, t, Epilogue::Residual(1.0));
      &b.x
    });
    p.layer_norm(&b.x, &self.norm, &b.y, t, d, None);
  }
}
