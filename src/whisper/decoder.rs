// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Decoder on the GPU, one token per pass: the token's embedding plus its learned position, pre-LN
//! layers of causal self-attention over a K/V cache, cross-attention over the encoder output and a
//! GELU FFN, a final LayerNorm and logits from the tied embedding. Caches are F16, as the
//! reference keeps them; the cross K/V are computed once per window.

use super::encoder::{norm, qkv};
use crate::conformer;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Linear, Pass};

pub struct Config {
  pub d: usize,
  pub heads: usize,
  pub layers: usize,
  pub d_ff: usize,
  pub vocab: usize,
  /// Positions (448).
  pub context: usize,
}

pub struct Decoder {
  pub cfg: Config,
  layers: Vec<Layer>,
  norm: Channels,
  /// The token embedding, also the output layer (no bias).
  embed: Linear,
  /// `[context][d]`
  positions: Buffer,
}

struct Layer {
  norm_self: Channels,
  qkv: Linear,
  self_out: Linear,
  norm_cross: Channels,
  cross_q: Linear,
  /// `[k | v]` of the encoder output; k has no bias.
  cross_kv: Linear,
  cross_out: Linear,
  norm_ffn: Channels,
  fc1: Linear,
  fc2: Linear,
}

/// One token's activations and the caches. A cache row is K then V of all heads, as f16 pairs.
pub struct Buffers {
  x: Buffer,
  y: Buffer,
  qkv: Buffer,
  h: Buffer,
  att: Buffer,
  pub logits: Buffer,
  /// Per layer `[context][d]` words.
  own: Vec<Buffer>,
  /// Per layer `[encoder frames][d]` words.
  cross: Vec<Buffer>,
}

impl Decoder {
  pub fn load(gpu: &Gpu, g: &Gguf, cfg: Config) -> Result<Self> {
    let (d, f) = (cfg.d, cfg.d_ff);
    let linear = |name: String, dims: &[usize]| conformer::linear(gpu, g, &[name], dims, true);
    let layer = |i: usize| {
      let p = |n: &str| format!("dec.blocks.{i}.{n}");
      let kv = [p("cross_attn.k.weight"), p("cross_attn.v.weight")].map(|n| g.tensor(&n, &[d, d]));
      let kv_bias = [vec![0.0; d], g.tensor(&p("cross_attn.v.bias"), &[d])?.to_f32()?].concat();
      Ok(Layer {
        norm_self: norm(gpu, g, &p("norm_self"), d)?,
        qkv: qkv(gpu, g, &p("self_attn"), d)?,
        self_out: linear(p("self_attn.out"), &[d, d])?,
        norm_cross: norm(gpu, g, &p("norm_cross"), d)?,
        cross_q: linear(p("cross_attn.q"), &[d, d])?,
        cross_kv: gpu.linear(&kv.into_iter().collect::<Result<Vec<_>>>()?, &kv_bias)?,
        cross_out: linear(p("cross_attn.out"), &[d, d])?,
        norm_ffn: norm(gpu, g, &p("norm_ffn"), d)?,
        fc1: linear(p("ffn.fc1"), &[d, f])?,
        fc2: linear(p("ffn.fc2"), &[f, d])?,
      })
    };
    let (layers, embed) = std::thread::scope(|s| {
      let threads: Vec<_> = (0..cfg.layers).map(|i| s.spawn(move || layer(i))).collect();
      let embed = gpu.linear(&[g.tensor("dec.token_embd.weight", &[d, cfg.vocab])?], &[]);
      let layers = threads.into_iter().map(|t| t.join().unwrap()).collect::<Result<Vec<_>>>();
      Ok::<_, crate::Error>((layers?, embed?))
    })?;
    let positions = g.tensor("dec.pos_emb.weight", &[d, cfg.context])?.to_f32()?;
    Ok(Self {
      layers,
      norm: norm(gpu, g, "dec.final_norm", d)?,
      embed,
      positions: gpu.upload(&positions)?,
      cfg,
    })
  }

  pub fn buffers(&self, gpu: &Gpu, frames: usize) -> Buffers {
    let c = &self.cfg;
    let cache = |rows: usize| (0..c.layers).map(|_| gpu.buffer(rows * c.d)).collect();
    Buffers {
      x: gpu.buffer(c.d),
      y: gpu.buffer(c.d),
      qkv: gpu.buffer(3 * c.d),
      h: gpu.buffer(c.d_ff),
      att: gpu.buffer(c.d),
      logits: gpu.buffer(self.embed.width()),
      own: cache(c.context),
      cross: cache(frames),
    }
  }

  /// Records the cross-attention K and V of every layer from the encoder output `enc` (`t` rows),
  /// through `scratch` (`t` × 2·d floats).
  pub fn record_cross(&self, p: &mut Pass, b: &Buffers, enc: &Buffer, scratch: &Buffer, t: usize) {
    for (l, cache) in self.layers.iter().zip(&b.cross) {
      p.gemm(enc, &l.cross_kv, scratch, t, Epilogue::Bias);
      p.pack(scratch, cache, t * self.cfg.d, 0, 0);
    }
  }

  /// Records the decoder on `token` at position `pos`, attending to `frames` encoder frames; the
  /// logits end up in `b.logits`.
  pub fn record_step(&self, p: &mut Pass, b: &Buffers, token: usize, pos: usize, frames: usize) {
    let (d, heads) = (self.cfg.d, self.cfg.heads);
    let hd = d / heads;
    p.embed(&self.embed, &self.positions, &b.x, token, pos * d);
    for (l, (own, cross)) in self.layers.iter().zip(b.own.iter().zip(&b.cross)) {
      p.layer_norm(&b.x, &l.norm_self, &b.y, 1, d, None);
      p.gemv(&b.y, &l.qkv, &b.qkv, Epilogue::Bias, 0);
      // k and v (2·d floats from d) into the cache row `pos`, d words.
      p.pack(&b.qkv, own, d, d, pos * d);
      p.attend(&b.qkv, own, &b.att, pos + 1, heads, hd);
      p.gemv(&b.att, &l.self_out, &b.x, Epilogue::Residual(1.0), 0);
      p.layer_norm(&b.x, &l.norm_cross, &b.y, 1, d, None);
      p.gemv(&b.y, &l.cross_q, &b.qkv, Epilogue::Bias, 0);
      p.attend(&b.qkv, cross, &b.att, frames, heads, hd);
      p.gemv(&b.att, &l.cross_out, &b.x, Epilogue::Residual(1.0), 0);
      p.layer_norm(&b.x, &l.norm_ffn, &b.y, 1, d, None);
      p.gemv(&b.y, &l.fc1, &b.h, Epilogue::Gelu, 0);
      p.gemv(&b.h, &l.fc2, &b.x, Epilogue::Residual(1.0), 0);
    }
    p.layer_norm(&b.x, &self.norm, &b.y, 1, d, None);
    p.gemv(&b.y, &self.embed, &b.logits, Epilogue::Bias, 0);
  }
}
