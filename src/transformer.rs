// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Pre-LN transformer pieces of the encoder–decoder families (Whisper, Moonshine), on the GPU and
//! in transcribe.cpp's tensor names. Encoder blocks `x += out(attn(LN x))`, `x += fc2(GELU(fc1(LN
//! x)))` and a final LayerNorm. The decoder runs one token per pass: its embedding (plus a learned
//! position), layers of causal self-attention over a K/V cache, cross-attention over the encoder
//! output and an FFN, a final LayerNorm and logits from the tied embedding. Caches are F16, as
//! Whisper's reference keeps them. Whisper has biases on q, v, the outputs and the LayerNorms;
//! Moonshine has none, rotates part of q and k (RoPE) and gates its decoder FFN (SwiGLU).

use crate::conformer;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Linear, Pass, attend_len};

pub struct Config {
  pub d: usize,
  pub heads: usize,
  pub layers: usize,
  pub d_ff: usize,
  /// Biases on q, v, the attention outputs and the LayerNorms (Whisper).
  pub bias: bool,
  /// Dims per head that rotate (Moonshine), 0 for learned positions (Whisper's decoder).
  pub rot: usize,
  /// The decoder FFN is SwiGLU: fc1 gives `[x | gate]`, 2·d_ff wide.
  pub gated: bool,
}

impl Config {
  pub fn head_dim(&self) -> usize {
    self.d / self.heads
  }
}

/// A LayerNorm `{name}.weight`, with `.bias` when `bias`, else none (zeros).
pub fn norm(gpu: &Gpu, g: &Gguf, name: &str, d: usize, bias: bool) -> Result<Channels> {
  let t = |p: &str| g.tensor(&format!("{name}.{p}"), &[d])?.to_f32();
  gpu.channels(&t("weight")?, &if bias { t("bias")? } else { vec![0.0; d] })
}

/// `{prefix}.{name}` for `names` stacked; with `bias`, each has its bias but k (zeros).
fn stacked(
  gpu: &Gpu,
  g: &Gguf,
  prefix: &str,
  names: &[&str],
  d: usize,
  bias: bool,
) -> Result<Linear> {
  let tensor = |n: &str, p: &str, dims: &[usize]| g.tensor(&format!("{prefix}.{n}.{p}"), dims);
  let w = names.iter().map(|n| tensor(n, "weight", &[d, d])).collect::<Result<Vec<_>>>()?;
  let mut b = Vec::new();
  for &n in names.iter().filter(|_| bias) {
    b.extend(if n == "k" { vec![0.0; d] } else { tensor(n, "bias", &[d])?.to_f32()? });
  }
  gpu.linear(&w, &b)
}

/// Loads `load(i)` for each of `n` layers on its own thread: repacking and uploading dominate load
/// time.
fn parallel<T: Send>(n: usize, load: impl Fn(usize) -> Result<T> + Sync) -> Result<Vec<T>> {
  std::thread::scope(|s| {
    let load = &load;
    let threads: Vec<_> = (0..n).map(|i| s.spawn(move || load(i))).collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
  })
}

/// Encoder activations: `x` the residual stream `[t][d]`, then scratch; `y` the output after
/// `Encoder::record`; `qkv` `[t][3·d]`; `h` the FFN hidden `[t][d_ff]`; `s` `Gpu::scores_len`
/// floats.
pub struct Buffers {
  pub x: Buffer,
  pub y: Buffer,
  pub qkv: Buffer,
  pub h: Buffer,
  pub s: Buffer,
}

pub struct Encoder {
  pub cfg: Config,
  blocks: Vec<Block>,
  norm: Channels,
}

struct Block {
  norm_attn: Channels,
  qkv: Linear,
  out: Linear,
  norm_ffn: Channels,
  fc1: Linear,
  fc2: Linear,
}

impl Encoder {
  pub fn load(gpu: &Gpu, g: &Gguf, cfg: Config) -> Result<Self> {
    let (d, f, bias) = (cfg.d, cfg.d_ff, cfg.bias);
    let blocks = parallel(cfg.layers, |i| {
      let p = |n: &str| format!("enc.blocks.{i}.{n}");
      let linear = |n: &str, dims: &[usize], b| conformer::linear(gpu, g, &[p(n)], dims, b);
      Ok(Block {
        norm_attn: norm(gpu, g, &p("norm_attn"), d, bias)?,
        qkv: stacked(gpu, g, &p("attn"), &["q", "k", "v"], d, bias)?,
        out: linear("attn.out", &[d, d], bias)?,
        norm_ffn: norm(gpu, g, &p("norm_ffn"), d, bias)?,
        fc1: linear("ffn.fc1", &[d, f], true)?,
        fc2: linear("ffn.fc2", &[f, d], true)?,
      })
    })?;
    Ok(Self { blocks, norm: norm(gpu, g, "enc.final_norm", d, bias)?, cfg })
  }

  /// Records the blocks over `t` rows of `b.x`, then the final LayerNorm into `b.y`. With rotary
  /// dims, `rope` is the table of `Pass::rope` from position 0.
  pub fn record(&self, p: &mut Pass, b: &Buffers, t: usize, rope: Option<&Buffer>) {
    let (d, heads, hd) = (self.cfg.d, self.cfg.heads, self.cfg.head_dim());
    p.layers(&self.blocks, |p, k| {
      p.layer_norm(&b.x, &k.norm_attn, &b.y, t, d, None);
      p.gemm(&b.y, &k.qkv, &b.qkv, t, Epilogue::Bias);
      if let Some(table) = rope {
        p.rope(table, &b.qkv, t, 0, [heads, hd, self.cfg.rot]);
      }
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

pub struct Decoder {
  pub cfg: Config,
  pub vocab: usize,
  /// Positions of the self-attention cache.
  pub context: usize,
  layers: Vec<Layer>,
  norm: Channels,
  /// The token embedding, also the output layer (no bias).
  embed: Linear,
  /// Learned positions `[context][d]`, or one row of zeros with rotary dims.
  positions: Buffer,
}

struct Layer {
  norm_self: Channels,
  qkv: Linear,
  self_out: Linear,
  norm_cross: Channels,
  cross_q: Linear,
  /// `[k | v]` of the encoder output.
  cross_kv: Linear,
  cross_out: Linear,
  norm_ffn: Channels,
  fc1: Linear,
  fc2: Linear,
}

/// One token's activations and the caches, F16 in `Pass::pack`'s layout.
pub struct Steps {
  x: Buffer,
  y: Buffer,
  qkv: Buffer,
  /// fc1's output: `[x | gate]` when gated, SwiGLU'd in place.
  h: Buffer,
  att: Buffer,
  /// `Pass::attend`'s scratch.
  part: Buffer,
  pub logits: Buffer,
  /// Per layer `[context][d]` words.
  own: Vec<Buffer>,
  /// Per layer `[encoder frames][d]` words.
  cross: Vec<Buffer>,
}

impl Decoder {
  pub fn load(gpu: &Gpu, g: &Gguf, cfg: Config, vocab: usize, context: usize) -> Result<Self> {
    let (d, f, bias) = (cfg.d, cfg.d_ff, cfg.bias);
    let up = if cfg.gated { 2 * f } else { f };
    let layers = |i: usize| {
      let p = |n: &str| format!("dec.blocks.{i}.{n}");
      let linear = |n: &str, dims: &[usize], b| conformer::linear(gpu, g, &[p(n)], dims, b);
      Ok(Layer {
        norm_self: norm(gpu, g, &p("norm_self"), d, bias)?,
        qkv: stacked(gpu, g, &p("self_attn"), &["q", "k", "v"], d, bias)?,
        self_out: linear("self_attn.out", &[d, d], bias)?,
        norm_cross: norm(gpu, g, &p("norm_cross"), d, bias)?,
        cross_q: linear("cross_attn.q", &[d, d], bias)?,
        cross_kv: stacked(gpu, g, &p("cross_attn"), &["k", "v"], d, bias)?,
        cross_out: linear("cross_attn.out", &[d, d], bias)?,
        norm_ffn: norm(gpu, g, &p("norm_ffn"), d, bias)?,
        fc1: linear("ffn.fc1", &[d, up], true)?,
        fc2: linear("ffn.fc2", &[f, d], true)?,
      })
    };
    let (layers, embed) = std::thread::scope(|s| {
      let layers = s.spawn(|| parallel(cfg.layers, layers));
      let embed = gpu.linear(&[g.tensor("dec.token_embd.weight", &[d, vocab])?], &[]);
      Ok::<_, crate::Error>((layers.join().unwrap()?, embed?))
    })?;
    let positions = match cfg.rot {
      0 => g.tensor("dec.pos_emb.weight", &[d, context])?.to_f32()?,
      _ => vec![0.0; d],
    };
    Ok(Self {
      layers,
      norm: norm(gpu, g, "dec.final_norm", d, bias)?,
      embed,
      positions: gpu.upload(&positions)?,
      vocab,
      context,
      cfg,
    })
  }

  pub fn steps(&self, gpu: &Gpu, frames: usize) -> Steps {
    let c = &self.cfg;
    let cache = |rows: usize| (0..c.layers).map(|_| gpu.buffer(rows * c.d)).collect();
    Steps {
      x: gpu.buffer(c.d),
      y: gpu.buffer(c.d),
      qkv: gpu.buffer(3 * c.d),
      h: gpu.buffer(if c.gated { 2 * c.d_ff } else { c.d_ff }),
      att: gpu.buffer(c.d),
      part: gpu.buffer(attend_len(frames.max(self.context), c.heads, c.head_dim())),
      logits: gpu.buffer(self.embed.width()),
      own: cache(self.context),
      cross: cache(frames),
    }
  }

  /// Records the cross-attention K and V of every layer from the encoder output `enc` (`t` rows),
  /// through `scratch` (`t` × 2·d floats).
  pub fn record_cross(&self, p: &mut Pass, b: &Steps, enc: &Buffer, scratch: &Buffer, t: usize) {
    for (l, cache) in self.layers.iter().zip(&b.cross) {
      p.gemm(enc, &l.cross_kv, scratch, t, Epilogue::Bias);
      p.pack(scratch, [0, 2 * self.cfg.d], cache, t, self.cfg.d, [t, 0]);
    }
  }

  /// Records the decoder on a token at position `pos`, attending to `frames` encoder frames; the
  /// logits end up in `b.logits`. The token is `ids[i]` (u32 bits) when the step runs. With
  /// rotary dims, `rope` is the table of `Pass::rope` from position 0.
  #[allow(clippy::too_many_arguments)]
  pub fn record_step(
    &self,
    p: &mut Pass,
    b: &Steps,
    token: (&Buffer, usize),
    pos: usize,
    frames: usize,
    rope: Option<&Buffer>,
  ) {
    let (d, heads, hd, f) = (self.cfg.d, self.cfg.heads, self.cfg.head_dim(), self.cfg.d_ff);
    let at = if self.cfg.rot == 0 { pos * d } else { 0 };
    p.embed(&self.embed, &self.positions, &b.x, token, at);
    for (l, (own, cross)) in self.layers.iter().zip(b.own.iter().zip(&b.cross)) {
      p.layer_norm_row(&b.x, &l.norm_self, &b.y, d);
      p.gemv(&b.y, &l.qkv, &b.qkv, Epilogue::Bias, 0);
      if let Some(table) = rope {
        p.rope(table, &b.qkv, 1, pos, [heads, hd, self.cfg.rot]);
      }
      // k and v (2·d floats from d) into the cache at `pos`.
      p.pack(&b.qkv, [d, 0], own, 1, d, [self.context, pos]);
      p.attend(&b.qkv, own, &b.part, &b.att, pos + 1, self.context, heads, hd);
      p.gemv(&b.att, &l.self_out, &b.x, Epilogue::Residual(1.0), 0);
      p.layer_norm_row(&b.x, &l.norm_cross, &b.y, d);
      p.gemv(&b.y, &l.cross_q, &b.qkv, Epilogue::Bias, 0);
      p.attend(&b.qkv, cross, &b.part, &b.att, frames, frames, heads, hd);
      p.gemv(&b.att, &l.cross_out, &b.x, Epilogue::Residual(1.0), 0);
      p.layer_norm_row(&b.x, &l.norm_ffn, &b.y, d);
      let act = if self.cfg.gated { Epilogue::Bias } else { Epilogue::Gelu };
      p.gemv(&b.y, &l.fc1, &b.h, act, 0);
      if self.cfg.gated {
        p.swiglu(&b.h, f);
      }
      p.gemv(&b.h, &l.fc2, &b.x, Epilogue::Residual(1.0), 0);
    }
    p.layer_norm_row(&b.x, &self.norm, &b.y, d);
    p.gemv(&b.y, &self.embed, &b.logits, Epilogue::Bias, 0);
  }
}
