// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Conformer blocks on the GPU, shared by the families: `x += ½·ffn(LN x)`, `x += mhsa(LN x)`,
//! `x += conv(LN x)`, `x += ½·ffn(LN x)`, `x = LN x`. The conv module is pointwise1, GLU,
//! depthwise conv, a norm (LayerNorm, or BatchNorm folded into a per-channel affine at load), SiLU
//! and pointwise2. The attention is rotary (GigaAM) or with relative positions (Parakeet).

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, ConvNorm, Epilogue, Gpu, Linear, Pass, pos_rows};

#[derive(Clone, Copy)]
pub enum Attention {
  /// Rotary embedding on q and k.
  Rotary,
  /// Transformer-XL relative positions with learned biases u and v, no layer biases.
  Relative,
}

pub struct Config {
  pub layers: usize,
  pub d: usize,
  pub heads: usize,
  pub d_ff: usize,
  /// Depthwise conv kernel: odd, centred.
  pub kernel: usize,
  pub conv_norm: ConvNorm,
  pub attention: Attention,
  /// Linear and conv layers have biases.
  pub bias: bool,
}

impl Config {
  /// Reads `stt.<arch>.encoder.*`; the family picks the attention and whether layers have biases.
  pub fn load(g: &Gguf, arch: &str, attention: Attention, bias: bool) -> Result<Self> {
    let key = |name: &str| format!("stt.{arch}.encoder.{name}");
    let u = |name: &str| g.get::<u32>(&key(name)).map(|v| v as usize);
    let conv_norm = match g.get::<&str>(&key("conv_norm_type"))? {
      "layer_norm" => ConvNorm::LayerNorm,
      "batch_norm" => ConvNorm::Affine,
      other => bail!("unsupported conv_norm_type {other:?}"),
    };
    let cfg = Self {
      layers: u("n_layers")?,
      d: u("d_model")?,
      heads: u("n_heads")?,
      d_ff: u("d_ff")?,
      kernel: u("conv_kernel")?,
      conv_norm,
      attention,
      bias,
    };
    let (d, heads, hd) = (cfg.d, cfg.heads, cfg.head_dim());
    // Kernels keep a row of up to 1024 channels and a head of up to 128 in shared memory.
    if !d.is_multiple_of(heads) || d > 1024 || hd > 128 || !hd.is_multiple_of(2) {
      bail!("unsupported attention shape: d_model {d}, {heads} heads");
    }
    if cfg.kernel.is_multiple_of(2) {
      bail!("conv kernel {}: only odd kernels are supported", cfg.kernel);
    }
    Ok(cfg)
  }

  pub fn head_dim(&self) -> usize {
    self.d / self.heads
  }
}

/// Activations of the blocks. A family allocates them for its longest input so far and may use
/// them for its subsampling too.
pub struct Buffers {
  /// The blocks' input `[t][d]`.
  pub x: Buffer,
  pub y: Buffer,
  /// FFN hidden `[t][d_ff]`; also v for rotary attention.
  pub h: Buffer,
  /// Attention projections, `[t][q | k]` (rotary) or `[t][q + u | k | v | q + v]` (relative);
  /// also the pointwise1 output `[t][2·d]`.
  pub qk: Buffer,
  /// Attention scores: `Gpu::scores_len` floats.
  pub s: Buffer,
  /// Rotary: the normalized rows rotated `[t][d]`. Relative: the projected positions
  /// `[pos_rows(t) + 64][d]`.
  pub yr: Buffer,
  /// Rotary: per frame head_dim / 2 cosines, then as many sines. Relative: the sinusoidal position
  /// table `[pos_rows(t)][d]` (row r is position t + 15 − r).
  pub pos: Buffer,
}

pub struct Conformer {
  pub cfg: Config,
  blocks: Vec<Block>,
}

impl Conformer {
  pub fn load(gpu: &Gpu, g: &Gguf, cfg: Config) -> Result<Self> {
    // Repacking and uploading dominate load time, so blocks load in parallel.
    let c = &cfg;
    let blocks = std::thread::scope(|s| {
      let threads: Vec<_> =
        (0..c.layers).map(|i| s.spawn(move || Block::load(gpu, g, c, i))).collect();
      threads.into_iter().map(|t| t.join().unwrap()).collect::<Result<Vec<_>>>()
    })?;
    Ok(Self { cfg, blocks })
  }

  /// Records the blocks over `t` frames of `b.x`; returns the buffer that holds their output.
  pub fn record<'b>(&self, p: &mut Pass, b: &'b Buffers, t: usize) -> &'b Buffer {
    let (mut x, mut y) = (&b.x, &b.y);
    p.layers(&self.blocks, |p, k| {
      k.record(p, &self.cfg, b, x, y, t);
      std::mem::swap(&mut x, &mut y);
      x
    });
    x
  }
}

/// The linear layer `{name}.weight` with ggml `dims`, several names stacked along the output, with
/// their `{name}.bias` when `bias`.
pub fn linear(gpu: &Gpu, g: &Gguf, names: &[String], dims: &[usize], bias: bool) -> Result<Linear> {
  let w = names.iter().map(|p| g.tensor(&format!("{p}.weight"), dims));
  let mut b = Vec::new();
  for p in names.iter().filter(|_| bias) {
    b.extend(g.tensor(&format!("{p}.bias"), &dims[dims.len() - 1..])?.to_f32()?);
  }
  gpu.linear(&w.collect::<Result<Vec<_>>>()?, &b)
}

struct Block {
  norm_ff1: Channels,
  ff1: [Linear; 2],
  norm_attn: Channels,
  mhsa: Mhsa,
  out: Linear,
  norm_conv: Channels,
  pw1: Linear,
  dw: Channels,
  conv_norm: Channels,
  pw2: Linear,
  norm_ff2: Channels,
  ff2: [Linear; 2],
  norm_out: Channels,
}

enum Mhsa {
  /// `linear_q` stacked over `linear_k`: both take the rotated input.
  Rotary { qk: Linear, v: Linear },
  /// `[q | k | v]` with biases `[pos_bias_u | 0 | 0]` and q again with `pos_bias_v`, and
  /// `linear_pos`.
  Relative { qkv: Linear, pos: Linear },
}

impl Block {
  fn load(gpu: &Gpu, g: &Gguf, c: &Config, i: usize) -> Result<Self> {
    let (d, f) = (c.d, c.d_ff);
    let p = |name: &str| format!("enc.blocks.{i}.{name}");
    let linear = |names: &[&str], dims: &[usize]| {
      linear(gpu, g, &names.iter().map(|n| p(n)).collect::<Vec<_>>(), dims, c.bias)
    };
    let f32s = |name: &str, dims: &[usize]| g.tensor(&p(name), dims)?.to_f32();
    let norm = |n: &str| {
      gpu.channels(&f32s(&format!("{n}.weight"), &[d])?, &f32s(&format!("{n}.bias"), &[d])?)
    };
    let mhsa = match c.attention {
      Attention::Rotary => Mhsa::Rotary {
        qk: linear(&["attn.linear_q", "attn.linear_k"], &[d, d])?,
        v: linear(&["attn.linear_v"], &[d, d])?,
      },
      Attention::Relative => {
        let w = ["q", "k", "v"].map(|n| g.tensor(&p(&format!("attn.linear_{n}.weight")), &[d, d]));
        let uv = |n: &str| f32s(&format!("attn.pos_bias_{n}"), &[c.head_dim(), c.heads]);
        let bias = [uv("u")?, vec![0.0; 2 * d], uv("v")?].concat();
        Mhsa::Relative {
          qkv: gpu.linear(&w.into_iter().collect::<Result<Vec<_>>>()?, &bias)?,
          pos: self::linear(gpu, g, &[p("attn.linear_pos")], &[d, d], false)?,
        }
      }
    };
    let conv_norm = match c.conv_norm {
      ConvNorm::LayerNorm => norm("conv.ln")?,
      // As `fuse_batch_norm` in transcribe.cpp's `src/arch/parakeet/model.cpp`.
      ConvNorm::Affine => {
        let bn = |name: &str| f32s(&format!("conv.bn.{name}"), &[d]);
        let (w, b) = (bn("weight")?, bn("bias")?);
        let (mean, var) = (bn("running_mean")?, bn("running_var")?);
        let s: Vec<f32> = w.iter().zip(&var).map(|(w, v)| w / (v + 1e-5).sqrt()).collect();
        let b: Vec<f32> = b.iter().zip(&mean).zip(&s).map(|((b, m), s)| b - m * s).collect();
        gpu.channels(&s, &b)?
      }
    };
    let dw_b = if c.bias { f32s("conv.depthwise.bias", &[d])? } else { vec![0.0; d] };
    let dw = gpu.channels(&f32s("conv.depthwise.weight", &[c.kernel, 1, d])?, &dw_b)?;
    Ok(Self {
      norm_ff1: norm("norm_ff1")?,
      ff1: [linear(&["ff1.linear1"], &[d, f])?, linear(&["ff1.linear2"], &[f, d])?],
      norm_attn: norm("norm_attn")?,
      mhsa,
      out: linear(&["attn.linear_out"], &[d, d])?,
      norm_conv: norm("norm_conv")?,
      pw1: linear(&["conv.pointwise1"], &[1, d, 2 * d])?,
      dw,
      conv_norm,
      pw2: linear(&["conv.pointwise2"], &[1, d, d])?,
      norm_ff2: norm("norm_ff2")?,
      ff2: [linear(&["ff2.linear1"], &[d, f])?, linear(&["ff2.linear2"], &[f, d])?],
      norm_out: norm("norm_out")?,
    })
  }

  /// Records the block from `x` into `y`; `x` is scratch.
  fn record(&self, p: &mut Pass, c: &Config, b: &Buffers, x: &Buffer, y: &Buffer, t: usize) {
    let (d, hd) = (c.d, c.head_dim());
    p.layer_norm(x, &self.norm_ff1, y, t, d, None);
    p.gemm(y, &self.ff1[0], &b.h, t, Epilogue::Silu);
    p.gemm(&b.h, &self.ff1[1], x, t, Epilogue::Residual(0.5));

    match &self.mhsa {
      Mhsa::Rotary { qk, v } => {
        p.layer_norm(x, &self.norm_attn, y, t, d, Some((&b.pos, &b.yr, hd)));
        p.gemm(&b.yr, qk, &b.qk, t, Epilogue::Bias);
        p.gemm(y, v, &b.h, t, Epilogue::Bias);
        p.attention(&b.qk, Some(&b.h), None, &b.s, y, t, c.heads, hd);
      }
      Mhsa::Relative { qkv, pos } => {
        p.layer_norm(x, &self.norm_attn, y, t, d, None);
        p.gemm(y, qkv, &b.qk, t, Epilogue::Bias);
        p.gemm(&b.pos, pos, &b.yr, pos_rows(t), Epilogue::Bias);
        p.attention(&b.qk, None, Some(&b.yr), &b.s, y, t, c.heads, hd);
      }
    }
    p.gemm(y, &self.out, x, t, Epilogue::Residual(1.0));

    p.layer_norm(x, &self.norm_conv, y, t, d, None);
    p.gemm(y, &self.pw1, &b.qk, t, Epilogue::Bias);
    p.conv_glu(&b.qk, &self.dw, c.kernel, &self.conv_norm, c.conv_norm, y, t, d);
    p.gemm(y, &self.pw2, x, t, Epilogue::Residual(1.0));

    p.layer_norm(x, &self.norm_ff2, y, t, d, None);
    p.gemm(y, &self.ff2[0], &b.h, t, Epilogue::Silu);
    p.gemm(&b.h, &self.ff2[1], x, t, Epilogue::Residual(0.5));

    // The block output is norm_out(x), not the residual stream.
    p.layer_norm(x, &self.norm_out, y, t, d, None);
  }
}
