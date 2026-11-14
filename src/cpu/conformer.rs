// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The Conformer blocks of `src/conformer.rs` on the CPU, with rotary attention (GigaAM) or
//! relative positions (Parakeet).

use super::gemm::{Linear, NR, gemm};
use super::ops::{attention, conv_glu, layer_norm, relative_attention};
use crate::conformer::{Attention, Config, batch_norm};
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{ConvNorm, Epilogue, pos_rows};

pub struct Conformer {
  pub cfg: Config,
  blocks: Vec<Block>,
}

/// Per-channel weights and biases.
type Channels = (Vec<f32>, Vec<f32>);

struct Block {
  norm_ff1: Channels,
  ff1: [Linear; 2],
  norm_attn: Channels,
  mhsa: Mhsa,
  out: Linear,
  norm_conv: Channels,
  pw1: Linear,
  /// Depthwise taps `[kernel][d]` and bias.
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
  /// `[q | k | v]`, `linear_pos`, and the position biases u and v `[heads][head_dim]`.
  Relative { qkv: Linear, pos: Linear, uv: Channels },
}

/// Activations `[t][…]` of a call.
struct Buffers {
  x: Vec<f32>,
  y: Vec<f32>,
  /// FFN hidden `[t][d_ff]`; also v (rotary).
  h: Vec<f32>,
  /// `[t][q | k]` (rotary) or `[t][q | k | v]` (relative); also the pointwise1 output `[t][2·d]`.
  qk: Vec<f32>,
  /// Rotary: the normalized rows rotated. Relative: the projected positions `[pos_rows(t)][d]`.
  yr: Vec<f32>,
}

impl Conformer {
  pub fn load(g: &Gguf, cfg: Config) -> Result<Self> {
    if !cfg.head_dim().is_multiple_of(NR) {
      bail!("the CPU runs attention heads of a multiple of {NR} dims");
    }

    let c = &cfg;
    let blocks = std::thread::scope(|s| {
      let threads: Vec<_> = (0..c.layers).map(|i| s.spawn(move || Block::load(g, c, i))).collect();
      threads.into_iter().map(|t| t.join().unwrap()).collect::<Result<Vec<_>>>()
    })?;
    Ok(Self { cfg, blocks })
  }

  /// The blocks over `x` `[t][d]`, with `pos` per frame head_dim / 2 cosines, then as many sines
  /// (rotary), or the sinusoidal table `[pos_rows(t)][d]`, row r position t + 15 − r (relative).
  pub fn run(&self, x: Vec<f32>, t: usize, pos: &[f32]) -> Vec<f32> {
    let c = &self.cfg;
    let (qk, yr) = match c.attention {
      Attention::Rotary => (2 * t, t),
      Attention::Relative => (3 * t, pos_rows(t)),
    };
    let mut b = Buffers {
      y: vec![0.0; t * c.d],
      h: vec![0.0; t * c.d_ff.max(c.d)],
      qk: vec![0.0; qk * c.d],
      yr: vec![0.0; yr * c.d],
      x,
    };
    for block in &self.blocks {
      block.run(c, &mut b, t, pos);
      std::mem::swap(&mut b.x, &mut b.y);
    }

    b.x
  }
}

/// A `Channels`' weights and biases as slices.
fn n(c: &Channels) -> (&[f32], &[f32]) {
  (&c.0, &c.1)
}

impl Block {
  fn load(g: &Gguf, c: &Config, i: usize) -> Result<Self> {
    let (d, f) = (c.d, c.d_ff);
    let p = |name: &str| format!("enc.blocks.{i}.{name}");
    let f32s = |name: &str, dims: &[usize]| g.tensor(&p(name), dims)?.to_f32();
    let linear = |names: &[&str], dims: &[usize]| {
      let w = names.iter().map(|n| g.tensor(&p(&format!("{n}.weight")), dims));
      let mut b = Vec::new();
      for n in names.iter().filter(|_| c.bias) {
        b.extend(f32s(&format!("{n}.bias"), &dims[dims.len() - 1..])?);
      }

      Linear::new(&w.collect::<Result<Vec<_>>>()?, b)
    };
    let norm = |n: &str| {
      Ok::<_, crate::Error>((
        f32s(&format!("{n}.weight"), &[d])?,
        f32s(&format!("{n}.bias"), &[d])?,
      ))
    };
    let conv_norm = match c.conv_norm {
      ConvNorm::LayerNorm => norm("conv.ln")?,
      ConvNorm::Affine => batch_norm(g, &p("conv.bn"), d)?,
    };
    let dw_b = if c.bias { f32s("conv.depthwise.bias", &[d])? } else { vec![0.0; d] };
    // `[d][kernel]` in the file, `[kernel][d]` here so a tap runs over the channels.
    let w = f32s("conv.depthwise.weight", &[c.kernel, 1, d])?;
    let taps = (0..c.kernel).flat_map(|k| w.iter().skip(k).step_by(c.kernel).copied()).collect();
    let mhsa = match c.attention {
      Attention::Rotary => Mhsa::Rotary {
        qk: linear(&["attn.linear_q", "attn.linear_k"], &[d, d])?,
        v: linear(&["attn.linear_v"], &[d, d])?,
      },
      Attention::Relative => {
        let uv = |n: &str| f32s(&format!("attn.pos_bias_{n}"), &[c.head_dim(), c.heads]);
        Mhsa::Relative {
          qkv: linear(&["attn.linear_q", "attn.linear_k", "attn.linear_v"], &[d, d])?,
          pos: Linear::new(&[g.tensor(&p("attn.linear_pos.weight"), &[d, d])?], Vec::new())?,
          uv: (uv("u")?, uv("v")?),
        }
      }
    };
    Ok(Self {
      norm_ff1: norm("norm_ff1")?,
      ff1: [linear(&["ff1.linear1"], &[d, f])?, linear(&["ff1.linear2"], &[f, d])?],
      norm_attn: norm("norm_attn")?,
      mhsa,
      out: linear(&["attn.linear_out"], &[d, d])?,
      norm_conv: norm("norm_conv")?,
      pw1: linear(&["conv.pointwise1"], &[1, d, 2 * d])?,
      dw: (taps, dw_b),
      conv_norm,
      pw2: linear(&["conv.pointwise2"], &[1, d, d])?,
      norm_ff2: norm("norm_ff2")?,
      ff2: [linear(&["ff2.linear1"], &[d, f])?, linear(&["ff2.linear2"], &[f, d])?],
      norm_out: norm("norm_out")?,
    })
  }

  /// The block from `b.x` into `b.y`; `b.x` is scratch.
  fn run(&self, c: &Config, b: &mut Buffers, t: usize, pos: &[f32]) {
    let (d, f) = (c.d, c.d_ff);
    let Buffers { x, y, h, qk, yr } = b;
    let (y, h) = (&mut y[..t * d], &mut h[..t * f]);

    layer_norm(x, n(&self.norm_ff1), y, None);
    gemm(y, &self.ff1[0], h, t, Epilogue::Silu);
    gemm(h, &self.ff1[1], x, t, Epilogue::Residual(0.5));

    match &self.mhsa {
      Mhsa::Rotary { qk: w, v: wv } => {
        let (qk, yr) = (&mut qk[..t * 2 * d], &mut yr[..t * d]);
        layer_norm(x, n(&self.norm_attn), y, Some((pos, yr, c.head_dim())));
        gemm(yr, w, qk, t, Epilogue::Bias);
        let v = &mut h[..t * d];
        gemm(y, wv, v, t, Epilogue::Bias);
        attention(qk, v, y, t, c.heads);
      }
      Mhsa::Relative { qkv, pos: wp, uv } => {
        let (qkv_out, p) = (&mut qk[..t * 3 * d], &mut yr[..pos_rows(t) * d]);
        layer_norm(x, n(&self.norm_attn), y, None);
        gemm(y, qkv, qkv_out, t, Epilogue::Bias);
        gemm(pos, wp, p, pos_rows(t), Epilogue::Bias);
        relative_attention(qkv_out, p, n(uv), y, t, c.heads);
      }
    }

    gemm(y, &self.out, x, t, Epilogue::Residual(1.0));

    let qk = &mut qk[..t * 2 * d];
    layer_norm(x, n(&self.norm_conv), y, None);
    gemm(y, &self.pw1, qk, t, Epilogue::Bias);
    conv_glu(qk, n(&self.dw), n(&self.conv_norm), c.conv_norm == ConvNorm::LayerNorm, y);
    gemm(y, &self.pw2, x, t, Epilogue::Residual(1.0));

    layer_norm(x, n(&self.norm_ff2), y, None);
    gemm(y, &self.ff2[0], h, t, Epilogue::Silu);
    gemm(h, &self.ff2[1], x, t, Epilogue::Residual(0.5));

    // The block output is norm_out(x), not the residual stream.
    layer_norm(x, n(&self.norm_out), y, None);
  }
}
