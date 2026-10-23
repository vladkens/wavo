// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The Conformer blocks of `src/conformer.rs` as MIL ops over a fixed window of `t` frames, plus
//! the op helpers the families' subsampling uses. Activations are F16. Padded frames are zeroed
//! before the depthwise conv (`m`, `[t][1]`) and masked out of attention (`keys`, `[1][1][t]`).

use super::mil::{Program, Var};
use crate::conformer::Config;
use crate::cpu;
use crate::error::Result;
use crate::gguf::{Gguf, Type};
use crate::gpu::ConvNorm;

/// A program being written from a GGUF's tensors.
pub struct Graph<'a> {
  pub p: &'a mut Program,
  pub g: &'a Gguf,
}

fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
  let n = a.len().max(b.len());
  let dim = |s: &[usize], i: usize| if i + s.len() >= n { s[i + s.len() - n] } else { 1 };
  (0..n).map(|i| dim(a, i).max(dim(b, i))).collect()
}

impl Graph<'_> {
  pub fn f32s(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    self.g.tensor(name, dims)?.to_f32()
  }

  /// An F16 constant from the tensor `name` with ggml `dims`, shaped `shape`.
  pub fn constant(&mut self, name: &str, dims: &[usize], shape: &[usize]) -> Result<Var> {
    let v = self.f32s(name, dims)?;
    Ok(self.p.tensor(shape, &v))
  }

  pub fn op(&mut self, ty: &str, inputs: &[(&str, &Var)], shape: Vec<usize>) -> Var {
    let inputs: Vec<(&str, [&Var; 1])> = inputs.iter().map(|&(k, v)| (k, [v])).collect();
    let inputs: Vec<(&str, &[&Var])> = inputs.iter().map(|(k, v)| (*k, &v[..])).collect();
    self.p.op(ty, &inputs, shape)
  }

  pub fn unary(&mut self, ty: &str, x: &Var) -> Var {
    self.op(ty, &[("x", x)], x.shape.clone())
  }

  pub fn binary(&mut self, ty: &str, x: &Var, y: &Var) -> Var {
    self.op(ty, &[("x", x), ("y", y)], broadcast(&x.shape, &y.shape))
  }

  pub fn scale(&mut self, x: &Var, s: f32) -> Var {
    let s = self.p.scalar(s);
    self.binary("mul", x, &s)
  }

  pub fn reshape(&mut self, x: &Var, shape: &[usize]) -> Var {
    let s = self.p.list(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>());
    self.op("reshape", &[("x", x), ("shape", &s)], shape.to_vec())
  }

  pub fn transpose(&mut self, x: &Var, perm: &[usize]) -> Var {
    let p = self.p.list(&perm.iter().map(|&d| d as i64).collect::<Vec<_>>());
    self.op("transpose", &[("x", x), ("perm", &p)], perm.iter().map(|&i| x.shape[i]).collect())
  }

  pub fn slice(&mut self, x: &Var, begin: &[usize], end: &[usize]) -> Var {
    let b = self.p.list(&begin.iter().map(|&d| d as i64).collect::<Vec<_>>());
    let e = self.p.list(&end.iter().map(|&d| d as i64).collect::<Vec<_>>());
    let shape = begin.iter().zip(end).map(|(b, e)| e - b).collect();
    self.op("slice_by_index", &[("x", x), ("begin", &b), ("end", &e)], shape)
  }

  /// `x @ y` (`yᵀ` with `transpose_y`) over the last two axes.
  pub fn matmul(&mut self, x: &Var, y: &Var, transpose_y: bool) -> Var {
    let (tx, ty) = (self.p.boolean(false), self.p.boolean(transpose_y));
    let mut shape = x.shape.clone();
    let n = y.shape.len();
    *shape.last_mut().unwrap() = y.shape[if transpose_y { n - 2 } else { n - 1 }];
    let inputs = [("x", x), ("y", y), ("transpose_x", &tx), ("transpose_y", &ty)];
    self.op("matmul", &inputs, shape)
  }

  /// A 1-D or 2-D conv over `x` `[1][c][..]` with weight `[out][c / groups][k..]`, `stride`
  /// and symmetric `pad` on every spatial axis.
  pub fn conv(
    &mut self,
    x: &Var,
    w: &Var,
    b: Option<&Var>,
    stride: usize,
    pad: usize,
    groups: usize,
  ) -> Var {
    let spatial = x.shape.len() - 2;
    let strides = self.p.list(&vec![stride as i64; spatial]);
    let pads = self.p.list(&vec![pad as i64; 2 * spatial]);
    let dilations = self.p.list(&vec![1; spatial]);
    let (pad_type, groups_v) = (self.p.string("custom"), self.p.int(groups as i64));
    let mut shape = vec![1, w.shape[0]];
    for i in 0..spatial {
      shape.push((x.shape[2 + i] + 2 * pad - w.shape[2 + i]) / stride + 1);
    }
    let mut inputs = vec![
      ("x", x),
      ("weight", w),
      ("strides", &strides),
      ("pad_type", &pad_type),
      ("pad", &pads),
      ("dilations", &dilations),
      ("groups", &groups_v),
    ];
    inputs.extend(b.map(|b| ("bias", b)));
    self.op("conv", &inputs, shape)
  }

  /// The conv layer `{name}.weight` (ggml `dims`, laid out as `shape`, `[out][in / groups][k..]`)
  /// and `{name}.bias` over `x`, with `(stride, pad, groups)` as [`conv`](Self::conv) takes them.
  pub fn conv_layer(
    &mut self,
    x: &Var,
    name: &str,
    dims: &[usize],
    shape: &[usize],
    (stride, pad, groups): (usize, usize, usize),
  ) -> Result<Var> {
    let w = self.constant(&format!("{name}.weight"), dims, shape)?;
    let b = self.constant(&format!("{name}.bias"), &[shape[0]], &[shape[0]])?;
    Ok(self.conv(x, &w, Some(&b), stride, pad, groups))
  }

  /// The linear layer `{name}.weight` (ggml `dims`, `[.., in, out]`) over the last axis of `x`,
  /// with `{name}.bias` when `bias`. Q8_0 weights become int8 per output row, others F16.
  pub fn linear(&mut self, x: &Var, name: &str, dims: &[usize], bias: bool) -> Result<Var> {
    let (n_in, n_out) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let t = self.g.tensor(&format!("{name}.weight"), dims)?;
    let w = t.to_f32()?;
    let w = match t.ty {
      Type::Q8_0 => self.p.int8(n_out, n_in, &w),
      _ => self.p.tensor(&[n_out, n_in], &w),
    };
    let b =
      if bias { Some(self.constant(&format!("{name}.bias"), &[n_out], &[n_out])?) } else { None };
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = n_out;
    let mut inputs = vec![("x", x), ("weight", &w)];
    inputs.extend(b.as_ref().map(|b| ("bias", b)));
    Ok(self.op("linear", &inputs, shape))
  }

  /// LayerNorm (eps 1e-5) over the last axis with `{name}.weight` and `{name}.bias`.
  pub fn layer_norm(&mut self, x: &Var, name: &str) -> Result<Var> {
    let d = *x.shape.last().unwrap();
    let gamma = self.constant(&format!("{name}.weight"), &[d], &[d])?;
    let beta = self.constant(&format!("{name}.bias"), &[d], &[d])?;
    let (axes, eps) = (self.p.list(&[-1]), self.p.scalar(1e-5));
    let inputs = [("x", x), ("axes", &axes), ("gamma", &gamma), ("beta", &beta), ("epsilon", &eps)];
    Ok(self.op("layer_norm", &inputs, x.shape.clone()))
  }
}

/// The blocks over `x` `[t][d]`, with `m` `[t][1]` 1 on valid frames and `keys` `[1][1][t]` the
/// additive key mask. Attention is rotary with base `theta` when given, else relative.
pub fn blocks(
  gr: &mut Graph,
  c: &Config,
  theta: Option<f64>,
  x: Var,
  m: &Var,
  keys: &Var,
) -> Result<Var> {
  let (t, hd) = (x.shape[0], c.head_dim());
  let pos = theta.map(|theta| rotary(gr, t, hd, theta));
  let mut x = x;
  for i in 0..c.layers {
    let b = format!("enc.blocks.{i}.");
    x = feed_forward(gr, c, &b, "ff1", &x)?;
    x = attention(gr, c, &b, &x, keys, pos.as_ref())?;
    x = conv(gr, c, &b, &x, m)?;
    x = feed_forward(gr, c, &b, "ff2", &x)?;
    // The block output is norm_out(x), not the residual stream.
    x = gr.layer_norm(&x, &format!("{b}norm_out"))?;
  }
  Ok(x)
}

fn feed_forward(gr: &mut Graph, c: &Config, b: &str, n: &str, x: &Var) -> Result<Var> {
  let y = gr.layer_norm(x, &format!("{b}norm_{n}"))?;
  let y = gr.linear(&y, &format!("{b}{n}.linear1"), &[c.d, c.d_ff], c.bias)?;
  let y = gr.unary("silu", &y);
  let y = gr.linear(&y, &format!("{b}{n}.linear2"), &[c.d_ff, c.d], c.bias)?;
  let y = gr.scale(&y, 0.5);
  Ok(gr.binary("add", x, &y))
}

/// Cosines and sines `[t][1][hd]` of the rotary embedding (NEOX split-half: frequency f for
/// dims f and f + hd / 2), computed in f64 as the GPU path does.
fn rotary(gr: &mut Graph, t: usize, hd: usize, theta: f64) -> [Var; 2] {
  let angle = |p: usize, d: usize| p as f64 * theta.powf(-2.0 * (d % (hd / 2)) as f64 / hd as f64);
  let table = |f: fn(f64) -> f64| -> Vec<f32> {
    (0..t * hd).map(|i| f(angle(i / hd, i % hd)) as f32).collect()
  };
  [gr.p.tensor(&[t, 1, hd], &table(f64::cos)), gr.p.tensor(&[t, 1, hd], &table(f64::sin))]
}

/// `[heads][t][hd]` from `[t][d]`.
fn heads(gr: &mut Graph, c: &Config, x: &Var) -> Var {
  let x = gr.reshape(x, &[x.shape[0], c.heads, c.head_dim()]);
  gr.transpose(&x, &[1, 0, 2])
}

fn attention(
  gr: &mut Graph,
  c: &Config,
  b: &str,
  x: &Var,
  keys: &Var,
  rope: Option<&[Var; 2]>,
) -> Result<Var> {
  let (t, d, h, hd) = (x.shape[0], c.d, c.heads, c.head_dim());
  let a = |n: &str| format!("{b}attn.{n}");
  let y = gr.layer_norm(x, &format!("{b}norm_attn"))?;
  let scores = match rope {
    Some([cos, sin]) => {
      // Rotary on the normalized input of q and k: y·cos + rotate_half(y)·sin per head.
      let y3 = gr.reshape(&y, &[t, h, hd]);
      let lo = gr.slice(&y3, &[0, 0, 0], &[t, h, hd / 2]);
      let hi = gr.slice(&y3, &[0, 0, hd / 2], &[t, h, hd]);
      let hi = gr.scale(&hi, -1.0);
      let (axis, interleave) = (gr.p.int(2), gr.p.boolean(false));
      let rot = gr.p.op(
        "concat",
        &[("values", &[&hi, &lo]), ("axis", &[&axis]), ("interleave", &[&interleave])],
        vec![t, h, hd],
      );
      let (yc, rs) = (gr.binary("mul", &y3, cos), gr.binary("mul", &rot, sin));
      let yr = gr.binary("add", &yc, &rs);
      let yr = gr.reshape(&yr, &[t, d]);
      let q = gr.linear(&yr, &a("linear_q"), &[d, d], c.bias)?;
      let k = gr.linear(&yr, &a("linear_k"), &[d, d], c.bias)?;
      let (q, k) = (heads(gr, c, &q), heads(gr, c, &k));
      gr.matmul(&q, &k, true)
    }
    None => {
      // Transformer-XL: (q + u)·kᵀ + rel_shift((q + v)·pᵀ), p the projected positions
      // t − 1 … 1 − t, constant for the window.
      let q = gr.linear(&y, &a("linear_q"), &[d, d], c.bias)?;
      let q = gr.reshape(&q, &[t, h, hd]);
      let k = gr.linear(&y, &a("linear_k"), &[d, d], c.bias)?;
      let k = heads(gr, c, &k);
      let u = gr.constant(&a("pos_bias_u"), &[hd, h], &[h, hd])?;
      let qu = gr.binary("add", &q, &u);
      let qu = gr.transpose(&qu, &[1, 0, 2]);
      let ac = gr.matmul(&qu, &k, true);
      let v = gr.constant(&a("pos_bias_v"), &[hd, h], &[h, hd])?;
      let qv = gr.binary("add", &q, &v);
      let qv = gr.transpose(&qv, &[1, 0, 2]);
      let p = positions(gr, &a("linear_pos.weight"), t, d, h)?;
      let bd = gr.matmul(&qv, &p, false); // [h][t][2t − 1]
      // NeMo's rel_shift: a zero column in front, viewed as [h][2t][t] without its first row,
      // viewed back; then the first t columns.
      let n = 2 * t - 1;
      let pad = gr.p.list(&[0, 0, 0, 0, 1, 0]);
      let (mode, zero) = (gr.p.string("constant"), gr.p.scalar(0.0));
      let bd = gr.op(
        "pad",
        &[("x", &bd), ("pad", &pad), ("mode", &mode), ("constant_val", &zero)],
        vec![h, t, n + 1],
      );
      let bd = gr.reshape(&bd, &[h, n + 1, t]);
      let bd = gr.slice(&bd, &[0, 1, 0], &[h, n + 1, t]);
      let bd = gr.reshape(&bd, &[h, t, n]);
      let bd = gr.slice(&bd, &[0, 0, 0], &[h, t, t]);
      gr.binary("add", &ac, &bd)
    }
  };
  let v = gr.linear(&y, &a("linear_v"), &[d, d], c.bias)?;
  let v = heads(gr, c, &v);
  let s = gr.scale(&scores, 1.0 / (hd as f32).sqrt());
  let s = gr.binary("add", &s, keys);
  let axis = gr.p.int(-1);
  let s = gr.op("softmax", &[("x", &s), ("axis", &axis)], s.shape.clone());
  let o = gr.matmul(&s, &v, false);
  let o = gr.transpose(&o, &[1, 0, 2]);
  let o = gr.reshape(&o, &[t, d]);
  let o = gr.linear(&o, &a("linear_out"), &[d, d], c.bias)?;
  Ok(gr.binary("add", x, &o))
}

/// `linear_pos` of the sinusoidal table for positions t − 1 down to 1 − t, as `[h][hd][2t − 1]`.
fn positions(gr: &mut Graph, name: &str, t: usize, d: usize, h: usize) -> Result<Var> {
  let n = 2 * t - 1;
  let table = crate::conformer::rel_positions(d, t as i64 - 1, n);
  let w = cpu::rows4(gr.f32s(name, &[d, d])?, d);
  let p = cpu::matmul(&w, &table, d); // [n][d], d a multiple of 4
  let hd = d / h;
  let p: Vec<f32> = (0..d * n).map(|i| p[(i % n) * d + i / n]).collect(); // [d = h·hd][n]
  Ok(gr.p.tensor(&[h, hd, n], &p))
}

fn conv(gr: &mut Graph, c: &Config, b: &str, x: &Var, m: &Var) -> Result<Var> {
  let (t, d, k) = (x.shape[0], c.d, c.kernel);
  let y = gr.layer_norm(x, &format!("{b}norm_conv"))?;
  let y = gr.linear(&y, &format!("{b}conv.pointwise1"), &[1, d, 2 * d], c.bias)?;
  let (value, gate) = (gr.slice(&y, &[0, 0], &[t, d]), gr.slice(&y, &[0, d], &[t, 2 * d]));
  let gate = gr.unary("sigmoid", &gate);
  let y = gr.binary("mul", &value, &gate);
  let y = gr.binary("mul", &y, m);
  let y = gr.transpose(&y, &[1, 0]);
  let y = gr.reshape(&y, &[1, d, t]);
  let w = gr.f32s(&format!("{b}conv.depthwise.weight"), &[k, 1, d])?; // [d][k]
  let (w, bias) = match c.conv_norm {
    // BatchNorm folded into the depthwise conv.
    ConvNorm::Affine => {
      let (scale, shift) = crate::conformer::batch_norm(gr.g, &format!("{b}conv.bn"), d)?;
      (w.iter().enumerate().map(|(i, w)| w * scale[i / k]).collect(), shift)
    }
    ConvNorm::LayerNorm => {
      (w, if c.bias { gr.f32s(&format!("{b}conv.depthwise.bias"), &[d])? } else { vec![0.0; d] })
    }
  };
  let (w, bias) = (gr.p.tensor(&[d, 1, k], &w), gr.p.tensor(&[d], &bias));
  let y = gr.conv(&y, &w, Some(&bias), 1, k / 2, d);
  let y = gr.reshape(&y, &[d, t]);
  let y = gr.transpose(&y, &[1, 0]);
  let y = match c.conv_norm {
    ConvNorm::LayerNorm => gr.layer_norm(&y, &format!("{b}conv.ln"))?,
    ConvNorm::Affine => y,
  };
  let y = gr.unary("silu", &y);
  let y = gr.linear(&y, &format!("{b}conv.pointwise2"), &[1, d, d], c.bias)?;
  Ok(gr.binary("add", x, &y))
}
