// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Conformer encoder on the GPU: two conv1d (kernel 5, stride 2) + ReLU subsampling, then blocks of
//! FFN/2, rotary self-attention, conv module, FFN/2 and LayerNorm, then the head's linear layer
//! (the RNN-T joint's encoder projection or the CTC logits). All dispatches of a call go into one
//! compute pass over a preallocated arena.

use std::sync::{Mutex, PoisonError};

use super::Config;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{BindGroup, Buffer, Epilogue, Gpu, Linear, Norm, Pass};

pub struct Encoder {
  cfg: Config,
  conv0: Linear,
  conv2: Linear,
  blocks: Vec<Block>,
  head: Linear,
  /// The head's outputs; its GEMM computes them padded to `head.width()`.
  outputs: usize,
  arena: Mutex<Option<Arena>>,
}

struct Block {
  norm_ff1: Norm,
  ff1: [Linear; 2],
  norm_attn: Norm,
  /// `linear_q` stacked over `linear_k`: both take the rotated input.
  qk: Linear,
  v: Linear,
  out: Linear,
  norm_conv: Norm,
  pw1: Linear,
  dw: Buffer,
  dw_b: Buffer,
  conv_ln: Norm,
  pw2: Linear,
  norm_ff2: Norm,
  ff2: [Linear; 2],
  norm_out: Norm,
}

/// Activations for up to `frames` encoder frames, and the bind groups recorded over them.
struct Arena {
  frames: usize,
  bufs: Buffers,
  groups: Vec<BindGroup>,
}

/// Buffers are reused across roles to keep the arena small: `h` holds the FFN hidden layer, the
/// im2col columns and v; `qk` the first conv's output and the pointwise1 output; `y` the attention
/// output.
struct Buffers {
  mel: Buffer,
  x: Buffer,
  y: Buffer,
  yr: Buffer,
  qk: Buffer,
  s: Buffer,
  h: Buffer,
  out: Buffer,
  read: Buffer,
  rope: Buffer,
}

/// Output length of a conv1d with kernel 5, stride 2, padding 2.
fn half(t: usize) -> usize {
  (t - 1) / 2 + 1
}

impl Encoder {
  /// `head` names the last linear layer and gives its weight's dims.
  pub fn new(gpu: &Gpu, g: &Gguf, cfg: Config, head: (&str, &[usize])) -> Result<Self> {
    let (d, f) = (cfg.d, cfg.d_ff);
    let linear = |names: &[String], dims: &[usize]| {
      let n = dims[dims.len() - 1];
      let w = names.iter().map(|p| g.tensor(&format!("{p}.weight"), dims));
      let b = names.iter().map(|p| g.tensor(&format!("{p}.bias"), &[n]));
      gpu.linear(&w.collect::<Result<Vec<_>>>()?, &b.collect::<Result<Vec<_>>>()?)
    };
    let norm = |p: &str| -> Result<Norm> {
      gpu.norm(&g.tensor(&format!("{p}.weight"), &[d])?, &g.tensor(&format!("{p}.bias"), &[d])?)
    };
    let f32s = |name: &str, dims: &[usize]| -> Result<Buffer> {
      gpu.upload(&g.tensor(name, dims)?.to_f32()?)
    };
    let block = |i: usize| {
      let p = |name: &str| format!("enc.blocks.{i}.{name}");
      Ok(Block {
        norm_ff1: norm(&p("norm_ff1"))?,
        ff1: [linear(&[p("ff1.linear1")], &[d, f])?, linear(&[p("ff1.linear2")], &[f, d])?],
        norm_attn: norm(&p("norm_attn"))?,
        qk: linear(&[p("attn.linear_q"), p("attn.linear_k")], &[d, d])?,
        v: linear(&[p("attn.linear_v")], &[d, d])?,
        out: linear(&[p("attn.linear_out")], &[d, d])?,
        norm_conv: norm(&p("norm_conv"))?,
        pw1: linear(&[p("conv.pointwise1")], &[1, d, 2 * d])?,
        dw: f32s(&p("conv.depthwise.weight"), &[5, 1, d])?,
        dw_b: f32s(&p("conv.depthwise.bias"), &[d])?,
        conv_ln: norm(&p("conv.ln"))?,
        pw2: linear(&[p("conv.pointwise2")], &[1, d, d])?,
        norm_ff2: norm(&p("norm_ff2"))?,
        ff2: [linear(&[p("ff2.linear1")], &[d, f])?, linear(&[p("ff2.linear2")], &[f, d])?],
        norm_out: norm(&p("norm_out"))?,
      })
    };
    // Repacking and uploading dominate load time, so blocks load in parallel.
    let blocks = std::thread::scope(|s| {
      let threads: Vec<_> = (0..cfg.layers).map(|i| s.spawn(move || block(i))).collect();
      threads.into_iter().map(|t| t.join().unwrap()).collect::<Result<Vec<_>>>()
    })?;
    Ok(Self {
      conv0: linear(&["enc.pre_encode.conv.0".into()], &[5, cfg.mels, d])?,
      conv2: linear(&["enc.pre_encode.conv.2".into()], &[5, d, d])?,
      blocks,
      head: linear(&[head.0.into()], head.1)?,
      outputs: head.1[head.1.len() - 1],
      cfg,
      arena: Mutex::new(None),
    })
  }

  /// Encodes `mel` `[mel frames][mels]` and returns the head's outputs `[frames][outputs]`.
  pub fn run(&self, gpu: &Gpu, mel: &[f32]) -> Result<Vec<f32>> {
    let mel_frames = mel.len() / self.cfg.mels;
    let (frames, width) = (half(half(mel_frames)), self.head.width());
    let mut arena = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    let Arena { bufs, groups, .. } = match arena.take() {
      Some(a) if a.frames >= frames => arena.insert(a),
      old => {
        drop(old);
        arena.insert(Arena::new(gpu, &self.cfg, width, frames.next_multiple_of(64))?)
      }
    };
    gpu.write(&bufs.mel, mel);
    let out = gpu
      .run(groups, &bufs.out, &bufs.read, frames * width, |p| self.record(p, bufs, mel_frames))?;
    if width == self.outputs {
      return Ok(out);
    }
    Ok(out.chunks_exact(width).flat_map(|row| &row[..self.outputs]).copied().collect())
  }

  fn record(&self, p: &mut Pass, b: &Buffers, mel_frames: usize) {
    let c = &self.cfg;
    let (d, hd) = (c.d, c.d / c.heads);
    let t1 = half(mel_frames);
    let t = half(t1);
    p.im2col(&b.mel, &b.h, mel_frames, c.mels, t1);
    p.gemm(&b.h, &self.conv0, &b.qk, t1, Epilogue::Relu);
    p.im2col(&b.qk, &b.h, t1, d, t);
    p.gemm(&b.h, &self.conv2, &b.x, t, Epilogue::Relu);

    let (mut x, mut y) = (&b.x, &b.y);
    for k in &self.blocks {
      p.layer_norm(x, &k.norm_ff1, y, t, d);
      p.gemm(y, &k.ff1[0], &b.h, t, Epilogue::Silu);
      p.gemm(&b.h, &k.ff1[1], x, t, Epilogue::Residual(0.5));

      p.layer_norm_rope(x, &k.norm_attn, y, &b.rope, &b.yr, t, d, hd);
      p.gemm(&b.yr, &k.qk, &b.qk, t, Epilogue::Bias);
      p.gemm(y, &k.v, &b.h, t, Epilogue::Bias);
      p.attention(&b.qk, &b.h, &b.s, y, t, c.heads, hd);
      p.gemm(y, &k.out, x, t, Epilogue::Residual(1.0));

      p.layer_norm(x, &k.norm_conv, y, t, d);
      p.gemm(y, &k.pw1, &b.qk, t, Epilogue::Bias);
      p.conv_glu(&b.qk, &k.dw, &k.dw_b, &k.conv_ln, y, t, d);
      p.gemm(y, &k.pw2, x, t, Epilogue::Residual(1.0));

      p.layer_norm(x, &k.norm_ff2, y, t, d);
      p.gemm(y, &k.ff2[0], &b.h, t, Epilogue::Silu);
      p.gemm(&b.h, &k.ff2[1], x, t, Epilogue::Residual(0.5));

      // The block output is norm_out(x), not the residual stream.
      p.layer_norm(x, &k.norm_out, y, t, d);
      std::mem::swap(&mut x, &mut y);
    }
    p.gemm(x, &self.head, &b.out, t, Epilogue::Bias);
  }
}

impl Arena {
  fn new(gpu: &Gpu, c: &Config, width: usize, frames: usize) -> Result<Self> {
    let (d, hd) = (c.d, c.d / c.heads);
    // Per position: hd / 2 cosines, then hd / 2 sines. Computed in f64 because Metal's fast-math
    // sin/cos are inaccurate for large angles.
    let rope: Vec<f32> = (0..frames)
      .flat_map(|t| {
        let angles: Vec<f64> =
          (0..hd / 2).map(|i| t as f64 * c.theta.powf(-2.0 * i as f64 / hd as f64)).collect();
        let cos = angles.iter().map(|a| a.cos() as f32);
        cos.chain(angles.iter().map(|a| a.sin() as f32)).collect::<Vec<_>>()
      })
      .collect();
    let bufs = Buffers {
      mel: gpu.buffer(4 * frames * c.mels),
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      yr: gpu.buffer(frames * d),
      // Also the first conv's output: 2·frames rows of d.
      qk: gpu.buffer(frames * 2 * d),
      s: gpu.buffer(gpu.scores_len(frames, c.heads, hd)),
      // Also the im2col columns of both convs: frames rows of 5·d, 2·frames rows of 5·mels.
      h: gpu.buffer(frames * c.d_ff.max(5 * d).max(10 * c.mels)),
      out: gpu.buffer(frames * width),
      read: gpu.readback(frames * width),
      rope: gpu.upload(&rope)?,
    };
    Ok(Self { frames, bufs, groups: Vec::new() })
  }
}
