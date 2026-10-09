// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Conformer encoder on the GPU: two conv1d (kernel 5, stride 2) + ReLU subsampling, then blocks of
//! FFN/2, rotary self-attention, conv module, FFN/2 and LayerNorm, then the joint's encoder
//! projection. All dispatches of a call go into one compute pass over a preallocated arena.

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
  joint: Linear,
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

struct Buffers {
  mel: Buffer,
  col: Buffer,
  x: Buffer,
  y: Buffer,
  yr: Buffer,
  qk: Buffer,
  v: Buffer,
  s: Buffer,
  h: Buffer,
  joint: Buffer,
  read: Buffer,
  rope: Buffer,
}

/// Output length of a conv1d with kernel 5, stride 2, padding 2.
fn half(t: usize) -> usize {
  (t - 1) / 2 + 1
}

impl Encoder {
  pub fn new(gpu: &Gpu, g: &Gguf, cfg: Config) -> Result<Self> {
    let (d, f) = (cfg.d, cfg.d_ff);
    let linear = |names: &[String], dims: &[usize]| {
      let n = dims[dims.len() - 1];
      let w = names.iter().map(|p| g.tensor(&format!("{p}.weight"), dims));
      let b = names.iter().map(|p| g.tensor(&format!("{p}.bias"), &[n]));
      gpu.linear(&w.collect::<Result<Vec<_>>>()?, &b.collect::<Result<Vec<_>>>()?)
    };
    let norm = |p: &str| -> Result<Norm> {
      Ok(gpu.norm(&g.tensor(&format!("{p}.weight"), &[d])?, &g.tensor(&format!("{p}.bias"), &[d])?))
    };
    let f32s = |name: &str, dims: &[usize]| -> Result<Buffer> {
      Ok(gpu.upload(&g.tensor(name, dims)?.to_f32()))
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
      joint: linear(&["joint.enc".into()], &[d, cfg.joint])?,
      cfg,
      arena: Mutex::new(None),
    })
  }

  /// Encodes `mel` `[mel frames][mels]` and returns the joint's encoder projection
  /// `[frames][joint]`.
  pub fn run(&self, gpu: &Gpu, mel: &[f32]) -> Result<Vec<f32>> {
    let mel_frames = mel.len() / self.cfg.mels;
    let frames = half(half(mel_frames));
    let mut arena = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    if arena.as_ref().is_some_and(|a| a.frames < frames) {
      *arena = None;
    }
    let Arena { bufs, groups, .. } =
      arena.get_or_insert_with(|| Arena::new(gpu, &self.cfg, frames.next_multiple_of(64)));
    gpu.write(&bufs.mel, mel);
    let len = frames * self.cfg.joint;
    gpu.run(groups, &bufs.joint, &bufs.read, len, |p| self.record(p, bufs, mel_frames))
  }

  fn record(&self, p: &mut Pass, b: &Buffers, mel_frames: usize) {
    let c = &self.cfg;
    let (d, hd) = (c.d, c.d / c.heads);
    let t1 = half(mel_frames);
    let t = half(t1);
    p.im2col(&b.mel, &b.col, mel_frames, c.mels, t1);
    p.gemm(&b.col, &self.conv0, &b.h, t1, Epilogue::Relu);
    p.im2col(&b.h, &b.col, t1, d, t);
    p.gemm(&b.col, &self.conv2, &b.x, t, Epilogue::Relu);

    let (mut x, mut y) = (&b.x, &b.y);
    for k in &self.blocks {
      p.layer_norm(x, &k.norm_ff1, y, t, d);
      p.gemm(y, &k.ff1[0], &b.h, t, Epilogue::Silu);
      p.gemm(&b.h, &k.ff1[1], x, t, Epilogue::Residual(0.5));

      p.layer_norm_rope(x, &k.norm_attn, y, &b.rope, &b.yr, t, d, hd);
      p.gemm(&b.yr, &k.qk, &b.qk, t, Epilogue::Bias);
      p.gemm(y, &k.v, &b.v, t, Epilogue::Bias);
      p.attention(&b.qk, &b.v, &b.s, &b.yr, t, c.heads, hd);
      p.gemm(&b.yr, &k.out, x, t, Epilogue::Residual(1.0));

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
    p.gemm(x, &self.joint, &b.joint, t, Epilogue::Bias);
  }
}

impl Arena {
  fn new(gpu: &Gpu, c: &Config, frames: usize) -> Self {
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
      col: gpu.buffer(frames * 5 * d.max(2 * c.mels)),
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      yr: gpu.buffer(frames * d),
      qk: gpu.buffer(frames * 2 * d),
      v: gpu.buffer(frames * d),
      s: gpu.buffer(gpu.scores_len(frames, c.heads, hd)),
      h: gpu.buffer(frames * c.d_ff.max(2 * d)),
      joint: gpu.buffer(frames * c.joint),
      read: gpu.readback(frames * c.joint),
      rope: gpu.upload(&rope),
    };
    Self { frames, bufs, groups: Vec::new() }
  }
}
