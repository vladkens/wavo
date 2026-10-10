// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Encoder on the GPU: two conv1d (kernel 5, stride 2) + ReLU subsampling, the Conformer blocks
//! with rotary attention, then the head's linear layer (the RNN-T joint's encoder projection or the
//! CTC logits). All dispatches of a call go into one compute pass over a preallocated arena.

use std::sync::{Mutex, PoisonError};

use crate::conformer::{self, Buffers, Conformer};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Epilogue, Gpu, Groups, Linear, Pass, half};

pub struct Encoder {
  mels: usize,
  /// RoPE base: GigaAM passes `pos_emb_max_len` (5000) as theta.
  theta: f64,
  conv0: Linear,
  conv2: Linear,
  conformer: Conformer,
  head: Linear,
  /// The head's outputs; its GEMM computes them padded to `head.width()`.
  outputs: usize,
  arena: Mutex<Option<Arena>>,
}

/// Activations for up to `frames` encoder frames, and the bind groups recorded over them. The
/// block buffers also hold the subsampling: `h` the im2col columns, `qk` the first conv's output.
struct Arena {
  frames: usize,
  bufs: Buffers,
  mel: Buffer,
  out: Buffer,
  read: Buffer,
  groups: Groups,
}

impl Encoder {
  /// `head` names the last linear layer and gives its weight's dims.
  pub fn new(
    gpu: &Gpu,
    g: &Gguf,
    cfg: conformer::Config,
    mels: usize,
    theta: f64,
    head: (&str, &[usize]),
  ) -> Result<Self> {
    let d = cfg.d;
    let linear = |name: &str, dims: &[usize]| conformer::linear(gpu, g, &[name.into()], dims, true);
    Ok(Self {
      mels,
      theta,
      conv0: linear("enc.pre_encode.conv.0", &[5, mels, d])?,
      conv2: linear("enc.pre_encode.conv.2", &[5, d, d])?,
      conformer: Conformer::load(gpu, g, cfg)?,
      head: linear(head.0, head.1)?,
      outputs: head.1[head.1.len() - 1],
      arena: Mutex::new(None),
    })
  }

  /// Encodes `mel` `[mel frames][mels]` and returns the head's outputs `[frames][outputs]`.
  pub fn run(&self, gpu: &Gpu, mel: &[f32]) -> Result<Vec<f32>> {
    let mel_frames = mel.len() / self.mels;
    let (frames, width) = (half(half(mel_frames)), self.head.width());
    let mut arena = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    let a = match arena.take() {
      Some(a) if a.frames >= frames => arena.insert(a),
      old => {
        drop(old);
        arena.insert(self.arena(gpu, frames.next_multiple_of(64))?)
      }
    };
    gpu.write(&a.mel, mel);
    let Arena { bufs, mel, out, read, groups, .. } = a;
    let out =
      gpu.run(groups, out, read, frames * width, |p| self.record(p, bufs, mel, out, mel_frames))?;
    if width == self.outputs {
      return Ok(out);
    }
    Ok(out.chunks_exact(width).flat_map(|row| &row[..self.outputs]).copied().collect())
  }

  fn record(&self, p: &mut Pass, b: &Buffers, mel: &Buffer, out: &Buffer, mel_frames: usize) {
    let d = self.conformer.cfg.d;
    let t1 = half(mel_frames);
    let t = half(t1);
    p.im2col(mel, &b.h, mel_frames, self.mels, [5, 2, 2]);
    p.gemm(&b.h, &self.conv0, &b.qk, t1, Epilogue::Relu);
    p.im2col(&b.qk, &b.h, t1, d, [5, 2, 2]);
    p.gemm(&b.h, &self.conv2, &b.x, t, Epilogue::Relu);
    let x = self.conformer.record(p, b, t);
    p.gemm(x, &self.head, out, t, Epilogue::Bias);
  }

  fn arena(&self, gpu: &Gpu, frames: usize) -> Result<Arena> {
    let c = &self.conformer.cfg;
    let (d, hd) = (c.d, c.head_dim());
    // Per position: hd / 2 cosines, then hd / 2 sines. Computed in f64 because Metal's fast-math
    // sin/cos are inaccurate for large angles.
    let rope: Vec<f32> = (0..frames)
      .flat_map(|t| {
        let angles: Vec<f64> =
          (0..hd / 2).map(|i| t as f64 * self.theta.powf(-2.0 * i as f64 / hd as f64)).collect();
        let cos = angles.iter().map(|a| a.cos() as f32);
        cos.chain(angles.iter().map(|a| a.sin() as f32)).collect::<Vec<_>>()
      })
      .collect();
    let width = self.head.width();
    let bufs = Buffers {
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      // Also the im2col columns of both convs: frames rows of 5·d, 2·frames rows of 5·mels.
      h: gpu.buffer(frames * c.d_ff.max(5 * d).max(10 * self.mels)),
      // Also the first conv's output: 2·frames rows of d.
      qk: gpu.buffer(frames * 2 * d),
      s: gpu.buffer(gpu.scores_len(frames, c.heads, hd, false)),
      yr: gpu.buffer(frames * d),
      pos: gpu.upload(&rope)?,
    };
    Ok(Arena {
      frames,
      bufs,
      mel: gpu.buffer(4 * frames * self.mels),
      out: gpu.buffer(frames * width),
      read: gpu.readback(frames * width),
      groups: Vec::new(),
    })
  }
}
