// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Encoder on the GPU: 8× subsampling of the mel image `[t][freq]` by 3×3 stride-2 convs (conv0 +
//! ReLU, depthwise conv2, pointwise conv3 + ReLU, depthwise conv5, pointwise conv6 + ReLU),
//! flattened to `[t][channel · 16 + freq]` and projected to d_model, then the Conformer blocks with
//! relative positions, then the joint's encoder projection. All dispatches of a call go into one
//! compute pass over a preallocated arena.

use std::sync::{Mutex, PoisonError};

use super::frontend::MELS;
use crate::conformer::{self, Buffers, Conformer};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{BindGroup, Buffer, Depthwise, Epilogue, Gpu, Linear, Pass, half};

pub struct Encoder {
  /// Subsampling channels.
  ch: usize,
  conv0: Depthwise,
  conv2: Depthwise,
  conv3: Linear,
  conv5: Depthwise,
  conv6: Linear,
  out: Linear,
  conformer: Conformer,
  joint: Linear,
  outputs: usize,
  arena: Mutex<Option<Arena>>,
}

/// Activations for up to `frames` encoder frames, and the bind groups recorded over them. The
/// block buffers also hold the subsampling: `h` the depthwise and flattened outputs, `qk` the
/// pointwise ones.
struct Arena {
  frames: usize,
  bufs: Buffers,
  /// Sinusoidal positions for `frames`: `2·frames − 1` rows of d, row r is position
  /// `frames − 1 − r`. A call over t frames takes its `2t − 1` rows from row `frames − t`.
  table: Vec<f32>,
  mel: Buffer,
  out: Buffer,
  read: Buffer,
  groups: Vec<BindGroup>,
}

impl Encoder {
  pub fn new(gpu: &Gpu, g: &Gguf, cfg: conformer::Config, ch: usize, joint: usize) -> Result<Self> {
    let d = cfg.d;
    let f32s = |name: String, dims: &[usize]| g.tensor(&name, dims)?.to_f32();
    let conv = |i: usize| {
      let p = format!("enc.pre_encode.conv.{i}");
      gpu.depthwise(
        &f32s(format!("{p}.weight"), &[3, 3, 1, ch])?,
        &f32s(format!("{p}.bias"), &[ch])?,
      )
    };
    let linear = |name: &str, dims: &[usize]| conformer::linear(gpu, g, &[name.into()], dims, true);
    Ok(Self {
      ch,
      conv0: conv(0)?,
      conv2: conv(2)?,
      conv3: linear("enc.pre_encode.conv.3", &[1, 1, ch, ch])?,
      conv5: conv(5)?,
      conv6: linear("enc.pre_encode.conv.6", &[1, 1, ch, ch])?,
      out: linear("enc.pre_encode.out", &[ch * half(half(half(MELS))), d])?,
      conformer: Conformer::load(gpu, g, cfg)?,
      joint: linear("joint.enc", &[d, joint])?,
      outputs: joint,
      arena: Mutex::new(None),
    })
  }

  /// Encodes `mel` `[mel frames][MELS]` and returns the joint's encoder projections
  /// `[frames][joint]`.
  pub fn run(&self, gpu: &Gpu, mel: &[f32]) -> Result<Vec<f32>> {
    let mel_frames = mel.len() / MELS;
    let (t, d, width) = (half(half(half(mel_frames))), self.conformer.cfg.d, self.joint.width());
    let mut arena = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    let a = match arena.take() {
      Some(a) if a.frames >= t => arena.insert(a),
      old => {
        drop(old);
        arena.insert(self.arena(gpu, t.next_multiple_of(64))?)
      }
    };
    gpu.write(&a.mel, mel);
    gpu.write(&a.bufs.pos, &a.table[(a.frames - t) * d..(a.frames + t - 1) * d]);
    let Arena { bufs, mel, out, read, groups, .. } = a;
    let out =
      gpu.run(groups, out, read, t * width, |p| self.record(p, bufs, mel, out, mel_frames))?;
    if width == self.outputs {
      return Ok(out);
    }
    Ok(out.chunks_exact(width).flat_map(|row| &row[..self.outputs]).copied().collect())
  }

  fn record(&self, p: &mut Pass, b: &Buffers, mel: &Buffer, out: &Buffer, mel_frames: usize) {
    let ch = self.ch;
    let (t2, f2) = (half(half(mel_frames)), half(half(MELS)));
    let (t, f) = (half(t2), half(f2));
    p.first_depthwise(mel, &self.conv0, &self.conv2, &b.h, mel_frames, MELS, ch);
    p.gemm(&b.h, &self.conv3, &b.qk, t2 * f2, Epilogue::Relu);
    p.depthwise(&b.qk, &self.conv5, &b.h, t2, f2, ch);
    p.gemm(&b.h, &self.conv6, &b.qk, t * f, Epilogue::Relu);
    p.flatten(&b.qk, &b.h, t, f, ch);
    p.gemm(&b.h, &self.out, &b.x, t, Epilogue::Bias);
    let x = self.conformer.record(p, b, t);
    p.gemm(x, &self.joint, out, t, Epilogue::Bias);
  }

  fn arena(&self, gpu: &Gpu, frames: usize) -> Result<Arena> {
    let c = &self.conformer.cfg;
    let (d, n) = (c.d, 2 * frames - 1);
    // In f32 as transcribe.cpp's `src/arch/parakeet/model.cpp` computes it: [2k] = sin(pos·d_k),
    // [2k + 1] = cos(pos·d_k).
    let div: Vec<f32> =
      (0..d / 2).map(|k| ((2 * k) as f32 * (-10000f32.ln() / d as f32)).exp()).collect();
    let table: Vec<f32> = (0..n)
      .flat_map(|r| {
        let pos = (frames as i64 - 1 - r as i64) as f32;
        div.iter().flat_map(move |dk| [(pos * dk).sin(), (pos * dk).cos()])
      })
      .collect();
    // The first two subsampling outputs: up to 2·frames rows of [MELS / 4][ch].
    let sub = 2 * frames * half(half(MELS)) * self.ch;
    let width = self.joint.width();
    let bufs = Buffers {
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      h: gpu.buffer((frames * c.d_ff).max(sub)),
      qk: gpu.buffer((frames * 4 * d).max(sub)),
      s: gpu.buffer(gpu.scores_len(frames, c.heads, c.head_dim())),
      yr: gpu.buffer(n * d),
      pos: gpu.buffer(n * d),
      ps: gpu.buffer(c.heads * frames * n),
    };
    Ok(Arena {
      frames,
      bufs,
      table,
      mel: gpu.buffer(8 * frames * MELS),
      out: gpu.buffer(frames * width),
      read: gpu.readback(frames * width),
      groups: Vec::new(),
    })
  }
}
