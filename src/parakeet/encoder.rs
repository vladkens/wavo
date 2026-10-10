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
use crate::gpu::{BindGroup, Buffer, Channels, Epilogue, Gpu, Linear, Pass, half, pos_rows};

/// conv0 to conv5 run in this many chunks of output rows, so their outputs (4× the blocks'
/// activations) never exist whole.
const CHUNKS: usize = 8;

pub struct Encoder {
  /// Subsampling channels.
  ch: usize,
  conv0: Channels,
  conv2: Channels,
  conv3: Linear,
  conv5: Channels,
  conv6: Linear,
  out: Linear,
  conformer: Conformer,
  joint: Linear,
  arena: Mutex<Option<Arena>>,
}

/// Activations for up to `frames` encoder frames, and the bind groups recorded over them. The
/// block buffers also hold the subsampling: `yr` a chunk of conv2's output, `qk` of conv3's and
/// conv6's output, `h` conv5's and the flattened output.
struct Arena {
  frames: usize,
  bufs: Buffers,
  /// Sinusoidal positions for `frames`: `pos_rows(frames)` rows of d, row r is position
  /// `frames + 15 − r`. A call over t frames takes its `pos_rows(t)` rows from row `frames − t`.
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
    // Weights from [ch][time tap][frequency tap] to [tap][ch].
    let conv = |i: usize| {
      let p = format!("enc.pre_encode.conv.{i}");
      let w = f32s(format!("{p}.weight"), &[3, 3, 1, ch])?;
      let w: Vec<f32> = (0..9 * ch).map(|e| w[e % ch * 9 + e / ch]).collect();
      gpu.channels(&w, &f32s(format!("{p}.bias"), &[ch])?)
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
      arena: Mutex::new(None),
    })
  }

  /// Encodes `mel` `[mel frames][MELS]` and returns the joint's encoder projections in rows of
  /// the returned width (the joint's size padded to a multiple of 64).
  pub fn run(&self, gpu: &Gpu, mel: &[f32]) -> Result<(Vec<f32>, usize)> {
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
    gpu.write(&a.bufs.pos, &a.table[(a.frames - t) * d..][..pos_rows(t) * d]);
    let Arena { bufs, mel, out, read, groups, .. } = a;
    let out =
      gpu.run(groups, out, read, t * width, |p| self.record(p, bufs, mel, out, mel_frames))?;
    Ok((out, width))
  }

  fn record(&self, p: &mut Pass, b: &Buffers, mel: &Buffer, out: &Buffer, mel_frames: usize) {
    let ch = self.ch;
    let (t2, f2) = (half(half(mel_frames)), half(half(MELS)));
    let (t, f) = (half(t2), half(f2));
    // Always CHUNKS chunks, some maybe empty, so the dispatches match the cached bind groups.
    let c = t.div_ceil(CHUNKS);
    for t0 in (0..CHUNKS).map(|i| i * c) {
      let n = c.min(t.saturating_sub(t0));
      let rows = if n == 0 { 0 } else { 2 * n + 1 };
      p.depthwise(mel, Some(&self.conv0), &self.conv2, &b.yr, [mel_frames, MELS, ch], t0, rows);
      p.gemm(&b.yr, &self.conv3, &b.qk, rows * f2, Epilogue::Relu);
      p.depthwise(&b.qk, None, &self.conv5, &b.h, [t2, f2, ch], t0, n);
    }
    p.gemm(&b.h, &self.conv6, &b.qk, t * f, Epilogue::Relu);
    p.flatten(&b.qk, &b.h, t, f, ch);
    p.gemm(&b.h, &self.out, &b.x, t, Epilogue::Bias);
    let x = self.conformer.record(p, b, t);
    p.gemm(x, &self.joint, out, t, Epilogue::Bias);
  }

  fn arena(&self, gpu: &Gpu, frames: usize) -> Result<Arena> {
    let c = &self.conformer.cfg;
    let (d, n) = (c.d, pos_rows(frames));
    // In f32 as transcribe.cpp's `src/arch/parakeet/model.cpp` computes it: [2k] = sin(pos·d_k),
    // [2k + 1] = cos(pos·d_k).
    let div: Vec<f32> =
      (0..d / 2).map(|k| ((2 * k) as f32 * (-10000f32.ln() / d as f32)).exp()).collect();
    let table: Vec<f32> = (0..n)
      .flat_map(|r| {
        let pos = (frames as i64 + 15 - r as i64) as f32;
        div.iter().flat_map(move |dk| [(pos * dk).sin(), (pos * dk).cos()])
      })
      .collect();
    // A subsampling chunk of the first two outputs: 2·frames / CHUNKS + 1 rows of
    // [MELS / 4][ch]; the third output [frames][MELS / 8][ch].
    let chunk = (2 * frames / CHUNKS + 1) * half(half(MELS)) * self.ch;
    let third = frames * half(half(half(MELS))) * self.ch;
    let width = self.joint.width();
    let bufs = Buffers {
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      h: gpu.buffer((frames * c.d_ff).max(third)),
      qk: gpu.buffer((frames * 4 * d).max(chunk).max(third)),
      s: gpu.buffer(gpu.scores_len(frames, c.heads, c.head_dim(), true)),
      yr: gpu.buffer(((n + 64) * d).max(chunk)),
      pos: gpu.buffer(n * d),
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
