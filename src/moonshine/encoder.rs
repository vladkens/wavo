// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Encoder on the GPU over raw samples (`src/arch/moonshine/encoder.cpp`): a conv1d of kernel 127
//! and stride 64 without bias, tanh and a GroupNorm with one group over all values; conv1d kernel
//! 7 stride 3 and GELU; conv1d kernel 3 stride 2 and GELU, all without padding; then the shared
//! pre-LN blocks with partial RoPE on q and k, and the final LayerNorm.

use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Linear, Pass};
use crate::transformer::{self, Encoder as Blocks};

/// The conv stem's kernels and strides.
pub const STEM: [(usize, usize); 3] = [(127, 64), (7, 3), (3, 2)];

pub struct Encoder {
  conv0: Linear,
  norm: Channels,
  eps: f32,
  conv1: Linear,
  conv2: Linear,
  pub blocks: Blocks,
}

/// Activations for up to `samples` samples.
pub struct Buffers {
  pub samples: usize,
  pub pcm: Buffer,
  /// im2col columns of each conv.
  cols: Buffer,
  /// conv0's output `[t1][d]`, then conv1's `[t2][2·d]`.
  conv: Buffer,
  pub t: transformer::Buffers,
}

/// Rows after each conv of the stem for `n` samples; 0 when too short for one.
pub fn frames(n: usize) -> [usize; 3] {
  let mut t = n;
  STEM.map(|(kernel, stride)| {
    t = if t < kernel { 0 } else { (t - kernel) / stride + 1 };
    t
  })
}

impl Encoder {
  pub fn load(gpu: &Gpu, g: &Gguf, blocks: transformer::Config, eps: f32) -> Result<Self> {
    let d = blocks.d;
    let f32s = |name: &str, n: usize| g.tensor(name, &[n])?.to_f32();
    // conv0 has no bias.
    let conv = |i: usize, ch: usize, out: usize| {
      let bias = if i == 0 { Vec::new() } else { f32s(&format!("enc.conv.{i}.bias"), out)? };
      gpu.conv(g.tensor(&format!("enc.conv.{i}.weight"), &[STEM[i].0, ch, out])?, &bias)
    };
    Ok(Self {
      conv0: conv(0, 1, d)?,
      norm: gpu.channels(&f32s("enc.conv.norm.weight", d)?, &f32s("enc.conv.norm.bias", d)?)?,
      eps,
      conv1: conv(1, d, 2 * d)?,
      conv2: conv(2, 2 * d, d)?,
      blocks: Blocks::load(gpu, g, blocks)?,
    })
  }

  pub fn buffers(&self, gpu: &Gpu, samples: usize) -> Buffers {
    let b = &self.blocks.cfg;
    let [t1, t2, t] = frames(samples);
    let cols = [t1 * 128, t2 * (STEM[1].0 * b.d).next_multiple_of(32), t * 6 * b.d];
    Buffers {
      samples,
      pcm: gpu.buffer(samples),
      cols: gpu.buffer(cols.into_iter().max().unwrap()),
      conv: gpu.buffer((t1 * b.d).max(t2 * 2 * b.d)),
      t: transformer::Buffers {
        x: gpu.buffer(t * b.d),
        y: gpu.buffer(t * b.d),
        qkv: gpu.buffer(t * 3 * b.d),
        // Also the decoder's cross K/V, t × 2·d.
        h: gpu.buffer(t * b.d_ff.max(2 * b.d)),
        s: gpu.buffer(gpu.scores_len(t, b.heads, b.head_dim(), false)),
      },
    }
  }

  /// Records the encoder over the first `n` samples of `b.pcm`; the output, `frames(n)[2]` rows,
  /// is in `b.t.y`. `rope` is the table of `Pass::rope` from position 0.
  pub fn record(&self, p: &mut Pass, b: &Buffers, n: usize, rope: &Buffer) {
    let d = self.blocks.cfg.d;
    let [t1, t2, t] = frames(n);
    p.im2col(&b.pcm, &b.cols, n, 1, [STEM[0].0, STEM[0].1, 0]);
    p.gemm(&b.cols, &self.conv0, &b.conv, t1, Epilogue::Tanh);
    p.group_norm(&b.conv, &self.norm, t1, d, self.eps);
    p.im2col(&b.conv, &b.cols, t1, d, [STEM[1].0, STEM[1].1, 0]);
    p.gemm(&b.cols, &self.conv1, &b.conv, t2, Epilogue::Gelu);
    p.im2col(&b.conv, &b.cols, t2, 2 * d, [STEM[2].0, STEM[2].1, 0]);
    p.gemm(&b.cols, &self.conv2, &b.t.x, t, Epilogue::Gelu);
    self.blocks.record(p, &b.t, t, Some(rope));
  }
}
