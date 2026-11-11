// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The network on the GPU in two calls per chunk, as parakeet.cpp's `src/diarization_encoder.cpp`
//! and `src/diarization_head.cpp`: `embed`, the chunk's mel frames stacked by 8 and projected;
//! `predict`, over the speaker cache, the FIFO and the chunk, a LayerNorm, the pre-LN transformer
//! blocks with rotary positions (GPT-NeoX pairs) and a final LayerNorm, then the head: a projection
//! to its width, a conv1d (kernel 3) to 8 frames of that width per frame, ReLU, a hidden layer with
//! ReLU and an output per speaker, through a sigmoid. Both run over one arena.

use std::sync::{Mutex, PoisonError};

use crate::conformer;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Groups, Linear, Pass};
use crate::parakeet::frontend::MELS;
use crate::transformer::{self, Block, Buffers, Config, norm, parallel};

/// Mel frames per embedding, and output frames per embedding.
pub const STACK: usize = 8;

pub struct Encoder {
  stack: Linear,
  embed_norm: Channels,
  blocks: transformer::Encoder,
  proj: Linear,
  conv: Linear,
  hidden: Linear,
  head: Linear,
  speakers: usize,
  arena: Mutex<Arena>,
}

/// Activations for up to `frames` embeddings, and the bind groups of each call.
struct Arena {
  bufs: Buffers,
  /// The rotary table from position 0: per position head_dim / 2 cosines, then sines.
  rope: Buffer,
  mel: Buffer,
  out: Buffer,
  read: Buffer,
  embed: Groups,
  predict: Groups,
}

impl Encoder {
  /// Loads the network of `cfg`, the head's `width` and `speakers`, for calls over up to `frames`
  /// embeddings, with rotary frequencies `theta^(−2i / head_dim)`.
  pub fn new(
    gpu: &Gpu,
    g: &Gguf,
    cfg: Config,
    [width, speakers, frames]: [usize; 3],
    theta: f32,
  ) -> Result<Self> {
    let (d, f) = (cfg.d, cfg.d_ff);
    let linear =
      |name: &str, dims: &[usize], bias| conformer::linear(gpu, g, &[name.into()], dims, bias);
    let blocks = parallel(cfg.layers, |i| {
      let p = |n: &str| format!("encoder.layers.{i}.{n}");
      Ok(Block {
        norm_attn: norm(gpu, g, &p("norm1"), d, true)?,
        qkv: linear(&p("attn.w_qkv"), &[d, 3 * d], false)?,
        out: linear(&p("attn.out_proj"), &[d, d], true)?,
        norm_ffn: norm(gpu, g, &p("norm2"), d, true)?,
        fc1: linear(&p("ffn.net.0"), &[d, f], true)?,
        fc2: linear(&p("ffn.net.3"), &[f, d], true)?,
      })
    })?;
    let m = |n: &str| format!("sortformer_modules.{n}");
    let conv_bias = g.tensor(&m("subpixel_upsample.bias"), &[STACK * width])?.to_f32()?;
    let conv = g.tensor(&m("subpixel_upsample.weight"), &[3, width, STACK * width])?;

    let hd = cfg.head_dim();
    let mut table = Vec::with_capacity(frames * hd);
    for pos in 0..frames {
      let angles = (0..hd / 2).map(|i| pos as f32 * theta.powf(-((2 * i) as f32) / hd as f32));
      table.extend(angles.clone().map(f32::cos).chain(angles.map(f32::sin)));
    }

    let bufs = Buffers {
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      // Also the conv's columns [t][3·width] and the hidden layer's output [8t][width].
      qkv: gpu.buffer(frames * (3 * d).max(STACK * width)),
      // Also the conv's output [t][8·width].
      h: gpu.buffer(frames * f.max(STACK * width)),
      s: gpu.buffer(gpu.scores_len(frames, cfg.heads, hd, false)),
    };
    let head = linear(&m("single_hidden_to_spks"), &[width, speakers], true)?;
    let out = frames * (STACK * head.width()).max(d);
    let arena = Arena {
      bufs,
      rope: gpu.upload(&table)?,
      mel: gpu.buffer(frames * STACK * MELS),
      out: gpu.buffer(out),
      read: gpu.readback(out),
      embed: Vec::new(),
      predict: Vec::new(),
    };

    Ok(Self {
      stack: linear("encoder.pre_encode.proj", &[STACK * MELS, d], false)?,
      embed_norm: norm(gpu, g, "encoder.embed_norm", d, true)?,
      blocks: transformer::Encoder {
        norm: norm(gpu, g, "encoder.final_norm", d, true)?,
        cfg,
        blocks,
      },
      proj: linear(&m("encoder_proj"), &[d, width], true)?,
      conv: gpu.conv(conv, &conv_bias)?,
      hidden: linear(&m("first_hidden_to_hidden"), &[width, width], true)?,
      head,
      speakers,
      arena: Mutex::new(arena),
    })
  }

  /// The embeddings `[⌈m / 8⌉][d]` of `m` mel frames `[m][MELS]`, the last 8 padded with zeros.
  pub fn embed(&self, gpu: &Gpu, mel: &[f32]) -> Result<Vec<f32>> {
    let t = (mel.len() / MELS).div_ceil(STACK);
    let mut padded = mel.to_vec();
    padded.resize(t * STACK * MELS, 0.0);
    let mut a = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    gpu.write(&a.mel, &padded);
    let Arena { mel, out, read, embed, .. } = &mut *a;
    let len = t * self.blocks.cfg.d;
    gpu.run(embed, out, read, len, |p| p.gemm(mel, &self.stack, out, t, Epilogue::Bias))
  }

  /// Per output frame, 8 per embedding of `x` `[t][d]`, each speaker's probability:
  /// `[8t][speakers]`.
  pub fn predict(&self, gpu: &Gpu, x: &[f32]) -> Result<Vec<f32>> {
    let t = x.len() / self.blocks.cfg.d;
    let width = self.head.width();
    let mut a = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    gpu.write(&a.bufs.y, x);
    let Arena { bufs, rope, out, read, predict, .. } = &mut *a;
    let logits =
      gpu.run(predict, out, read, STACK * t * width, |p| self.record(p, bufs, rope, out, t))?;
    let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
    Ok(
      logits
        .chunks(width)
        .flat_map(|row| row[..self.speakers].iter().map(|&z| sigmoid(z)))
        .collect(),
    )
  }

  fn record(&self, p: &mut Pass, b: &Buffers, rope: &Buffer, out: &Buffer, t: usize) {
    let (d, width) = (self.blocks.cfg.d, self.proj.width());
    p.layer_norm(&b.y, &self.embed_norm, &b.x, t, d, None);
    self.blocks.record(p, b, t, Some(rope));
    p.gemm(&b.y, &self.proj, &b.x, t, Epilogue::Bias);
    p.im2col(&b.x, &b.qkv, t, width, [3, 1, 1]);
    // Each frame's 8·width outputs are 8 frames of width.
    p.gemm(&b.qkv, &self.conv, &b.h, t, Epilogue::Bias);
    p.relu(&b.h, t * STACK * width);
    p.gemm(&b.h, &self.hidden, &b.qkv, STACK * t, Epilogue::Relu);
    p.gemm(&b.qkv, &self.head, out, STACK * t, Epilogue::Bias);
  }
}
