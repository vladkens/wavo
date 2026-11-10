// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The network on the GPU in two calls per chunk: `embed`, Parakeet's 8× subsampling of the chunk's
//! mel window; `predict`, over the speaker cache, the FIFO and the chunk scaled by √d, the
//! FastConformer blocks, the projection to the transformer's width, its post-LN layers
//! `x = LN(x + attn(x))`, `x = LN(x + ffn(x))` (ReLU, no positions), then ReLU, fc1 + ReLU and the
//! speaker head, whose sigmoid the caller takes. Both run over one preallocated arena.

use std::sync::{Mutex, PoisonError};

use crate::conformer::{self, Buffers, Conformer};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Channels, Epilogue, Gpu, Groups, Linear, Pass, half, pos_rows};
use crate::parakeet::encoder::Subsampling;
use crate::parakeet::frontend::MELS;
use crate::transformer::{Config, norm, parallel};

pub struct Encoder {
  subsampling: Subsampling,
  conformer: Conformer,
  proj: Linear,
  cfg: Config,
  blocks: Vec<Block>,
  fc1: Linear,
  head: Linear,
  arena: Mutex<Option<Arena>>,
}

struct Block {
  qkv: Linear,
  out: Linear,
  norm1: Channels,
  ff_in: Linear,
  ff_out: Linear,
  norm2: Channels,
}

/// Activations for up to `frames` frames, and the bind groups of each call.
struct Arena {
  frames: usize,
  bufs: Buffers,
  /// Sinusoidal positions: `pos_rows(frames)` rows of d, row r is position `frames + 15 − r`.
  table: Vec<f32>,
  mel: Buffer,
  out: Buffer,
  read: Buffer,
  embed: Groups,
  predict: Groups,
}

impl Encoder {
  /// Loads the network with `ch` subsampling channels, the transformer `cfg` (post-LN, biases, no
  /// positions) and a head of `speakers`.
  pub fn new(
    gpu: &Gpu,
    g: &Gguf,
    enc: conformer::Config,
    ch: usize,
    cfg: Config,
    speakers: usize,
  ) -> Result<Self> {
    let (d, f, ed) = (cfg.d, cfg.d_ff, enc.d);
    let linear = |name: &str, dims: &[usize]| conformer::linear(gpu, g, &[name.into()], dims, true);
    let (blocks, subsampling, conformer) = std::thread::scope(|s| {
      let subsampling = s.spawn(|| Subsampling::new(gpu, g, ch, ed));
      let blocks = s.spawn(|| {
        parallel(cfg.layers, |i| {
          let p = |n: &str| format!("tf.blocks.{i}.{n}");
          let qkv = ["q", "k", "v"].map(|n| p(&format!("attn.{n}")));
          Ok(Block {
            qkv: conformer::linear(gpu, g, &qkv, &[d, d], true)?,
            out: linear(&p("attn.out"), &[d, d])?,
            norm1: norm(gpu, g, &p("norm_1"), d, true)?,
            ff_in: linear(&p("ff.in"), &[d, f])?,
            ff_out: linear(&p("ff.out"), &[f, d])?,
            norm2: norm(gpu, g, &p("norm_2"), d, true)?,
          })
        })
      });
      let conformer = Conformer::load(gpu, g, enc)?;
      Ok::<_, crate::Error>((blocks.join().unwrap()?, subsampling.join().unwrap()?, conformer))
    })?;

    Ok(Self {
      subsampling,
      proj: linear("diar.encoder_proj", &[ed, d])?,
      blocks,
      fc1: linear("diar.fc1", &[d, d])?,
      head: linear("diar.single_spk_head", &[d, speakers])?,
      conformer,
      cfg,
      arena: Mutex::new(None),
    })
  }

  /// The conformer's width: the embeddings' size.
  pub fn width(&self) -> usize {
    self.conformer.cfg.d
  }

  /// Allocates the arena for calls over up to `frames` frames, of embeddings or of mel frames / 8.
  pub fn reserve(&self, gpu: &Gpu, frames: usize) {
    self.with(gpu, frames, |_| ());
  }

  /// Runs `f` with the arena for up to `frames` frames.
  fn with<T>(&self, gpu: &Gpu, frames: usize, f: impl FnOnce(&mut Arena) -> T) -> T {
    let mut arena = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    let a = match arena.take() {
      Some(a) if a.frames >= frames => arena.insert(a),
      old => {
        drop(old);
        arena.insert(self.arena(gpu, frames.next_multiple_of(64)))
      }
    };
    f(a)
  }

  /// The embeddings `[half³(m)][d]` of a mel window `[m][MELS]`, unscaled.
  pub fn embed(&self, gpu: &Gpu, mel: &[f32]) -> Result<Vec<f32>> {
    let m = mel.len() / MELS;
    let t = half(half(half(m)));
    self.with(gpu, m.div_ceil(8), |a| {
      gpu.write(&a.mel, mel);
      let Arena { bufs, mel, read, embed, .. } = a;
      let x = &bufs.x;
      gpu
        .run(embed, x, read, t * self.width(), |p| self.subsampling.record(p, bufs, mel, [m, m], t))
    })
  }

  /// The speaker head's outputs before the sigmoid, `[t][speakers]`, for the embeddings `x`
  /// `[t][d]`, already scaled by √d.
  pub fn predict(&self, gpu: &Gpu, x: &[f32], speakers: usize) -> Result<Vec<f32>> {
    let (d, t, width) = (self.width(), x.len() / self.width(), self.head.width());
    let out = self.with(gpu, t, |a| {
      gpu.write(&a.bufs.x, x);
      gpu.write(&a.bufs.pos, &a.table[(a.frames - t) * d..][..pos_rows(t) * d]);
      let Arena { bufs, out, read, predict, .. } = a;
      gpu.run(predict, out, read, t * width, |p| self.record(p, bufs, out, t))
    })?;
    Ok(out.chunks(width).flat_map(|row| &row[..speakers]).copied().collect())
  }

  fn record(&self, p: &mut Pass, b: &Buffers, out: &Buffer, t: usize) {
    let (d, heads) = (self.cfg.d, self.cfg.heads);
    let enc = self.conformer.record(p, b, t);
    let (x, y) = if std::ptr::eq(enc, &b.x) { (&b.y, &b.x) } else { (&b.x, &b.y) };
    p.gemm(enc, &self.proj, x, t, Epilogue::Bias);
    p.layers(&self.blocks, |p, k| {
      p.gemm(x, &k.qkv, &b.qk, t, Epilogue::Bias);
      p.attention(&b.qk, None, None, &b.s, y, t, heads, self.cfg.head_dim());
      p.gemm(y, &k.out, x, t, Epilogue::Residual(1.0));
      p.layer_norm(x, &k.norm1, y, t, d, None);
      p.gemm(y, &k.ff_in, &b.h, t, Epilogue::Relu);
      p.gemm(&b.h, &k.ff_out, y, t, Epilogue::Residual(1.0));
      p.layer_norm(y, &k.norm2, x, t, d, None);
      x
    });
    p.relu(x, t * d);
    p.gemm(x, &self.fc1, y, t, Epilogue::Relu);
    p.gemm(y, &self.head, out, t, Epilogue::Bias);
  }

  fn arena(&self, gpu: &Gpu, frames: usize) -> Arena {
    let (c, tf) = (&self.conformer.cfg, &self.cfg);
    let (d, n) = (c.d, pos_rows(frames));
    let [chunk, third] = self.subsampling.scratch(frames);
    let scores = gpu.scores_len(frames, c.heads, c.head_dim(), true).max(gpu.scores_len(
      frames,
      tf.heads,
      tf.head_dim(),
      false,
    ));
    let bufs = Buffers {
      x: gpu.buffer(frames * d),
      y: gpu.buffer(frames * d),
      h: gpu.buffer((frames * c.d_ff.max(tf.d_ff)).max(third)),
      qk: gpu.buffer((frames * 4 * d).max(chunk).max(third)),
      s: gpu.buffer(scores),
      yr: gpu.buffer(((n + 64) * d).max(chunk)),
      pos: gpu.buffer(n * d),
    };
    let width = self.head.width().max(d);
    Arena {
      frames,
      bufs,
      table: conformer::rel_positions(d, frames as i64 + 15, n),
      mel: gpu.buffer(8 * frames * MELS),
      out: gpu.buffer(frames * self.head.width()),
      read: gpu.readback(frames * width),
      embed: Vec::new(),
      predict: Vec::new(),
    }
  }
}
