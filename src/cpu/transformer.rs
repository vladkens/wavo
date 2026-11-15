// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The pre-LN transformer of `src/transformer.rs` (Whisper, Moonshine) on the CPU: the encoder
//! blocks and the decoder, one token per step, with its caches in f32 rounded to f16 as the GPU
//! keeps them.

use half::f16;

use super::gemm::{Linear, gemm, gemv};
use super::ops::{attend, attention, layer_norm, rope, silu};
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::Epilogue;
use crate::transformer::Config;

/// LayerNorm weights and biases.
pub type Norm = (Vec<f32>, Vec<f32>);

/// A LayerNorm `{name}.weight`, with `.bias` when `bias`, else zeros.
pub fn norm(g: &Gguf, name: &str, d: usize, bias: bool) -> Result<Norm> {
  let t = |p: &str| g.tensor(&format!("{name}.{p}"), &[d])?.to_f32();
  Ok((t("weight")?, if bias { t("bias")? } else { vec![0.0; d] }))
}

fn n(n: &Norm) -> (&[f32], &[f32]) {
  (&n.0, &n.1)
}

/// `{prefix}.{name}` for `names` stacked, `[d] → [d]` each; with `bias`, each has its bias but k
/// (zeros).
fn stacked(g: &Gguf, prefix: &str, names: &[&str], d: usize, bias: bool) -> Result<Linear> {
  let tensor = |n: &str, p: &str, dims: &[usize]| g.tensor(&format!("{prefix}.{n}.{p}"), dims);
  let w = names.iter().map(|n| tensor(n, "weight", &[d, d])).collect::<Result<Vec<_>>>()?;
  let mut b = Vec::new();
  for &n in names.iter().filter(|_| bias) {
    b.extend(if n == "k" { vec![0.0; d] } else { tensor(n, "bias", &[d])?.to_f32()? });
  }

  Linear::new(&w, b)
}

/// `name`, `dims` (ggml order), with its bias when `bias`.
pub fn linear(g: &Gguf, name: &str, dims: &[usize], bias: bool) -> Result<Linear> {
  let b = match bias {
    true => g.tensor(&format!("{name}.bias"), &dims[dims.len() - 1..])?.to_f32()?,
    false => Vec::new(),
  };
  Linear::new(&[g.tensor(&format!("{name}.weight"), dims)?], b)
}

/// `load(i)` for each of `n` layers on its own thread.
pub fn parallel<T: Send>(n: usize, load: impl Fn(usize) -> Result<T> + Sync) -> Result<Vec<T>> {
  std::thread::scope(|s| {
    let load = &load;
    let threads: Vec<_> = (0..n).map(|i| s.spawn(move || load(i))).collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
  })
}

pub struct Encoder {
  pub cfg: Config,
  pub blocks: Vec<Block>,
  /// The final LayerNorm.
  pub norm: Norm,
}

pub struct Block {
  pub norm_attn: Norm,
  pub qkv: Linear,
  pub out: Linear,
  pub norm_ffn: Norm,
  pub fc1: Linear,
  pub fc2: Linear,
}

impl Encoder {
  pub fn load(g: &Gguf, cfg: Config) -> Result<Self> {
    let (d, f, bias) = (cfg.d, cfg.d_ff, cfg.bias);
    let blocks = parallel(cfg.layers, |i| {
      let p = |n: &str| format!("enc.blocks.{i}.{n}");
      Ok(Block {
        norm_attn: norm(g, &p("norm_attn"), d, bias)?,
        qkv: stacked(g, &p("attn"), &["q", "k", "v"], d, bias)?,
        out: linear(g, &p("attn.out"), &[d, d], bias)?,
        norm_ffn: norm(g, &p("norm_ffn"), d, bias)?,
        fc1: linear(g, &p("ffn.fc1"), &[d, f], true)?,
        fc2: linear(g, &p("ffn.fc2"), &[f, d], true)?,
      })
    })?;
    Ok(Self { blocks, norm: norm(g, "enc.final_norm", d, bias)?, cfg })
  }

  /// The blocks over `x` `[t][d]`, then the final LayerNorm. With rotary dims, `rope` is the table
  /// of `ops::rope` from position 0.
  pub fn run(&self, mut x: Vec<f32>, t: usize, table: Option<&[f32]>) -> Vec<f32> {
    let c = &self.cfg;
    let (d, hd) = (c.d, c.head_dim());
    let mut y = vec![0.0; t * d];
    let mut qkv = vec![0.0; t * 3 * d];
    let mut h = vec![0.0; t * c.d_ff];
    for k in &self.blocks {
      layer_norm(&x, n(&k.norm_attn), &mut y, None);
      gemm(&y, &k.qkv, &mut qkv, t, Epilogue::Bias);
      if let Some(table) = table {
        rope(&mut qkv, table, [c.heads, hd, c.rot], c.neox);
      }

      attention(&qkv, 3 * d, &qkv[2 * d..], 3 * d, &mut y, t, c.heads);
      gemm(&y, &k.out, &mut x, t, Epilogue::Residual(1.0));
      layer_norm(&x, n(&k.norm_ffn), &mut y, None);
      gemm(&y, &k.fc1, &mut h, t, Epilogue::Gelu);
      gemm(&h, &k.fc2, &mut x, t, Epilogue::Residual(1.0));
    }

    layer_norm(&x, n(&self.norm), &mut y, None);
    y
  }
}

pub struct Decoder {
  pub cfg: Config,
  pub vocab: usize,
  /// Positions of the self-attention cache.
  pub context: usize,
  layers: Vec<Layer>,
  norm: Norm,
  /// The token embedding, also the output layer.
  embed: Linear,
  /// Learned positions `[context][d]`, empty with rotary dims.
  positions: Vec<f32>,
}

struct Layer {
  norm_self: Norm,
  qkv: Linear,
  self_out: Linear,
  norm_cross: Norm,
  cross_q: Linear,
  /// `[k | v]` of the encoder output.
  cross_kv: Linear,
  cross_out: Linear,
  norm_ffn: Norm,
  fc1: Linear,
  fc2: Linear,
}

/// The caches of a decoding, per layer K then V, each `[heads][positions][head_dim]`.
pub struct Cache {
  own: Vec<Vec<f32>>,
  cross: Vec<Vec<f32>>,
  frames: usize,
}

impl Decoder {
  pub fn load(g: &Gguf, cfg: Config, vocab: usize, context: usize) -> Result<Self> {
    let (d, f, bias) = (cfg.d, cfg.d_ff, cfg.bias);
    let up = if cfg.gated { 2 * f } else { f };
    let layers = |i: usize| {
      let p = |n: &str| format!("dec.blocks.{i}.{n}");
      Ok(Layer {
        norm_self: norm(g, &p("norm_self"), d, bias)?,
        qkv: stacked(g, &p("self_attn"), &["q", "k", "v"], d, bias)?,
        self_out: linear(g, &p("self_attn.out"), &[d, d], bias)?,
        norm_cross: norm(g, &p("norm_cross"), d, bias)?,
        cross_q: linear(g, &p("cross_attn.q"), &[d, d], bias)?,
        cross_kv: stacked(g, &p("cross_attn"), &["k", "v"], d, bias)?,
        cross_out: linear(g, &p("cross_attn.out"), &[d, d], bias)?,
        norm_ffn: norm(g, &p("norm_ffn"), d, bias)?,
        fc1: linear(g, &p("ffn.fc1"), &[d, up], true)?,
        fc2: linear(g, &p("ffn.fc2"), &[f, d], true)?,
      })
    };
    let (layers, embed) = std::thread::scope(|s| {
      let layers = s.spawn(|| parallel(cfg.layers, layers));
      let embed = linear(g, "dec.token_embd", &[d, vocab], false);
      Ok::<_, crate::Error>((layers.join().unwrap()?, embed?))
    })?;
    let positions = match cfg.rot {
      0 => g.tensor("dec.pos_emb.weight", &[d, context])?.to_f32()?,
      _ => Vec::new(),
    };
    Ok(Self {
      layers,
      norm: norm(g, "dec.final_norm", d, bias)?,
      embed,
      positions,
      vocab,
      context,
      cfg,
    })
  }

  /// The caches for a decoding over the encoder output `enc` (`t` rows): every layer's
  /// cross-attention K and V, and an empty self-attention cache.
  pub fn cache(&self, enc: &[f32], t: usize) -> Cache {
    let d = self.cfg.d;
    let mut kv = vec![0.0; t * 2 * d];
    let cross = (self.layers.iter())
      .map(|l| {
        gemm(enc, &l.cross_kv, &mut kv, t, Epilogue::Bias);
        let mut cache = vec![0.0; t * 2 * d];
        self.pack(&kv, &mut cache, t, 0);
        cache
      })
      .collect();
    let own = (0..self.cfg.layers).map(|_| vec![0.0; self.context * 2 * d]).collect();
    Cache { own, cross, frames: t }
  }

  /// Rows `[k | v]` of `kv` into `cache` (of `cols` positions) at positions `col..`, rounded to
  /// f16.
  fn pack(&self, kv: &[f32], cache: &mut [f32], cols: usize, col: usize) {
    let (d, hd) = (self.cfg.d, self.cfg.head_dim());
    for (j, row) in kv.chunks_exact(2 * d).enumerate() {
      for (h, x) in row.chunks_exact(hd).enumerate() {
        // K's heads, then V's.
        let at = (h * cols + col + j) * hd;
        for (c, x) in cache[at..][..hd].iter_mut().zip(x) {
          *c = f16::from_f32(*x).to_f32();
        }
      }
    }
  }

  /// The decoder on `token` at position `pos`; returns the logits. With rotary dims, `table` is
  /// the table of `ops::rope` from position 0.
  pub fn step(&self, c: &mut Cache, token: usize, pos: usize, table: Option<&[f32]>) -> Vec<f32> {
    let cfg = &self.cfg;
    let (d, heads, hd, f) = (cfg.d, cfg.heads, cfg.head_dim(), cfg.d_ff);
    let mut x = self.embed.row(token);
    if cfg.rot == 0 {
      x.iter_mut().zip(&self.positions[pos * d..]).for_each(|(x, p)| *x += p);
    }

    let (mut y, mut qkv, mut att) = (vec![0.0; d], vec![0.0; 3 * d], vec![0.0; d]);
    let mut h = vec![0.0; if cfg.gated { 2 * f } else { f }];
    for (l, (own, cross)) in self.layers.iter().zip(c.own.iter_mut().zip(&c.cross)) {
      layer_norm(&x, n(&l.norm_self), &mut y, None);
      gemv(&y, &l.qkv, &mut qkv, Epilogue::Bias);
      if let Some(table) = table {
        rope(&mut qkv, &table[pos * cfg.rot..], [heads, hd, cfg.rot], cfg.neox);
      }

      self.pack(&qkv[d..], own, self.context, pos);
      attend(&qkv[..d], own, self.context, pos + 1, &mut att, heads);
      gemv(&att, &l.self_out, &mut x, Epilogue::Residual(1.0));
      layer_norm(&x, n(&l.norm_cross), &mut y, None);
      gemv(&y, &l.cross_q, &mut qkv[..d], Epilogue::Bias);
      attend(&qkv[..d], cross, c.frames, c.frames, &mut att, heads);
      gemv(&att, &l.cross_out, &mut x, Epilogue::Residual(1.0));
      layer_norm(&x, n(&l.norm_ffn), &mut y, None);
      gemv(&y, &l.fc1, &mut h, if cfg.gated { Epilogue::Bias } else { Epilogue::Gelu });
      if cfg.gated {
        // SwiGLU: [x | gate] to SiLU(gate) · x.
        let (x, gate) = h.split_at_mut(f);
        x.iter_mut().zip(&*gate).for_each(|(x, g)| *x *= silu(*g));
      }

      gemv(&h[..f], &l.fc2, &mut x, Epilogue::Residual(1.0));
    }

    layer_norm(&x, n(&self.norm), &mut y, None);
    let mut logits = vec![0.0; self.vocab];
    gemv(&y, &self.embed, &mut logits, Epilogue::Bias);
    logits
  }
}
