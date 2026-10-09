// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GigaAM v3 with the RNN-T head (e2e-rnnt): log-mel frontend on the CPU, Conformer encoder with
//! rotary attention on the GPU, greedy RNN-T decoding on the CPU.

mod decoder;
mod encoder;
mod frontend;

use decoder::Decoder;
use encoder::Encoder;
use frontend::Frontend;

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::{Token, Transcript};

struct Config {
  layers: usize,
  d: usize,
  heads: usize,
  d_ff: usize,
  mels: usize,
  joint: usize,
  /// RoPE base: GigaAM passes `pos_emb_max_len` (5000) as theta.
  theta: f64,
}

pub struct Gigaam {
  gpu: Gpu,
  frontend: Frontend,
  encoder: Encoder,
  decoder: Decoder,
  vocab: Vec<String>,
}

impl Gigaam {
  pub fn load(g: &Gguf) -> Result<Self> {
    for (key, want) in [
      ("general.architecture", "gigaam"),
      ("stt.gigaam.head_kind", "rnnt"),
      ("stt.gigaam.encoder.self_attention_model", "rotary"),
      ("stt.gigaam.encoder.conv_norm_type", "layer_norm"),
      ("stt.gigaam.joint.activation", "relu"),
    ] {
      let got = g.str(key)?;
      if got != want {
        bail!("{key} is {got:?}, only {want:?} is supported");
      }
    }
    for (key, want) in [
      ("stt.gigaam.encoder.conv_kernel", 5),
      ("stt.gigaam.encoder.subs_kernel_size", 5),
      ("stt.gigaam.encoder.subsampling_factor", 4),
      ("stt.gigaam.predictor.n_layers", 1),
      ("stt.frontend.sample_rate", 16000),
      ("stt.frontend.n_fft", frontend::N_FFT as u32),
      ("stt.frontend.win_length", frontend::N_FFT as u32),
      ("stt.frontend.hop_length", 160),
    ] {
      let got = g.u32(key)?;
      if got != want {
        bail!("{key} is {got}, only {want} is supported");
      }
    }
    let u = |key: &str| g.u32(&format!("stt.gigaam.{key}")).map(|v| v as usize);
    let cfg = Config {
      layers: u("encoder.n_layers")?,
      d: u("encoder.d_model")?,
      heads: u("encoder.n_heads")?,
      d_ff: u("encoder.d_ff")?,
      mels: g.u32("stt.frontend.num_mels")? as usize,
      joint: u("joint.hidden")?,
      theta: u("encoder.pos_emb_max_len")? as f64,
    };
    let head_dim = cfg.d / cfg.heads;
    if !cfg.d.is_multiple_of(cfg.heads)
      || cfg.d > 1024
      || head_dim > 128
      || !head_dim.is_multiple_of(2)
    {
      bail!("unsupported attention shape: d_model {}, {} heads", cfg.d, cfg.heads);
    }
    let vocab: Vec<String> =
      g.strs("tokenizer.ggml.tokens")?.into_iter().map(String::from).collect();
    let classes = u("joint.num_classes")?;
    let blank = g.u32("tokenizer.ggml.blank_token_id")? as usize;
    if vocab.len() != classes || u("predictor.vocab")? != classes || blank >= classes {
      bail!("vocabulary of {} tokens, {classes} classes, blank {blank}", vocab.len());
    }

    let gpu = Gpu::new()?;
    Ok(Self {
      frontend: Frontend::new(g, cfg.mels)?,
      decoder: Decoder::new(g, u("predictor.hidden")?, cfg.joint, classes, blank)?,
      encoder: Encoder::new(&gpu, g, cfg)?,
      gpu,
      vocab,
    })
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let mel = self.frontend.compute(pcm);
    if mel.is_empty() {
      return Ok(Transcript::default());
    }
    let enc = self.encoder.run(&self.gpu, &mel)?;
    let tokens: Vec<Token> = (self.decoder.decode(&enc).into_iter())
      .map(|(id, frame)| Token { id, piece: self.vocab[id as usize].clone(), frame })
      .collect();
    let text = tokens.iter().map(|t| t.piece.as_str()).collect::<String>().replace('▁', " ");
    let text = text.strip_prefix(' ').unwrap_or(&text).to_string();
    Ok(Transcript { text, tokens })
  }
}

/// Dot product with 16 independent accumulators (four FMA vectors in flight), summed pairwise so
/// the final reduction also vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
  let ((a16, a_tail), (b16, b_tail)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
  let mut acc = [0f32; 16];
  for (x, y) in a16.iter().zip(b16) {
    for ((s, x), y) in acc.iter_mut().zip(x).zip(y) {
      *s = x.mul_add(*y, *s);
    }
  }
  for n in [8, 4, 2, 1] {
    let (lo, hi) = acc.split_at_mut(n);
    for (l, h) in lo.iter_mut().zip(&*hi) {
      *l += h;
    }
  }
  acc[0] + a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum::<f32>()
}
