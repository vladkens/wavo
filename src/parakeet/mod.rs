// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Parakeet TDT: log-mel frontend on the CPU, FastConformer encoder (8× conv subsampling, relative
//! positions) on the GPU, TDT greedy decoding on the CPU, where the joint also predicts how many
//! frames to advance.

mod decoder;
mod encoder;
mod frontend;

use decoder::Decoder;
use encoder::Encoder;
use frontend::{F_MAX, Frontend, HOP, MELS, N_FFT, PRE_EMPHASIS, WINDOW};

use crate::conformer::{self, Attention};
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::{Token, Transcript};

/// Encoder frame length: 10 ms hop × subsampling 8.
const FRAME_MS: u32 = 80;

pub struct Parakeet {
  gpu: Gpu,
  frontend: Frontend,
  encoder: Encoder,
  decoder: Decoder,
  vocab: Vec<String>,
  /// Tokens left out of the result: unknown and control ones.
  special: Vec<bool>,
}

impl Parakeet {
  pub fn load(g: &Gguf) -> Result<Self> {
    for (key, want) in [
      ("general.architecture", "parakeet"),
      ("stt.parakeet.head_kind", "tdt"),
      ("stt.parakeet.encoder.att_context_style", "regular"),
      ("tokenizer.ggml.model", "bpe"),
      ("stt.frontend.type", "mel"),
      ("stt.frontend.window", "hann"),
      ("stt.frontend.normalize", "per_feature"),
    ] {
      g.check(key, &[want])?;
    }
    for (key, want) in [
      ("stt.parakeet.encoder.subsampling_factor", 8),
      ("stt.frontend.sample_rate", 16000),
      ("stt.frontend.num_mels", MELS),
      ("stt.frontend.n_fft", N_FFT),
      ("stt.frontend.win_length", WINDOW),
      ("stt.frontend.hop_length", HOP),
    ] {
      g.check(key, &[want as u32])?;
    }
    for (key, want) in [("pre_emphasis", PRE_EMPHASIS), ("f_min", 0.0), ("f_max", F_MAX)] {
      g.check(&format!("stt.frontend.{key}"), &[want])?;
    }
    g.check("stt.parakeet.encoder.use_bias", &[false])?;
    g.check("stt.parakeet.encoder.xscaling", &[false])?;
    let cfg = conformer::Config::load(g, "parakeet", Attention::Relative, false)?;
    let pad = (cfg.kernel / 2) as i32;
    for (key, want) in [
      ("att_context_left", -1),
      ("att_context_right", -1),
      ("conv_context_left", pad),
      ("conv_context_right", pad),
    ] {
      g.check(&format!("stt.parakeet.encoder.{key}"), &[want])?;
    }

    let vocab: Vec<String> =
      g.array::<&str>("tokenizer.ggml.tokens")?.into_iter().map(String::from).collect();
    let types = g.array::<i32>("tokenizer.ggml.token_type")?;
    let unk = g.get::<u32>("tokenizer.ggml.unknown_token_id")? as usize;
    let blank = g.get::<u32>("tokenizer.ggml.blank_token_id")? as usize;
    if types.len() != vocab.len() || blank >= vocab.len() {
      bail!("vocabulary of {} tokens, {} token types, blank {blank}", vocab.len(), types.len());
    }
    // SentencePiece token types 2 and 3: unknown and control.
    let special = (0..vocab.len())
      .map(|i| matches!(types[i], 2 | 3) || i == unk || vocab[i] == "<unk>")
      .collect();

    let gpu = Gpu::new(true)?;
    let ch = g.get::<u32>("stt.parakeet.encoder.subsampling_channels")? as usize;
    let joint = g.get::<u32>("stt.parakeet.joint.hidden")? as usize;
    let encoder = Encoder::new(&gpu, g, cfg, ch, joint)?;
    let decoder = Decoder::new(g, vocab.len(), blank)?;
    // The GPU's first use of the pipelines and the weight memory: pay it here on 8 silent mel
    // frames instead of in the first call.
    encoder.run(&gpu, &vec![0.0; 8 * MELS])?;
    Ok(Self { gpu, frontend: Frontend::new(), encoder, decoder, vocab, special })
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let mel = self.frontend.compute(pcm);
    if mel.is_empty() {
      return Ok(Transcript::default());
    }
    let (enc, width) = self.encoder.run(&self.gpu, &mel)?;
    let tokens: Vec<Token> = (self.decoder.decode(&enc, width).into_iter())
      .filter(|&(id, _)| !self.special[id as usize])
      .map(|(id, frame)| Token {
        id,
        piece: self.vocab[id as usize].clone(),
        start_ms: frame * FRAME_MS,
      })
      .collect();
    let text = tokens.iter().map(|t| t.piece.as_str()).collect::<String>().replace('▁', " ");
    // Runs of spaces collapse to one, and the ends are trimmed.
    let text = text.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
    Ok(Transcript { text, tokens })
  }
}
