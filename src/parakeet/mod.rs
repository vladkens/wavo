// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Parakeet TDT: log-mel frontend on the CPU, FastConformer encoder (8× conv subsampling, relative
//! positions) on the GPU or the Neural Engine, TDT greedy decoding on the CPU, where the joint also
//! predicts how many frames to advance. Parakeet Ultra adds kestrel's length masking and symbol
//! budget; its voice activity head, which transcribe.cpp runs to split audio past 30 s, is not
//! used.

pub mod ane;
mod decoder;
mod encoder;
mod frontend;

use decoder::Decoder;
use encoder::Encoder;
use frontend::{F_MAX, Frontend, HOP, MELS, N_FFT, PRE_EMPHASIS, WINDOW};

use crate::ane::Backend;
use crate::conformer::{self, Attention};
use crate::error::{Error, Result, bail};
use crate::gguf::Gguf;
use crate::sentencepiece::Tokenizer;
use crate::{Token, Transcript};

/// Encoder frame length: 10 ms hop × subsampling 8.
const FRAME_MS: u32 = 80;

pub struct Parakeet {
  backend: Backend<Encoder>,
  frontend: Frontend,
  decoder: Decoder,
  tok: Tokenizer,
  /// Tokens left out of the result: unknown and control ones.
  special: Vec<bool>,
  /// kestrel length masking: the mel frames past `pcm.len() / HOP` are padding.
  kestrel: bool,
}

/// Checks the encoder's metadata and returns its config, subsampling channels and joint size.
fn config(g: &Gguf) -> Result<(conformer::Config, usize, usize)> {
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
  let ch = g.get::<u32>("stt.parakeet.encoder.subsampling_channels")? as usize;
  let joint = g.get::<u32>("stt.parakeet.joint.hidden")? as usize;
  Ok((cfg, ch, joint))
}

impl Parakeet {
  /// Runs the encoder on `ane` when given, else on the GPU.
  pub fn load(g: &Gguf, ane: Option<crate::ane::Model>) -> Result<Self> {
    let (cfg, ch, joint) = config(g)?;
    let kestrel = match g.optional::<&str>("stt.parakeet.encoder.length_masking")? {
      None | Some("none") => false,
      Some("kestrel") => true,
      Some(other) => bail!("stt.parakeet.encoder.length_masking is {other:?}"),
    };
    let vocab = g.array::<&str>("tokenizer.ggml.tokens")?;
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

    // The decoder loads on the CPU while the GPU starts and the encoder uploads.
    let (backend, decoder) = std::thread::scope(|s| {
      let decoder = s.spawn(|| Decoder::new(g, vocab.len(), blank));
      let backend = Backend::new(ane, |gpu| Encoder::new(gpu, g, cfg, ch, joint))?;
      Ok::<_, Error>((backend, decoder.join().unwrap()?))
    })?;
    let tok = Tokenizer::new(&vocab);
    Ok(Self { backend, frontend: Frontend::new(), decoder, tok, special, kestrel })
  }

  pub fn device(&self) -> &crate::Device {
    self.backend.device()
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let mel = self.frontend.compute(pcm);
    if mel.is_empty() {
      return Ok(Transcript::default());
    }
    // kestrel's valid length, pcm.len() / HOP, is all mel frames but the last.
    let valid = mel.len() / MELS - self.kestrel as usize;
    let (enc, width) = match &self.backend {
      Backend::Gpu(gpu) => gpu.1.run(&gpu.0, &mel, valid)?,
      Backend::Ane(model) => ane::run(model, &mel, valid)?,
    };
    let (ids, frames): (Vec<usize>, Vec<u32>) = (self.decoder.decode(&enc, width).into_iter())
      .filter(|&(id, _)| !self.special[id as usize])
      .map(|(id, frame)| (id as usize, frame))
      .unzip();
    let tokens: Vec<Token> = (ids.iter().zip(frames).zip(self.tok.pieces(&ids)))
      .map(|((&id, frame), piece)| Token { id: id as u32, piece, start_ms: frame * FRAME_MS })
      .collect();
    let text = tokens.iter().map(|t| t.piece.as_str()).collect::<String>().replace('▁', " ");
    // Runs of spaces collapse to one, and the ends are trimmed.
    let text = text.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
    Ok(Transcript { text, tokens })
  }
}
