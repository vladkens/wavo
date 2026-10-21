// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GigaAM v3: log-mel frontend on the CPU, Conformer encoder with rotary attention on the GPU or
//! the Neural Engine, and one of two heads: RNN-T (greedy decoding on the CPU) or CTC (logits as
//! the encoder's last layer, greedy collapse on the CPU). The e2e variants use SentencePiece
//! pieces, the others characters.

pub mod ane;
mod decoder;
mod encoder;
mod frontend;

use decoder::Decoder;
use encoder::Encoder;
use frontend::Frontend;

use crate::conformer::{self, Attention};
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::sentencepiece::{self, Tokenizer};
use crate::{Token, Transcript};

/// Encoder frame length: 10 ms hop × subsampling 4.
const FRAME_MS: u32 = 40;

enum Head {
  /// The encoder outputs the joint's encoder projection.
  Rnnt(Box<Decoder>),
  /// The encoder outputs the logits.
  Ctc { blank: usize },
}

enum Backend {
  Gpu(Box<(Gpu, Encoder)>),
  Ane(crate::ane::Model),
}

pub struct Gigaam {
  backend: Backend,
  frontend: Frontend,
  head: Head,
  tok: Tokenizer,
}

/// The encoder's checked metadata.
struct Config {
  conformer: conformer::Config,
  mels: usize,
  /// RoPE base: GigaAM passes `pos_emb_max_len` (5000) as theta.
  theta: f64,
  /// `rnnt` or `ctc`.
  rnnt: bool,
  /// The head's last linear layer and its weight's dims.
  head: (&'static str, Vec<usize>),
}

fn config(g: &Gguf) -> Result<Config> {
  g.check("general.architecture", &["gigaam"])?;
  g.check("stt.gigaam.encoder.self_attention_model", &["rotary"])?;
  g.check("tokenizer.ggml.model", &["bpe", "char"])?;
  g.check("stt.gigaam.encoder.subs_kernel_size", &[5u32])?;
  g.check("stt.gigaam.encoder.subsampling_factor", &[4u32])?;
  g.check("stt.frontend.sample_rate", &[16000u32])?;
  g.check("stt.frontend.n_fft", &[frontend::N_FFT as u32])?;
  g.check("stt.frontend.win_length", &[frontend::N_FFT as u32])?;
  g.check("stt.frontend.hop_length", &[frontend::HOP as u32])?;
  let u = |key: &str| g.get::<u32>(&format!("stt.gigaam.{key}")).map(|v| v as usize);
  let cfg = conformer::Config::load(g, "gigaam", Attention::Rotary, true)?;
  let d = cfg.d;
  let kind = g.get::<&str>("stt.gigaam.head_kind")?;
  let (rnnt, head) = match kind {
    "rnnt" => (true, ("joint.enc", vec![d, u("joint.hidden")?])),
    "ctc" => {
      g.check("stt.gigaam.head.feat_in", &[d as u32])?;
      (false, ("head.ctc", vec![1, d, u("head.num_classes")?]))
    }
    _ => bail!("stt.gigaam.head_kind is {kind:?}, only \"rnnt\" and \"ctc\" are supported"),
  };
  Ok(Config {
    conformer: cfg,
    mels: g.get::<u32>("stt.frontend.num_mels")? as usize,
    theta: u("encoder.pos_emb_max_len")? as f64,
    rnnt,
    head,
  })
}

impl Gigaam {
  /// Runs the encoder on `ane` when given, else on the GPU.
  pub fn load(g: &Gguf, ane: Option<crate::ane::Model>) -> Result<Self> {
    let cfg = config(g)?;
    let u = |key: &str| g.get::<u32>(&format!("stt.gigaam.{key}")).map(|v| v as usize);
    let tok = Tokenizer::new(&g.array("tokenizer.ggml.tokens")?);
    let blank = g.get::<u32>("tokenizer.ggml.blank_token_id")? as usize;
    let classes = if cfg.rnnt { u("joint.num_classes")? } else { u("head.num_classes")? };
    if tok.len() != classes || blank >= classes {
      bail!("vocabulary of {} tokens, {classes} classes, blank {blank}", tok.len());
    }
    let head = if cfg.rnnt {
      g.check("stt.gigaam.joint.activation", &["relu"])?;
      g.check("stt.gigaam.predictor.n_layers", &[1u32])?;
      g.check("stt.gigaam.predictor.vocab", &[classes as u32])?;
      let joint = u("joint.hidden")?;
      Head::Rnnt(Box::new(Decoder::new(g, u("predictor.hidden")?, joint, classes, blank)?))
    } else {
      Head::Ctc { blank }
    };
    let frontend = Frontend::new(g, cfg.mels)?;
    let backend = match ane {
      Some(model) => Backend::Ane(model),
      None => {
        let gpu = Gpu::new(true)?;
        let (name, dims) = &cfg.head;
        let encoder = Encoder::new(&gpu, g, cfg.conformer, cfg.mels, cfg.theta, (name, dims))?;
        Backend::Gpu(Box::new((gpu, encoder)))
      }
    };
    Ok(Self { backend, frontend, head, tok })
  }

  pub fn device(&self) -> &crate::Device {
    match &self.backend {
      Backend::Gpu(gpu) => gpu.0.info(),
      Backend::Ane(_) => crate::ane::device(),
    }
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let mel = self.frontend.compute(pcm);
    if mel.is_empty() {
      return Ok(Transcript::default());
    }
    let enc = match &self.backend {
      Backend::Gpu(gpu) => gpu.1.run(&gpu.0, &mel)?,
      Backend::Ane(model) => ane::run(model, &mel, self.frontend.mels())?,
    };
    let ids = match &self.head {
      Head::Rnnt(decoder) => decoder.decode(&enc),
      Head::Ctc { blank } => decoder::ctc(&enc, self.tok.len(), *blank),
    };
    let (ids, frames): (Vec<usize>, Vec<u32>) =
      ids.into_iter().map(|(id, frame)| (id as usize, frame)).unzip();
    let pieces = self.tok.pieces(&ids);
    // Character vocabularies have a plain space and no `▁`.
    let text = sentencepiece::text(&pieces);
    let tokens: Vec<Token> = (ids.iter().zip(frames).zip(pieces))
      .map(|((&id, frame), piece)| Token { id: id as u32, piece, start_ms: frame * FRAME_MS })
      .collect();
    Ok(Transcript { text, tokens })
  }
}
