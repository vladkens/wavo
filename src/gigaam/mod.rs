// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GigaAM v3: log-mel frontend on the CPU, Conformer encoder with rotary attention on the GPU, and
//! one of two heads: RNN-T (greedy decoding on the CPU) or CTC (logits as the encoder's last GEMM,
//! greedy collapse on the CPU). The e2e variants use SentencePiece pieces, the others characters.

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
use crate::{Token, Transcript};

/// Encoder frame length: 10 ms hop × subsampling 4.
const FRAME_MS: u32 = 40;

enum Head {
  /// The encoder outputs the joint's encoder projection.
  Rnnt(Box<Decoder>),
  /// The encoder outputs the logits.
  Ctc { blank: usize },
}

pub struct Gigaam {
  gpu: Gpu,
  frontend: Frontend,
  encoder: Encoder,
  head: Head,
  vocab: Vec<String>,
}

impl Gigaam {
  pub fn load(g: &Gguf) -> Result<Self> {
    g.check("general.architecture", &["gigaam"])?;
    g.check("stt.gigaam.encoder.self_attention_model", &["rotary"])?;
    g.check("tokenizer.ggml.model", &["bpe", "char"])?;
    g.check("stt.gigaam.encoder.subs_kernel_size", &[5u32])?;
    g.check("stt.gigaam.encoder.subsampling_factor", &[4u32])?;
    g.check("stt.frontend.sample_rate", &[16000u32])?;
    g.check("stt.frontend.n_fft", &[frontend::N_FFT as u32])?;
    g.check("stt.frontend.win_length", &[frontend::N_FFT as u32])?;
    g.check("stt.frontend.hop_length", &[160u32])?;
    let u = |key: &str| g.get::<u32>(&format!("stt.gigaam.{key}")).map(|v| v as usize);
    let cfg = conformer::Config::load(g, "gigaam", Attention::Rotary, true)?;
    let d = cfg.d;
    let mels = g.get::<u32>("stt.frontend.num_mels")? as usize;
    let theta = u("encoder.pos_emb_max_len")? as f64;
    let vocab: Vec<String> =
      g.array::<&str>("tokenizer.ggml.tokens")?.into_iter().map(String::from).collect();
    let blank = g.get::<u32>("tokenizer.ggml.blank_token_id")? as usize;
    let kind = g.get::<&str>("stt.gigaam.head_kind")?;
    let classes = match kind {
      "rnnt" => u("joint.num_classes")?,
      "ctc" => u("head.num_classes")?,
      _ => bail!("stt.gigaam.head_kind is {kind:?}, only \"rnnt\" and \"ctc\" are supported"),
    };
    if vocab.len() != classes || blank >= classes {
      bail!("vocabulary of {} tokens, {classes} classes, blank {blank}", vocab.len());
    }

    let gpu = Gpu::new(true)?;
    let frontend = Frontend::new(g, mels)?;
    let encoder = |head: (&str, &[usize])| Encoder::new(&gpu, g, cfg, mels, theta, head);
    let (head, encoder) = if kind == "rnnt" {
      g.check("stt.gigaam.joint.activation", &["relu"])?;
      g.check("stt.gigaam.predictor.n_layers", &[1u32])?;
      g.check("stt.gigaam.predictor.vocab", &[classes as u32])?;
      let joint = u("joint.hidden")?;
      let decoder = Decoder::new(g, u("predictor.hidden")?, joint, classes, blank)?;
      (Head::Rnnt(Box::new(decoder)), encoder(("joint.enc", &[d, joint]))?)
    } else {
      g.check("stt.gigaam.head.feat_in", &[d as u32])?;
      (Head::Ctc { blank }, encoder(("head.ctc", &[1, d, classes]))?)
    };
    // The GPU's first use of the pipelines and the weight memory costs 20–40 ms whatever the input
    // length: pay it here on 8 silent frames instead of in the first call (see `Gpu::warm_up`).
    if gpu.warm_up() {
      encoder.run(&gpu, &vec![0.0; 8 * mels])?;
    }
    Ok(Self { gpu, frontend, encoder, head, vocab })
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let mel = self.frontend.compute(pcm);
    if mel.is_empty() {
      return Ok(Transcript::default());
    }
    let enc = self.encoder.run(&self.gpu, &mel)?;
    let ids = match &self.head {
      Head::Rnnt(decoder) => decoder.decode(&enc),
      Head::Ctc { blank } => decoder::ctc(&enc, self.vocab.len(), *blank),
    };
    let tokens: Vec<Token> = (ids.into_iter())
      .map(|(id, frame)| Token {
        id,
        piece: self.vocab[id as usize].clone(),
        start_ms: frame * FRAME_MS,
      })
      .collect();
    // Character vocabularies have a plain space and no `▁`.
    let text = tokens.iter().map(|t| t.piece.as_str()).collect::<String>().replace('▁', " ");
    let text = text.strip_prefix(' ').unwrap_or(&text).to_string();
    Ok(Transcript { text, tokens })
  }
}
