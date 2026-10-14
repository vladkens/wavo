// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Whisper: log-mel frontend on the CPU, encoder and autoregressive decoder on the GPU, greedy
//! search with segment timestamps on the CPU. Audio runs in 30 s windows, the next one starting
//! where the last segment closed, as transcribe.cpp does for every input length.

mod encoder;
mod frontend;
mod search;
pub(crate) mod tokenizer;

use std::sync::{Mutex, PoisonError};

use encoder::Encoder;
use frontend::{Frontend, HOP, N_FFT, SAMPLES};
use search::Ids;
use tokenizer::Tokenizer;

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Gpu, Groups, Recorded};
use crate::transformer::{self, Decoder, Steps};
use crate::{Token, Transcript};

/// The no-speech skip: a window is dropped when the no-speech probability is above this and the
/// average log-probability of its tokens below `LOGPROB`.
const NO_SPEECH: f32 = 0.6;
const LOGPROB: f32 = -1.0;

pub struct Whisper {
  pub gpu: Gpu,
  frontend: Frontend,
  tok: Tokenizer,
  ids: Ids,
  sot: usize,
  transcribe: usize,
  no_speech: usize,
  languages: Vec<usize>,
  encoder: Encoder,
  decoder: Decoder,
  arena: Mutex<Arena>,
}

/// Activations, caches and the bind groups recorded over them, one list per kind of pass.
struct Arena {
  enc: encoder::Buffers,
  dec: Steps,
  /// The tokens of a pass's steps, as u32 bits.
  ids: Buffer,
  read: Buffer,
  window: Groups,
  steps: Groups,
  /// The step at a position, recorded while the GPU ran the one before.
  next: Option<(usize, Recorded)>,
}

impl Whisper {
  pub fn load(g: &Gguf) -> Result<Self> {
    let u = |key: &str| g.get::<u32>(&format!("stt.whisper.{key}")).map(|v| v as usize);
    for (key, want) in [
      ("stt.frontend.type", "mel"),
      ("stt.frontend.window", "hann_periodic"),
      ("stt.frontend.normalize", "whisper_logmel"),
      ("stt.frontend.pad_mode", "reflect"),
      ("stt.whisper.encoder.activation", "gelu"),
      ("stt.whisper.decoder.activation", "gelu"),
    ] {
      g.check(key, &[want])?;
    }
    for (key, want) in [
      ("stt.frontend.sample_rate", 16000),
      ("stt.frontend.n_fft", N_FFT),
      ("stt.frontend.win_length", N_FFT),
      ("stt.frontend.hop_length", HOP),
      ("stt.frontend.n_samples", SAMPLES),
      ("stt.frontend.nb_max_frames", SAMPLES / HOP),
      ("stt.whisper.encoder.max_source_positions", SAMPLES / HOP / 2),
    ] {
      g.check(key, &[want as u32])?;
    }
    g.check("stt.whisper.decoder.tie_word_embeddings", &[true])?;
    g.check("stt.whisper.decoder.scale_embedding", &[false])?;
    let d = u("encoder.d_model")?;
    let config = |part: &str| {
      let key = |name: &str| u(&format!("{part}.{name}"));
      Ok::<_, crate::Error>(transformer::Config {
        d: key("d_model")?,
        heads: key("n_heads")?,
        layers: key("n_layers")?,
        d_ff: key("ffn_dim")?,
        bias: true,
        rot: 0,
        gated: false,
      })
    };
    let (enc, dec) = (config("encoder")?, config("decoder")?);
    let (vocab, context) = (u("decoder.vocab_size")?, u("decoder.max_target_positions")?);
    // The flash kernel takes a head of up to 64.
    for heads in [enc.heads, dec.heads] {
      let hd = d / heads.max(1);
      if dec.d != d || hd * heads != d || hd > 64 || !hd.is_multiple_of(8) {
        bail!("unsupported shape: d_model {d}, {heads} heads");
      }
    }
    let tok = Tokenizer::new(g)?;
    if tok.len() != vocab {
      bail!("vocabulary of {} tokens, decoder.vocab_size {vocab}", tok.len());
    }
    let list = |key: &str| g.array::<i32>(key).map(|v| v.iter().map(|&i| i as usize).collect());
    let no_timestamps = u("no_timestamps_token_id")?;
    let ids = Ids {
      eot: g.get::<u32>("tokenizer.ggml.eos_token_id")? as usize,
      no_timestamps,
      begin: no_timestamps + 1,
      suppress: list("stt.whisper.suppress_tokens")?,
      begin_suppress: list("stt.whisper.begin_suppress_tokens")?,
    };
    // English-only (`.en`) models have no language or task tokens: they prompt with SOT alone.
    let languages = match g.get::<bool>("stt.capability.lang_detect")? {
      true => (g.array::<&str>("general.languages")?.iter())
        .map(|code| tok.find(&format!("<|{code}|>")))
        .collect::<Option<Vec<_>>>(),
      false => Some(Vec::new()),
    };
    let Some(languages) = languages else { bail!("a language of general.languages has no token") };
    let all = [&ids.suppress[..], &ids.begin_suppress, &languages, &[ids.eot, no_timestamps]];
    if all.iter().flat_map(|v| v.iter()).any(|&i| i >= vocab) {
      bail!("special token ids out of the vocabulary");
    }

    let gpu = Gpu::new(true)?;
    let mels = u("encoder.num_mel_bins")?;
    let frontend = Frontend::new(g, mels)?;
    let encoder = Encoder::load(&gpu, g, encoder::Config { mels, frames: SAMPLES / HOP / 2 }, enc)?;
    let decoder = Decoder::load(&gpu, g, dec, vocab, context)?;
    let frames = encoder.cfg.frames;
    let arena = Arena {
      enc: encoder.buffers(&gpu),
      dec: decoder.steps(&gpu, frames),
      ids: gpu.buffer(2),
      read: gpu.readback(vocab),
      window: Vec::new(),
      steps: Vec::new(),
      next: None,
    };
    Ok(Self {
      sot: u("sot_token_id")?,
      transcribe: u("transcribe_token_id")?,
      no_speech: no_timestamps - 1,
      languages,
      gpu,
      frontend,
      tok,
      ids,
      encoder,
      decoder,
      arena: Mutex::new(arena),
    })
  }

  fn lock(&self) -> std::sync::MutexGuard<'_, Arena> {
    self.arena.lock().unwrap_or_else(PoisonError::into_inner)
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let mels = self.encoder.cfg.mels;
    let window = 2 * self.encoder.cfg.frames;
    let mel = self.frontend.compute(pcm);
    let total = mel.len() / mels;
    let mut a = self.lock();
    let (mut seek, mut language) = (0, None);
    let (mut text, mut tokens) = (Vec::new(), Vec::new());
    while seek < total {
      let frames = (total - seek).min(window);
      let mut chunk = mel[seek * mels..(seek + frames) * mels].to_vec();
      chunk.resize(window * mels, 0.0);
      let sot = self.window(&mut a, &chunk)?;
      let no_speech = search::probability(&sot, self.no_speech);
      let prompt = match self.languages.first() {
        Some(&first) => {
          // Detected on the first window: the first of the highest.
          let best = |best: usize, &id: &usize| if sot[id] > sot[best] { id } else { best };
          let language = *language.get_or_insert_with(|| self.languages.iter().fold(first, best));
          vec![self.sot, language, self.transcribe]
        }
        None => vec![self.sot],
      };
      let logits = if prompt.len() > 1 { self.steps(&mut a, &prompt[1..], 1)? } else { sot };
      let (generated, logprob) =
        search::decode(&self.ids, logits, prompt.len(), no_speech > NO_SPEECH, |token, pos| {
          self.steps(&mut a, &[token], pos)
        })?;
      if no_speech > NO_SPEECH && logprob < LOGPROB {
        seek += frames;
        continue;
      }
      let (segments, advance) = search::retrieve(&self.ids, &generated, frames);
      for s in segments {
        let pieces = tokenizer::pieces(|id| self.tok.bytes(id), &s.text);
        if tokenizer::trim(&pieces.concat()).is_empty() {
          continue;
        }
        let start_ms = (seek * 10 + s.start * 20) as u32;
        let token = |(&id, piece): (&usize, String)| Token { id: id as u32, piece, start_ms };
        tokens.extend(s.text.iter().zip(pieces).map(token));
      }
      // Text tokens precede EOT; the ids after it are special or timestamps.
      text.extend(
        generated.iter().filter(|&&id| id < self.ids.eot).flat_map(|&id| self.tok.bytes(id)),
      );
      seek += if advance == 0 { frames } else { advance };
    }
    let text = tokenizer::trim(&String::from_utf8_lossy(&text)).to_string();
    Ok(Transcript { text, tokens })
  }

  /// Encodes a window of mel frames, computes the cross K/V and runs the decoder on SOT; returns
  /// its logits.
  fn window(&self, a: &mut Arena, mel: &[f32]) -> Result<Vec<f32>> {
    let Arena { enc, dec, ids, read, window, .. } = a;
    let t = self.encoder.cfg.frames;
    self.gpu.write(&enc.mel, mel);
    self.gpu.write(&enc.t.x, &self.encoder.positions);
    self.gpu.write(ids, &[f32::from_bits(self.sot as u32)]);
    self.gpu.run(window, &dec.logits, read, self.decoder.vocab, |p| {
      self.encoder.record(p, enc, t);
      self.decoder.record_cross(p, dec, &enc.t.y, &enc.t.h, t);
      self.decoder.record_step(p, dec, (ids, 0), 0, t, None);
    })
  }

  /// Runs the decoder on `tokens` (at most 2) from position `pos` in one pass; returns the last
  /// one's logits. While the GPU runs it, the CPU records a step at the position after, which the
  /// next call takes if it is for that position. Every step has the same dispatches over the same
  /// buffers, so the bind groups are shared.
  fn steps(&self, a: &mut Arena, tokens: &[usize], pos: usize) -> Result<Vec<f32>> {
    let frames = self.encoder.cfg.frames;
    let Arena { dec, ids, read, steps, next, .. } = a;
    let record = |steps: &mut Groups, pos: usize, n: usize| {
      self.gpu.record(steps, &dec.logits, read, self.decoder.vocab, |p| {
        for i in 0..n {
          self.decoder.record_step(p, dec, (ids, i), pos + i, frames, None);
        }
      })
    };
    let r = match next.take() {
      Some((at, r)) if at == pos && tokens.len() == 1 => r,
      _ => record(steps, pos, tokens.len()),
    };
    let bits: Vec<f32> = tokens.iter().map(|&t| f32::from_bits(t as u32)).collect();
    self.gpu.write(ids, &bits);
    let after = pos + tokens.len();
    self.gpu.finish(r, read, || {
      if after < self.decoder.context {
        *next = Some((after, record(steps, after, 1)));
      }
    })
  }
}
