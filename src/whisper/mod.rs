// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Whisper: log-mel frontend on the CPU, encoder and autoregressive decoder on the GPU or the CPU,
//! greedy
//! search with segment timestamps on the CPU. Audio runs in 30 s windows, the next one starting
//! where the last segment closed, as transcribe.cpp does for every input length.

mod cpu;
mod encoder;
mod frontend;
mod search;
pub(crate) mod tokenizer;

use std::sync::{Mutex, MutexGuard, PoisonError};

use encoder::Encoder;
pub(crate) use frontend::SAMPLES;
use frontend::{Frontend, HOP, N_FFT};
use search::Ids;
use tokenizer::Tokenizer;

use crate::ane::{Backend, Target};
use crate::cpu::transformer::{self as on_cpu, Cache};
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Gpu, Groups, Recorded};
use crate::transformer::{self, Decoder, Steps};
use crate::{Token, Transcript};

/// The no-speech skip: a window is dropped when the no-speech probability is above this and the
/// average log-probability of its tokens below `LOGPROB`.
pub(crate) const NO_SPEECH: f32 = 0.6;
const LOGPROB: f32 = -1.0;
/// Encoder frames of a window.
const FRAMES: usize = SAMPLES / HOP / 2;

pub struct Whisper {
  frontend: Frontend,
  tok: Tokenizer,
  ids: Ids,
  sot: usize,
  transcribe: usize,
  no_speech: usize,
  languages: Vec<usize>,
  mels: usize,
  backend: Backend<OnGpu, OnCpu>,
}

struct OnGpu {
  encoder: Encoder,
  decoder: Decoder,
  arena: Mutex<Arena>,
}

struct OnCpu {
  encoder: cpu::Encoder,
  decoder: on_cpu::Decoder,
}

/// A transcription's hold on its backend: the GPU's arena, or the CPU's caches of the window.
enum Run<'a> {
  Gpu(&'a Gpu, &'a OnGpu, MutexGuard<'a, Arena>),
  Cpu(&'a OnCpu, Option<Cache>),
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
  pub fn load(g: &Gguf, target: Target) -> Result<Self> {
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
        neox: false,
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
    // transcribe.cpp's converter writes a suppress list only when it isn't empty.
    let list =
      |key: &str| g.optional_array::<i32>(key).map(|v| v.iter().map(|&i| i as usize).collect());
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

    let mels = u("encoder.num_mel_bins")?;
    let frontend = Frontend::new(g, mels)?;
    let cfg = || encoder::Config { mels, frames: FRAMES };
    let gpu = |gpu: &Gpu| {
      let encoder = Encoder::load(gpu, g, cfg(), config("encoder")?)?;
      let decoder = Decoder::load(gpu, g, config("decoder")?, vocab, context)?;
      let arena = Arena {
        enc: encoder.buffers(gpu),
        dec: decoder.steps(gpu, FRAMES),
        ids: gpu.buffer(2),
        read: gpu.readback(vocab),
        window: Vec::new(),
        steps: Vec::new(),
        next: None,
      };
      Ok(OnGpu { encoder, decoder, arena: Mutex::new(arena) })
    };
    let cpu = || {
      let encoder = cpu::Encoder::load(g, cfg(), config("encoder")?)?;
      Ok(OnCpu { encoder, decoder: on_cpu::Decoder::load(g, config("decoder")?, vocab, context)? })
    };
    Ok(Self {
      sot: u("sot_token_id")?,
      transcribe: u("transcribe_token_id")?,
      no_speech: no_timestamps - 1,
      languages,
      frontend,
      tok,
      ids,
      mels,
      backend: Backend::new(target, gpu, cpu)?,
    })
  }

  pub fn device(&self) -> &crate::Device {
    self.backend.device()
  }

  fn lock(&self) -> Run<'_> {
    match &self.backend {
      Backend::Gpu(b) => {
        Run::Gpu(&b.0, &b.1, b.1.arena.lock().unwrap_or_else(PoisonError::into_inner))
      }
      Backend::Cpu(c) => Run::Cpu(c, None),
      Backend::Ane(_) => unreachable!("Whisper has no Neural Engine program"),
    }
  }

  /// Whether the model has language tokens: the multilingual ones, not `.en`.
  pub fn multilingual(&self) -> bool {
    !self.languages.is_empty()
  }

  /// With `language`, an index in `general.languages`, every window is decoded in that language;
  /// without, in the one detected on the first window. Returns the index of the language too
  /// (`None` for `.en` models).
  pub fn transcribe(
    &self,
    pcm: &[f32],
    mut language: Option<usize>,
  ) -> Result<(Transcript, Option<usize>)> {
    let (mels, window) = (self.mels, 2 * FRAMES);
    let mel = self.frontend.compute(pcm);
    let total = mel.len() / mels;
    let mut a = self.lock();
    let mut seek = 0;
    let (mut text, mut tokens) = (Vec::new(), Vec::new());
    while seek < total {
      let frames = (total - seek).min(window);
      let mut chunk = mel[seek * mels..(seek + frames) * mels].to_vec();
      chunk.resize(window * mels, 0.0);
      let sot = self.window(&mut a, &chunk)?;
      let no_speech = search::probability(&sot, self.no_speech);
      let prompt = match self.multilingual() {
        true => {
          let i = *language.get_or_insert_with(|| best(&self.language_logits(&sot)));
          vec![self.sot, self.languages[i], self.transcribe]
        }
        false => vec![self.sot],
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
    Ok((Transcript { text, tokens, language: None }, language))
  }

  /// The logits of the language tokens on SOT, in the order of `general.languages`, and the
  /// no-speech probability of up to 30 s of audio: one window, as `transcribe` runs its first.
  pub fn detect(&self, pcm: &[f32]) -> Result<(Vec<f32>, f32)> {
    let window = 2 * FRAMES * self.mels;
    let mut mel = self.frontend.compute(&pcm[..pcm.len().min(SAMPLES)]);
    mel.resize(window, 0.0);
    let sot = self.window(&mut self.lock(), &mel)?;
    Ok((self.language_logits(&sot), search::probability(&sot, self.no_speech)))
  }

  fn language_logits(&self, sot: &[f32]) -> Vec<f32> {
    self.languages.iter().map(|&id| sot[id]).collect()
  }

  /// Encodes a window of mel frames, computes the cross K/V and runs the decoder on SOT; returns
  /// its logits.
  fn window(&self, run: &mut Run, mel: &[f32]) -> Result<Vec<f32>> {
    let (gpu, on, a) = match run {
      Run::Gpu(gpu, on, a) => (gpu, on, &mut **a),
      Run::Cpu(on, cache) => {
        let enc = on.encoder.run(mel);
        let c = cache.insert(on.decoder.cache(&enc, FRAMES));
        return Ok(on.decoder.step(c, self.sot, 0, None));
      }
    };
    let Arena { enc, dec, ids, read, window, .. } = a;
    gpu.write(&enc.mel, mel);
    gpu.write(&enc.t.x, &on.encoder.positions);
    gpu.write(ids, &[f32::from_bits(self.sot as u32)]);
    gpu.run(window, &dec.logits, read, on.decoder.vocab, |p| {
      on.encoder.record(p, enc, FRAMES);
      on.decoder.record_cross(p, dec, &enc.t.y, &enc.t.h, FRAMES);
      on.decoder.record_step(p, dec, (ids, 0), 0, FRAMES, None);
    })
  }

  /// Runs the decoder on `tokens` (at most 2) from position `pos`; returns the last one's logits.
  /// On the GPU in one pass, and while the GPU runs it, the CPU records a step at the position
  /// after, which the next call takes if it is for that position. Every step has the same
  /// dispatches over the same buffers, so the bind groups are shared.
  fn steps(&self, run: &mut Run, tokens: &[usize], pos: usize) -> Result<Vec<f32>> {
    let (gpu, on, a) = match run {
      Run::Gpu(gpu, on, a) => (gpu, on, &mut **a),
      Run::Cpu(on, cache) => {
        let c = cache.as_mut().expect("a window before its steps");
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
          logits = on.decoder.step(c, token, pos + i, None);
        }

        return Ok(logits);
      }
    };
    let Arena { dec, ids, read, steps, next, .. } = a;
    let record = |steps: &mut Groups, pos: usize, n: usize| {
      gpu.record(steps, &dec.logits, read, on.decoder.vocab, |p| {
        for i in 0..n {
          on.decoder.record_step(p, dec, (ids, i), pos + i, FRAMES, None);
        }
      })
    };
    let r = match next.take() {
      Some((at, r)) if at == pos && tokens.len() == 1 => r,
      _ => record(steps, pos, tokens.len()),
    };
    let bits: Vec<f32> = tokens.iter().map(|&t| f32::from_bits(t as u32)).collect();
    gpu.write(ids, &bits);
    let after = pos + tokens.len();
    gpu.finish(r, read, || {
      if after < on.decoder.context {
        *next = Some((after, record(steps, after, 1)));
      }
    })
  }
}

/// The index of the highest logit, the first of equal ones, as transcribe.cpp picks a language.
fn best(logits: &[f32]) -> usize {
  (0..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best })
}
