// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Moonshine: an encoder over raw samples and an autoregressive decoder, both on the GPU, greedy
//! over the whole input in one pass as transcribe.cpp does (`src/arch/moonshine/model.cpp`): the
//! prompt is `<s>`, decoding stops at `</s>`, after the decoder's positions or on a loop. The text
//! has no timestamps: every token starts at 0, the one untimed segment.

mod encoder;
mod repeat;
mod tokenizer;

use std::sync::{Mutex, PoisonError};

use encoder::{Encoder, frames};
use tokenizer::Tokenizer;

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Gpu, Groups};
use crate::transformer::{self, Decoder, Steps};
use crate::{Token, Transcript};

pub struct Moonshine {
  gpu: Gpu,
  tok: Tokenizer,
  bos: usize,
  eos: usize,
  /// Rotary frequencies θ^(−2i/rot) for i < rot / 2, in f32 as ggml-metal's `rope.metal`
  /// computes them.
  freqs: Vec<f32>,
  encoder: Encoder,
  decoder: Decoder,
  arena: Mutex<Arena>,
}

/// Activations, caches and the bind groups recorded over them, for the longest input so far.
struct Arena {
  enc: encoder::Buffers,
  dec: Steps,
  /// Per position (encoder frames or decoder positions) rot / 2 cosines, then as many sines.
  rope: Buffer,
  /// The next token's id, as the bits of a float.
  next: Buffer,
  read: Buffer,
  window: Groups,
  steps: Groups,
}

impl Moonshine {
  pub fn load(g: &Gguf) -> Result<Self> {
    let u = |key: &str| g.get::<u32>(&format!("stt.moonshine.{key}")).map(|v| v as usize);
    g.check("stt.frontend.type", &["raw"])?;
    g.check("stt.frontend.sample_rate", &[16000u32])?;
    g.check("stt.moonshine.encoder.activation", &["gelu"])?;
    g.check("stt.moonshine.decoder.activation", &["silu"])?;
    g.check("stt.moonshine.decoder.tie_word_embeddings", &[true])?;
    g.check("stt.moonshine.attention_bias", &[false])?;
    g.check("stt.moonshine.conv_stem.groupnorm_num_groups", &[1u32])?;
    let stem = |key: &str| g.array::<u32>(&format!("stt.moonshine.conv_stem.{key}"));
    let (kernels, strides): (Vec<u32>, Vec<u32>) =
      encoder::STEM.into_iter().map(|(k, s)| (k as u32, s as u32)).unzip();
    if (stem("kernel_sizes")?, stem("strides")?) != (kernels, strides) {
      bail!("conv stem kernels or strides other than {:?}", encoder::STEM);
    }
    let factor = g.get::<f32>("stt.moonshine.partial_rotary_factor")?;
    let theta = g.get::<f32>("stt.moonshine.rope_theta")?;
    let config = |part: &str| {
      let key = |name: &str| u(&format!("{part}.{name}"));
      let (d, heads) = (key("d_model")?, key("n_heads")?);
      let hd = d / heads.max(1);
      // `attend` keeps a head of up to 128 as pairs.
      if hd * heads != d || key("n_kv_heads")? != heads || hd > 128 || hd % 2 != 0 {
        bail!("unsupported {part} shape: d_model {d}, {heads} heads");
      }
      Ok(transformer::Config {
        d,
        heads,
        layers: key("n_layers")?,
        d_ff: key("ffn_dim")?,
        bias: false,
        // The rotated dims of a head, rounded down to even (`weights.h`).
        rot: (hd as f32 * factor) as usize & !1,
        gated: part == "decoder",
      })
    };
    let (enc, dec) = (config("encoder")?, config("decoder")?);
    let channels = g.array::<u32>("stt.moonshine.conv_stem.channels")?;
    if dec.d != enc.d || dec.rot != enc.rot {
      bail!("encoder and decoder of different widths or rotary dims");
    }
    if channels != [enc.d, 2 * enc.d, enc.d].map(|c| c as u32) {
      bail!("conv stem channels {channels:?}, expected [d, 2·d, d] for d_model {}", enc.d);
    }
    let (vocab, context) = (u("decoder.vocab_size")?, u("decoder.max_position_embeddings")?);
    let tok = Tokenizer::new(g)?;
    let (bos, eos) = (u("decoder_start_token_id")?, u("eos_token_id")?);
    if tok.len() != vocab {
      bail!("vocabulary of {} tokens, decoder.vocab_size {vocab}", tok.len());
    }
    if bos >= vocab || eos >= vocab {
      bail!("special token ids {bos} and {eos} out of the vocabulary of {vocab}");
    }
    let eps = g.get::<f32>("stt.moonshine.conv_stem.groupnorm_eps")?;
    let rot = enc.rot;
    let freqs = (0..rot / 2).map(|i| theta.powf(-1.0 / rot as f32 * (2 * i) as f32)).collect();

    let gpu = Gpu::new(true)?;
    let encoder = Encoder::load(&gpu, g, enc, eps)?;
    let decoder = Decoder::load(&gpu, g, dec, vocab, context)?;
    let arena = Arena {
      enc: encoder.buffers(&gpu, 0),
      dec: decoder.steps(&gpu, 0),
      rope: gpu.buffer(1),
      next: gpu.buffer(1),
      read: gpu.readback(1),
      window: Vec::new(),
      steps: Vec::new(),
    };
    Ok(Self { gpu, tok, bos, eos, freqs, encoder, decoder, arena: Mutex::new(arena) })
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let n = pcm.len();
    let t = frames(n)[2];
    if t == 0 {
      return Ok(Transcript::default());
    }
    let mut a = self.arena.lock().unwrap_or_else(PoisonError::into_inner);
    if n > a.enc.samples {
      self.grow(&mut a, n);
    }
    let (eos, context) = (self.eos, self.decoder.context);
    self.gpu.write(&a.enc.pcm, pcm);
    // The encoder, the cross K/V and the step on `<s>` in one pass.
    let Arena { enc, dec, rope, next, read, window, steps } = &mut *a;
    let (enc, dec, rope, next) = (&*enc, &*dec, &*rope, &*next);
    let mut token = self.gpu.run(window, next, read, 1, |p| {
      self.encoder.record(p, enc, n, rope);
      self.decoder.record_cross(p, dec, &enc.t.y, &enc.t.h, t);
      self.decoder.record_step(p, dec, self.bos, 0, t, Some(rope));
      p.argmax(&dec.logits, next, self.decoder.vocab);
    })?[0]
      .to_bits() as usize;
    // As the reference's loop: a step per position up to the context, the repetition guard after
    // each token, and on a stop at the context a repeating tail trimmed.
    let mut ids = Vec::new();
    let mut looped = false;
    if token != eos {
      ids.push(token);
    }
    let mut pos = 1;
    while token != eos && pos < context {
      token = self.gpu.run(steps, next, read, 1, |p| {
        self.decoder.record_step(p, dec, token, pos, t, Some(rope));
        p.argmax(&dec.logits, next, self.decoder.vocab);
      })?[0]
        .to_bits() as usize;
      pos += 1;
      if token != eos {
        ids.push(token);
        if repeat::stop(&mut ids) {
          looped = true;
          break;
        }
      }
    }
    if !looped && token != eos {
      repeat::trim_at_budget(&mut ids);
    }
    let token = |(&id, piece): (&usize, String)| Token { id: id as u32, piece, start_ms: 0 };
    let tokens = ids.iter().zip(self.tok.pieces(&ids)).map(token).collect();
    Ok(Transcript { text: self.tok.text(&ids), tokens })
  }

  /// Buffers for `n` samples, the rotary table for as many positions as they and the decoder need.
  fn grow(&self, a: &mut Arena, n: usize) {
    let t = frames(n)[2];
    a.enc = self.encoder.buffers(&self.gpu, n);
    a.dec = self.decoder.steps(&self.gpu, t);
    let positions = t.max(self.decoder.context);
    let mut table = Vec::with_capacity(positions * 2 * self.freqs.len());
    for pos in 0..positions {
      let angles = self.freqs.iter().map(|f| pos as f32 * f);
      table.extend(angles.clone().map(f32::cos).chain(angles.map(f32::sin)));
    }
    a.rope = self.gpu.buffer(table.len());
    self.gpu.write(&a.rope, &table);
    a.window.clear();
    a.steps.clear();
  }
}
