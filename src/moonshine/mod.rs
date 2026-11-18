// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Moonshine: an encoder over raw samples and an autoregressive decoder, both on the GPU or the
//! CPU, greedy
//! over the whole input in one pass as transcribe.cpp does (`src/arch/moonshine/model.cpp`): the
//! prompt is `<s>`, decoding stops at `</s>`, after the decoder's positions or on a loop. The text
//! has no timestamps: every token starts at 0, the one untimed segment.

mod cpu;
mod encoder;
mod repeat;

use std::sync::{Mutex, PoisonError};

use encoder::{Encoder, STEM, frames};

use crate::ane::{Backend, Target};
use crate::cpu::transformer as on_cpu;
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::{Buffer, Gpu, Groups, Pass};
use crate::sentencepiece::{self, Tokenizer};
use crate::transformer::{self, Decoder, Steps};
use crate::{Token, Transcript};

pub struct Moonshine {
  tok: Tokenizer,
  bos: usize,
  eos: usize,
  /// Rotary frequencies θ^(−2i/rot) for i < rot / 2, in f32 as ggml-metal's `rope.metal`
  /// computes them.
  freqs: Vec<f32>,
  /// The decoder's positions.
  context: usize,
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
  pub fn load(g: &Gguf, target: Target) -> Result<Self> {
    let u = |key: &str| g.get::<u32>(&format!("stt.moonshine.{key}")).map(|v| v as usize);
    for (key, want) in [
      ("stt.frontend.type", "raw"),
      ("stt.moonshine.encoder.activation", "gelu"),
      ("stt.moonshine.decoder.activation", "silu"),
    ] {
      g.check(key, &[want])?;
    }
    g.check("stt.frontend.sample_rate", &[16000u32])?;
    g.check("stt.moonshine.decoder.tie_word_embeddings", &[true])?;
    g.check("stt.moonshine.attention_bias", &[false])?;
    g.check("stt.moonshine.conv_stem.groupnorm_num_groups", &[1u32])?;
    let stem = |key: &str| g.array::<u32>(&format!("stt.moonshine.conv_stem.{key}"));
    if stem("kernel_sizes")? != STEM.map(|s| s.0 as u32)
      || stem("strides")? != STEM.map(|s| s.1 as u32)
    {
      bail!("conv stem kernels or strides other than {STEM:?}");
    }
    let factor = g.get::<f32>("stt.moonshine.partial_rotary_factor")?;
    let config = |part: &str| {
      let key = |name: &str| u(&format!("{part}.{name}"));
      let (d, heads) = (key("d_model")?, key("n_heads")?);
      let hd = d / heads.max(1);
      // The rotated dims of a head, rounded down to even (`weights.h`).
      let rot = (hd as f32 * factor) as usize & !1;
      // `attend` keeps a head of up to 128 as pairs.
      if hd * heads != d || key("n_kv_heads")? != heads || hd > 128 || hd % 2 != 0 || rot > hd {
        bail!("unsupported {part} shape: d_model {d}, {heads} heads, {rot} rotary dims");
      }
      Ok(transformer::Config {
        d,
        heads,
        layers: key("n_layers")?,
        d_ff: key("ffn_dim")?,
        bias: false,
        rot,
        neox: false,
        gated: part == "decoder",
      })
    };
    let (enc, dec) = (config("encoder")?, config("decoder")?);
    if dec.d != enc.d || dec.rot != enc.rot {
      bail!("encoder and decoder of different widths or rotary dims");
    }
    let channels = stem("channels")?;
    if channels != [enc.d, 2 * enc.d, enc.d].map(|c| c as u32) {
      bail!("conv stem channels {channels:?}, expected [d, 2·d, d] for d_model {}", enc.d);
    }
    let (vocab, context) = (u("decoder.vocab_size")?, u("decoder.max_position_embeddings")?);
    g.check("tokenizer.ggml.model", &["bpe"])?;
    let tok = Tokenizer::new(&g.array("tokenizer.ggml.tokens")?);
    let (bos, eos) = (u("decoder_start_token_id")?, u("eos_token_id")?);
    if tok.len() != vocab {
      bail!("vocabulary of {} tokens, decoder.vocab_size {vocab}", tok.len());
    }
    if bos >= vocab || eos >= vocab {
      bail!("special token ids {bos} and {eos} out of the vocabulary of {vocab}");
    }
    let (theta, rot) = (g.get::<f32>("stt.moonshine.rope_theta")?, enc.rot);
    let freqs = (0..rot / 2).map(|i| theta.powf(-1.0 / rot as f32 * (2 * i) as f32)).collect();

    let eps = g.get::<f32>("stt.moonshine.conv_stem.groupnorm_eps")?;
    let gpu = |gpu: &Gpu| {
      let encoder = Encoder::load(gpu, g, config("encoder")?, eps)?;
      let decoder = Decoder::load(gpu, g, config("decoder")?, vocab, context)?;
      let arena = Arena {
        enc: encoder.buffers(gpu, 0),
        dec: decoder.steps(gpu, 0),
        rope: gpu.buffer(1),
        next: gpu.buffer(1),
        read: gpu.readback(1),
        window: Vec::new(),
        steps: Vec::new(),
      };
      Ok(OnGpu { encoder, decoder, arena: Mutex::new(arena) })
    };
    let cpu = || {
      let encoder = cpu::Encoder::load(g, config("encoder")?, eps)?;
      Ok(OnCpu { encoder, decoder: on_cpu::Decoder::load(g, config("decoder")?, vocab, context)? })
    };
    let backend = Backend::new(target, gpu, cpu)?;
    Ok(Self { tok, bos, eos, freqs, context, backend })
  }

  pub fn device(&self) -> &crate::Device {
    self.backend.device()
  }

  /// The linear uploads recorded while this model loaded on the GPU, then forgotten.
  #[cfg(all(test, feature = "hub"))]
  pub(crate) fn weight_bytes(&self) -> Vec<crate::gpu::WeightBytes> {
    match &self.backend {
      Backend::Gpu(b) => b.0.weight_bytes(),
      Backend::Cpu(_) | Backend::Ane(_) => Vec::new(),
    }
  }

  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    let t = frames(pcm.len())[2];
    if t == 0 {
      return Ok(Transcript::default());
    }
    let ids = match &self.backend {
      Backend::Gpu(b) => self.on_gpu(&b.0, &b.1, pcm)?,
      Backend::Cpu(on) => {
        let table = self.table(t.max(self.context));
        let enc = on.encoder.run(pcm, &table);
        let mut cache = on.decoder.cache(&enc, t);
        self
          .greedy(|token, pos| Ok(argmax(&on.decoder.step(&mut cache, token, pos, Some(&table)))))?
      }
      Backend::Ane(_) => unreachable!("Moonshine has no Neural Engine program"),
    };
    let pieces = self.tok.pieces(&ids);
    let text = sentencepiece::text(&pieces);
    let token = |(&id, piece): (&usize, String)| Token { id: id as u32, piece, start_ms: 0 };
    Ok(Transcript { text, tokens: ids.iter().zip(pieces).map(token).collect(), language: None })
  }

  /// As the reference's loop: `step(token, pos)` gives the token after `token` at `pos`, from `<s>`
  /// at 0, a step per position up to the context, the repetition guard after each token, and on a
  /// stop at the context a repeating tail trimmed.
  fn greedy(&self, mut step: impl FnMut(usize, usize) -> Result<usize>) -> Result<Vec<usize>> {
    let mut token = step(self.bos, 0)?;
    let mut ids = Vec::new();
    while token != self.eos {
      ids.push(token);
      if repeat::stop(&mut ids) {
        break;
      }

      if ids.len() == self.context {
        repeat::trim_at_budget(&mut ids);
        break;
      }

      token = step(token, ids.len())?;
    }

    Ok(ids)
  }

  fn on_gpu(&self, gpu: &Gpu, on: &OnGpu, pcm: &[f32]) -> Result<Vec<usize>> {
    let (n, t) = (pcm.len(), frames(pcm.len())[2]);
    let mut a = on.arena.lock().unwrap_or_else(PoisonError::into_inner);
    if n > a.enc.samples {
      self.grow(gpu, on, &mut a, n);
    }
    gpu.write(&a.enc.pcm, pcm);
    let Arena { enc, dec, rope, next, read, window, steps } = &mut *a;
    let (enc, dec, rope, next) = (&*enc, &*dec, &*rope, &*next);
    // Records `step` and the argmax of its logits; returns the next token.
    let run = |groups: &mut Groups, step: &dyn Fn(&mut Pass)| -> Result<usize> {
      let out = gpu.run(groups, next, read, 1, |p| {
        step(p);
        p.argmax(&dec.logits, next, on.decoder.vocab);
      })?;
      Ok(out[0].to_bits() as usize)
    };
    // The encoder, the cross K/V and the step on `<s>` in one pass. Each step reads its token
    // from `next`, where the step before left it.
    self.greedy(|_, pos| match pos {
      0 => {
        gpu.write(next, &[f32::from_bits(self.bos as u32)]);
        run(window, &|p| {
          on.encoder.record(p, enc, n, rope);
          on.decoder.record_cross(p, dec, &enc.t.y, &enc.t.h, t);
          on.decoder.record_step(p, dec, (next, 0), 0, t, Some(rope));
        })
      }
      _ => run(steps, &|p| on.decoder.record_step(p, dec, (next, 0), pos, t, Some(rope))),
    })
  }

  /// The rotary table for `positions`: per position rot / 2 cosines, then as many sines.
  fn table(&self, positions: usize) -> Vec<f32> {
    let mut table = Vec::new();
    for pos in 0..positions {
      let angles = self.freqs.iter().map(|f| pos as f32 * f);
      table.extend(angles.clone().map(f32::cos).chain(angles.map(f32::sin)));
    }

    table
  }

  /// Buffers for `n` samples, the rotary table for as many positions as they and the decoder need.
  fn grow(&self, gpu: &Gpu, on: &OnGpu, a: &mut Arena, n: usize) {
    let t = frames(n)[2];
    a.enc = on.encoder.buffers(gpu, n);
    a.dec = on.decoder.steps(gpu, t);
    let table = self.table(t.max(self.context));
    a.rope = gpu.buffer(table.len());
    gpu.write(&a.rope, &table);
    a.window.clear();
    a.steps.clear();
  }
}

/// The index of the largest logit, the lowest on a tie, as the GPU's `argmax`.
fn argmax(logits: &[f32]) -> usize {
  (0..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best })
}
