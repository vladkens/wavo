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
  /// RoPE base: GigaAM passes `pos_emb_max_len` (5000) as theta.
  theta: f64,
}

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
    let check_str = |key: &str, allowed: &[&str]| -> Result<()> {
      let got = g.str(key)?;
      if !allowed.contains(&got) {
        bail!("{key} is {got:?}, only {allowed:?} supported");
      }
      Ok(())
    };
    let check_u32 = |key: &str, want: u32| -> Result<()> {
      let got = g.u32(key)?;
      if got != want {
        bail!("{key} is {got}, only {want} is supported");
      }
      Ok(())
    };
    check_str("general.architecture", &["gigaam"])?;
    check_str("stt.gigaam.encoder.self_attention_model", &["rotary"])?;
    check_str("stt.gigaam.encoder.conv_norm_type", &["layer_norm"])?;
    check_str("tokenizer.ggml.model", &["bpe", "char"])?;
    check_u32("stt.gigaam.encoder.conv_kernel", 5)?;
    check_u32("stt.gigaam.encoder.subs_kernel_size", 5)?;
    check_u32("stt.gigaam.encoder.subsampling_factor", 4)?;
    check_u32("stt.frontend.sample_rate", 16000)?;
    check_u32("stt.frontend.n_fft", frontend::N_FFT as u32)?;
    check_u32("stt.frontend.win_length", frontend::N_FFT as u32)?;
    check_u32("stt.frontend.hop_length", 160)?;
    let u = |key: &str| g.u32(&format!("stt.gigaam.{key}")).map(|v| v as usize);
    let cfg = Config {
      layers: u("encoder.n_layers")?,
      d: u("encoder.d_model")?,
      heads: u("encoder.n_heads")?,
      d_ff: u("encoder.d_ff")?,
      mels: g.u32("stt.frontend.num_mels")? as usize,
      theta: u("encoder.pos_emb_max_len")? as f64,
    };
    let (d, head_dim) = (cfg.d, cfg.d / cfg.heads);
    if !d.is_multiple_of(cfg.heads) || d > 1024 || head_dim > 128 || !head_dim.is_multiple_of(2) {
      bail!("unsupported attention shape: d_model {d}, {} heads", cfg.heads);
    }
    let vocab: Vec<String> =
      g.strs("tokenizer.ggml.tokens")?.into_iter().map(String::from).collect();
    let blank = g.u32("tokenizer.ggml.blank_token_id")? as usize;
    let kind = g.str("stt.gigaam.head_kind")?;
    let classes = match kind {
      "rnnt" => u("joint.num_classes")?,
      "ctc" => u("head.num_classes")?,
      _ => bail!("stt.gigaam.head_kind is {kind:?}, only \"rnnt\" and \"ctc\" are supported"),
    };
    if vocab.len() != classes || blank >= classes {
      bail!("vocabulary of {} tokens, {classes} classes, blank {blank}", vocab.len());
    }

    let gpu = Gpu::new()?;
    let frontend = Frontend::new(g, cfg.mels)?;
    let (head, encoder) = if kind == "rnnt" {
      check_str("stt.gigaam.joint.activation", &["relu"])?;
      check_u32("stt.gigaam.predictor.n_layers", 1)?;
      check_u32("stt.gigaam.predictor.vocab", classes as u32)?;
      let joint = u("joint.hidden")?;
      let decoder = Decoder::new(g, u("predictor.hidden")?, joint, classes, blank)?;
      (Head::Rnnt(Box::new(decoder)), Encoder::new(&gpu, g, cfg, ("joint.enc", &[d, joint]))?)
    } else {
      check_u32("stt.gigaam.head.feat_in", d as u32)?;
      (Head::Ctc { blank }, Encoder::new(&gpu, g, cfg, ("head.ctc", &[1, d, classes]))?)
    };
    // The GPU's first use of the pipelines and the weight memory costs 20–40 ms whatever the input
    // length: pay it here on 8 silent frames instead of in the first call.
    encoder.run(&gpu, &vec![0.0; 8 * frontend.mels])?;
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
      .map(|(id, frame)| Token { id, piece: self.vocab[id as usize].clone(), frame })
      .collect();
    // Character vocabularies have a plain space and no `▁`.
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

/// `[frame][row]` dots of the `k`-long rows of `w` (a multiple of 4 rows) with the `k`-long frames
/// of `z`, each equal to `dot(row, frame)`. Frames go in pairs, two rows at a time; a last single
/// frame goes four rows at a time. Either way each loaded chunk serves several dots.
fn matmul(w: &[f32], z: &[f32], k: usize) -> Vec<f32> {
  let rows = w.len() / k;
  let mut out = vec![0.0; z.len() / k * rows];
  let pairs = z.chunks_exact(2 * k);
  let last = pairs.remainder();
  for (z, out) in pairs.zip(out.chunks_exact_mut(2 * rows)) {
    for (r, w) in w.chunks_exact(2 * k).enumerate() {
      let [d0, d1] = tile::<2, 2>(w, z, k);
      out[2 * r..][..2].copy_from_slice(&d0);
      out[rows + 2 * r..][..2].copy_from_slice(&d1);
    }
  }
  if !last.is_empty() {
    let out = out.rchunks_exact_mut(rows).next().unwrap();
    for (w, out) in w.chunks_exact(4 * k).zip(out.as_chunks_mut::<4>().0) {
      [*out] = tile::<4, 1>(w, last, k);
    }
  }
  out
}

/// `dot` of `R` rows of `w` with `F` frames of `z` (`[frame][row]`), loading each chunk once. It
/// keeps `dot`'s lanes, fused multiply-adds and pairwise sum, so the results are identical.
#[cfg(target_arch = "aarch64")]
fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
  use std::arch::aarch64::*;
  let (w, z, n) = (&w[..R * k], &z[..F * k], k / 16 * 16);
  // SAFETY: NEON is part of aarch64, and every load reads 4 floats below `n` ≤ k of a row of `w`
  // or a frame of `z`.
  let acc = unsafe {
    let mut acc = [[[vdupq_n_f32(0.0); 4]; R]; F];
    for i in (0..n).step_by(16) {
      for c in 0..4 {
        let wv: [_; R] = std::array::from_fn(|r| vld1q_f32(w.as_ptr().add(r * k + i + 4 * c)));
        for (f, acc) in acc.iter_mut().enumerate() {
          let zv = vld1q_f32(z.as_ptr().add(f * k + i + 4 * c));
          for (a, wv) in acc.iter_mut().zip(wv) {
            a[c] = vfmaq_f32(a[c], wv, zv);
          }
        }
      }
    }
    acc.map(|acc| {
      acc.map(|[a0, a1, a2, a3]| {
        let lanes = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
        let pair = vadd_f32(vget_low_f32(lanes), vget_high_f32(lanes));
        vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair)
      })
    })
  };
  std::array::from_fn(|f| {
    std::array::from_fn(|r| {
      let (w, z) = (&w[r * k + n..][..k - n], &z[f * k + n..][..k - n]);
      acc[f][r] + w.iter().zip(z).map(|(x, y)| x * y).sum::<f32>()
    })
  })
}

#[cfg(not(target_arch = "aarch64"))]
fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
  std::array::from_fn(|f| std::array::from_fn(|r| dot(&w[r * k..][..k], &z[f * k..][..k])))
}

#[cfg(test)]
mod tests {
  #[test]
  fn matmul_matches_dot() {
    let (k, rows, frames) = (333, 8, 5);
    let v: Vec<f32> =
      (0..(rows + frames) * k).map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0).collect();
    let (z, w) = v.split_at(frames * k);
    let want: Vec<u32> = (z.chunks_exact(k))
      .flat_map(|z| w.chunks_exact(k).map(|r| super::dot(r, z).to_bits()))
      .collect();
    let got: Vec<u32> = super::matmul(w, z, k).into_iter().map(f32::to_bits).collect();
    assert_eq!(got, want);
  }
}
