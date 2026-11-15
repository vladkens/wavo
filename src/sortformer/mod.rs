// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Sortformer: speaker diarization with NVIDIA's Nemotron-3-Diarization in the GGUF of
//! parakeet.cpp, as its `src/diarization.cpp` runs it. Parakeet's log-mel without the
//! normalization on the CPU; per chunk of the audio, the transformer and the head on the GPU
//! (`encoder`) or the CPU (`cpu`); the speaker cache that carries the speakers from chunk to chunk
//! on the CPU (`stream`). The output is, per 10 ms frame, the probability that each speaker talks,
//! speakers in order of their first words; a speaker's turns are the runs of at least 0.5.

mod cpu;
mod encoder;
mod stream;

use encoder::{Encoder, STACK};
use stream::{Params, State};

use crate::ane::{Backend, Target};
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::parakeet::frontend::{Frontend, HOP, MELS, N_FFT, PRE_EMPHASIS, WINDOW};
use crate::{Device, Turn, transformer};

/// Output frame length: the 10 ms hop.
const FRAME_MS: u32 = 10;
/// The chunking of NeMo's `diarize()` for this model, in embedding frames of 80 ms: the model
/// card's 30.4 s latency config (27.2 s chunks, 3.2 s ahead), as the GGUF doesn't carry it.
/// parakeet.cpp's 21.12 s chunks with nothing ahead give a DER of 12.3% instead of 11.7% on AMI's
/// test meetings (SDM, forced-alignment labels) and 6.0% instead of 5.7% on VoxConverse dev.
const PARAMS: Params =
  Params { chunk: 340, right: 40, fifo: 40, cache: 264, period: 300, silence: 1 };

pub struct Sortformer {
  frontend: Frontend,
  backend: Backend<Encoder, cpu::Encoder>,
  silence: Vec<f32>,
  speakers: usize,
}

impl Sortformer {
  /// The model of `g` on the GPU or the CPU.
  pub fn load(g: &Gguf, target: Target) -> Result<Self> {
    for (key, want) in [
      ("parakeet.arch", "diarization"),
      ("parakeet.preprocessor.normalize", "NA"),
      ("parakeet.encoder.self_attention_model", "rope"),
    ] {
      g.check(key, &[want])?;
    }

    for (key, want) in [
      ("parakeet.preprocessor.sample_rate", 16000),
      ("parakeet.preprocessor.n_mels", MELS),
      ("parakeet.preprocessor.n_fft", N_FFT),
      ("parakeet.preprocessor.win_length", WINDOW),
      ("parakeet.preprocessor.hop_length", HOP),
      ("parakeet.encoder.feat_in", MELS),
      ("parakeet.encoder.subsampling_factor", STACK),
      ("parakeet.diar.upsample_factor", STACK),
    ] {
      g.check(key, &[want as u32])?;
    }

    g.check("parakeet.preprocessor.preemph", &[PRE_EMPHASIS])?;
    g.check("parakeet.encoder.rotary_fraction", &[1.0f32])?;
    g.check("parakeet.encoder.pre_block_norm", &[true])?;
    g.check("parakeet.encoder.qkv_bias", &[false])?;

    let u = |key: &str| g.get::<u32>(&format!("parakeet.{key}")).map(|v| v as usize);
    let speakers = u("diar.n_speakers")?;
    let (d, heads) = (u("encoder.d_model")?, u("encoder.n_heads")?);
    let cfg = transformer::Config {
      d,
      heads,
      layers: u("encoder.n_layers")?,
      d_ff: u("encoder.ff_dim")?,
      bias: true,
      rot: d / heads,
      neox: true,
      gated: false,
    };
    // Kernels keep a row of up to 1024 channels in shared memory, and heads in steps of 4.
    if d > 1024 || !d.is_multiple_of(heads) || !cfg.head_dim().is_multiple_of(4) {
      bail!("unsupported transformer shape: d_model {d}, {heads} heads");
    }
    if speakers == 0 || PARAMS.cache / speakers <= PARAMS.silence {
      bail!("unsupported speaker cache for {speakers} speakers");
    }

    let silence = g.tensor("sortformer_modules.learnable_sil_emb", &[d])?.to_f32()?;
    let theta = g.get::<f32>("parakeet.encoder.rope_base")?;
    // Flash attention reads rows up to a multiple of 64.
    let frames = (PARAMS.cache + PARAMS.fifo + PARAMS.chunk + PARAMS.right).next_multiple_of(64);
    let table = rope(frames, cfg.head_dim(), theta);
    let width = u("diar.tf_d_model")?;
    let gpu = |gpu: &Gpu| Encoder::new(gpu, g, cfg, [width, speakers, frames], &table);
    let cpu = || cpu::Encoder::new(g, cfg, width, speakers, table.clone());
    let backend = Backend::new(target, gpu, cpu)?;
    Ok(Self { frontend: Frontend::new(), backend, silence, speakers })
  }

  pub fn device(&self) -> &Device {
    self.backend.device()
  }

  /// The turns of 16 kHz PCM, by start then speaker.
  pub fn diarize(&self, pcm: &[f32]) -> Result<Vec<Turn>> {
    let probs = self.probabilities(pcm)?;
    Ok(turns(&probs, self.speakers))
  }

  /// Per 10 ms frame, `pcm.len() / 160` of them, each speaker's probability.
  fn probabilities(&self, pcm: &[f32]) -> Result<Vec<f32>> {
    let (frames, n, d) = (pcm.len() / HOP, self.speakers, self.silence.len());
    let mut state = State::new(n, self.silence.clone());
    let mut probs = Vec::with_capacity(frames * n);
    for start in (0..frames).step_by(PARAMS.chunk * STACK) {
      let end = (start + PARAMS.chunk * STACK).min(frames);
      let ahead = (end + PARAMS.right * STACK).min(frames);
      let mel = self.frontend.log_mel(pcm, start..ahead);
      let embs = match &self.backend {
        Backend::Gpu(b) => b.1.embed(&b.0, &mel)?,
        Backend::Cpu(encoder) => encoder.embed(&mel),
        Backend::Ane(_) => unreachable!("the diarizer has no Neural Engine program"),
      };
      let x: Vec<f32> = state.context().chain(&embs).copied().collect();
      let out = match &self.backend {
        Backend::Gpu(b) => b.1.predict(&b.0, &x)?,
        Backend::Cpu(encoder) => encoder.predict(&x),
        Backend::Ane(_) => unreachable!("the diarizer has no Neural Engine program"),
      };
      // The chunk's frames follow the context's, and those ahead the chunk's; past the last mel
      // frame are the padding's.
      let context = (x.len() - embs.len()) / d;
      probs.extend_from_slice(&out[context * STACK * n..][..(end - start) * n]);
      if end < frames {
        // Per embedding, the mean of its 8 frames.
        let mean = |f: &[f32], s: usize| f.iter().skip(s).step_by(n).sum::<f32>() / STACK as f32;
        let rows = context + PARAMS.chunk;
        let preds: Vec<f32> = (out.chunks(STACK * n).take(rows))
          .flat_map(|f| (0..n).map(move |s| mean(f, s)))
          .collect();
        state.update(&PARAMS, &embs[..PARAMS.chunk * d], &preds);
      }
    }

    Ok(probs)
  }
}

/// The rotary table from position 0 for `frames` positions: per position `hd` / 2 cosines of
/// `pos · theta^(−2i / hd)`, then as many sines.
fn rope(frames: usize, hd: usize, theta: f32) -> Vec<f32> {
  let mut table = Vec::with_capacity(frames * hd);
  for pos in 0..frames {
    let angles = (0..hd / 2).map(|i| pos as f32 * theta.powf(-((2 * i) as f32) / hd as f32));
    table.extend(angles.clone().map(f32::cos).chain(angles.map(f32::sin)));
  }

  table
}

/// Per speaker, each run of frames with probability of at least 0.5, sorted by start then speaker.
fn turns(probs: &[f32], speakers: usize) -> Vec<Turn> {
  let frames = probs.len() / speakers;
  let mut out = Vec::new();
  for speaker in 0..speakers {
    let mut run = None;
    for t in 0..=frames {
      let on = t < frames && probs[t * speakers + speaker] >= 0.5;
      match (on, run) {
        (true, None) => run = Some(t),
        (false, Some(first)) => {
          let [start_ms, end_ms] = [first, t].map(|f| f as u32 * FRAME_MS);
          out.push(Turn { speaker: speaker as u32, start_ms, end_ms });
          run = None;
        }
        _ => {}
      }
    }
  }

  out.sort_by_key(|t| (t.start_ms, t.speaker));
  out
}

#[cfg(test)]
mod tests {
  #[test]
  fn turns() {
    // Four frames of two speakers: 0 talks in frames 0–2, 1 in frames 1 and 3.
    let probs = [0.9, 0.1, 0.6, 0.7, 0.5, 0.2, 0.1, 0.51];
    let turns: Vec<_> =
      super::turns(&probs, 2).iter().map(|t| (t.speaker, t.start_ms, t.end_ms)).collect();
    assert_eq!(turns, [(0, 0, 30), (1, 10, 20), (1, 30, 40)]);
    assert!(super::turns(&[], 4).is_empty());
  }
}
