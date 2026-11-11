// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Sortformer: speaker diarization with NVIDIA's Nemotron-3-Diarization in the GGUF of
//! parakeet.cpp, as its `src/diarization.cpp` runs it. Parakeet's log-mel without the
//! normalization on the CPU; per chunk of the audio, the transformer and the head on the GPU
//! (`encoder`); the speaker cache that carries the speakers from chunk to chunk on the CPU
//! (`stream`). The output is, per 10 ms frame, the probability that each speaker talks, speakers
//! in order of their first words; a speaker's turns are the runs of at least 0.5.

mod encoder;
mod stream;

use encoder::{Encoder, STACK};
use stream::{Params, State};

use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::parakeet::frontend::{Frontend, HOP, MELS, N_FFT, PRE_EMPHASIS, WINDOW};
use crate::{Device, Turn, transformer};

/// Output frame length: the 10 ms hop.
const FRAME_MS: u32 = 10;
/// The chunking of NeMo's `diarize()` for this model, in embedding frames of 80 ms (21.12 s
/// chunks): parakeet.cpp's defaults, as its GGUF doesn't carry them.
const PARAMS: Params = Params { chunk: 264, fifo: 0, cache: 264, period: 264, silence: 1 };

pub struct Sortformer {
  gpu: Gpu,
  frontend: Frontend,
  encoder: Encoder,
  silence: Vec<f32>,
  speakers: usize,
}

impl Sortformer {
  pub fn load(g: &Gguf) -> Result<Self> {
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
    let frames = (PARAMS.cache + PARAMS.fifo + PARAMS.chunk).next_multiple_of(64);
    let gpu = Gpu::new(true, None)?;
    let shape = [u("diar.tf_d_model")?, speakers, frames];
    let encoder = Encoder::new(&gpu, g, cfg, shape, theta)?;
    Ok(Self { gpu, frontend: Frontend::new(), encoder, silence, speakers })
  }

  pub fn device(&self) -> &Device {
    self.gpu.info()
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
      let embs = self.encoder.embed(&self.gpu, &self.frontend.log_mel(pcm, start..end))?;
      let x: Vec<f32> = state.context().chain(&embs).copied().collect();
      let out = self.encoder.predict(&self.gpu, &x)?;
      // The chunk's frames follow the context's; past the last mel frame are the padding's.
      let context = (x.len() - embs.len()) / d;
      probs.extend_from_slice(&out[context * STACK * n..][..(end - start) * n]);
      if end < frames {
        // Per embedding, the mean of its 8 frames.
        let mean = |f: &[f32], s: usize| f.iter().skip(s).step_by(n).sum::<f32>() / STACK as f32;
        let preds: Vec<f32> =
          out.chunks(STACK * n).flat_map(|f| (0..n).map(move |s| mean(f, s))).collect();
        state.update(&PARAMS, &embs, &preds);
      }
    }

    Ok(probs)
  }
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
