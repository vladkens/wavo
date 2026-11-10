// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Sortformer: speaker diarization, as transcribe.cpp's `src/arch/sortformer/` runs NVIDIA's
//! streaming Sortformer. Parakeet's log-mel without the normalization on the CPU; per chunk of the
//! audio, Parakeet's subsampling and the FastConformer, a transformer and a sigmoid head per
//! speaker on the GPU (`encoder`); the speaker cache and FIFO that carry the speakers from chunk to
//! chunk on the CPU (`stream`). The output is, per 80 ms frame, the probability that each speaker
//! talks, speakers in order of their first words; a speaker's turns are the runs above 0.5.

mod encoder;
mod stream;

use encoder::Encoder;
use stream::{Params, State};

use crate::conformer::{self, Attention};
use crate::error::{Result, bail};
use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::parakeet::frontend::{Frontend, HOP, MELS, N_FFT, PRE_EMPHASIS, WINDOW};
use crate::{Device, Turn};

/// Mel frames per output frame.
const SUBSAMPLING: usize = 8;
/// Output frame length: 10 ms hop × subsampling 8.
const FRAME_MS: u32 = 80;

pub struct Sortformer {
  gpu: Gpu,
  frontend: Frontend,
  encoder: Encoder,
  params: Params,
  speakers: usize,
}

impl Sortformer {
  pub fn load(g: &Gguf) -> Result<Self> {
    for (key, want) in [
      ("general.architecture", "sortformer"),
      ("stt.frontend.window", "hann"),
      ("stt.frontend.normalize", "none"),
      ("stt.sortformer.transformer.activation", "relu"),
    ] {
      g.check(key, &[want])?;
    }
    for (key, want) in [
      ("stt.frontend.sample_rate", 16000),
      ("stt.frontend.num_mels", MELS),
      ("stt.frontend.n_fft", N_FFT),
      ("stt.frontend.win_length", WINDOW),
      ("stt.frontend.hop_length", HOP),
      ("stt.sortformer.frame_hop", HOP * SUBSAMPLING),
      ("stt.sortformer.encoder.subsampling_factor", SUBSAMPLING),
      ("stt.sortformer.encoder.feat_in", MELS),
    ] {
      g.check(key, &[want as u32])?;
    }
    g.check("stt.frontend.pre_emphasis", &[PRE_EMPHASIS])?;
    g.check("stt.sortformer.transformer.pre_ln", &[false])?;

    let u = |key: &str| g.get::<u32>(&format!("stt.sortformer.{key}")).map(|v| v as usize);
    let speakers = u("max_speakers")?;
    let params = Params {
      chunk: u("stream.chunk_len")?,
      left: 1,
      right: 1,
      fifo: u("stream.fifo_len")?,
      cache: u("stream.spkcache_len")?,
      period: u("stream.spkcache_update_period")?,
    };
    // Each speaker keeps some frames of the compressed cache, past those of silence.
    if speakers == 0 || params.chunk == 0 || params.cache / speakers <= 3 {
      bail!("unsupported stream: {speakers} speakers, cache {}", params.cache);
    }

    let enc = conformer::Config::load(g, "sortformer", Attention::Relative, true)?;
    let ch = u("encoder.subsampling_channels")?;
    let tf = encoder::Config {
      layers: u("transformer.n_layers")?,
      d: u("transformer.d_model")?,
      heads: u("transformer.n_heads")?,
      d_ff: u("transformer.d_ff")?,
    };
    let gpu = Gpu::new(true)?;
    let encoder = Encoder::new(&gpu, g, enc, ch, tf)?;
    Ok(Self { gpu, frontend: Frontend::new(), encoder, params, speakers })
  }

  pub fn device(&self) -> &Device {
    self.gpu.info()
  }

  pub fn speakers(&self) -> usize {
    self.speakers
  }

  /// The turns of 16 kHz PCM, by start then speaker.
  pub fn diarize(&self, pcm: &[f32]) -> Result<Vec<Turn>> {
    let probs = self.probabilities(pcm)?;
    Ok(turns(&probs, self.speakers))
  }

  /// Per 80 ms frame, `⌈pcm.len() / 1280⌉` of them, each speaker's probability.
  fn probabilities(&self, pcm: &[f32]) -> Result<Vec<f32>> {
    // ⌈n / hop⌉ frames, those from ⌊n / hop⌋ on zero (NeMo's newer frame count).
    let frames = pcm.len().div_ceil(HOP);
    if frames < 2 {
      return Ok(Vec::new());
    }

    let mut mel = self.frontend.log_mel(pcm, frames);
    mel[pcm.len() / HOP * MELS..].fill(0.0);

    let (p, d, sub) = (self.params, self.encoder.width(), SUBSAMPLING);
    let window = p.left + p.chunk + p.right;
    self.encoder.reserve(&self.gpu, window.max(p.cache + p.fifo + window));
    let scale = (d as f32).sqrt();
    let mut state = State::new(d, self.speakers);
    let (mut start, mut end) = (0, 0);
    while end < frames {
      // A chunk's mel frames with up to `left` and `right` output frames of context.
      let left = (p.left * sub).min(start);
      end = (start + p.chunk * sub).min(frames);
      let right = (p.right * sub).min(frames - end);
      let embs =
        self.encoder.embed(&self.gpu, &mel[(start - left) * MELS..(end + right) * MELS])?;
      let x: Vec<f32> = state.context().chain(&embs).map(|v| v * scale).collect();
      let logits = self.encoder.predict(&self.gpu, &x, self.speakers)?;
      let preds: Vec<f32> = logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
      let context = [(left + sub / 2) / sub, right.div_ceil(sub)];
      state.update(&p, &embs, &preds, context);
      start = end;
    }

    let mut probs = state.preds;
    probs.truncate(frames.div_ceil(sub) * self.speakers);
    Ok(probs)
  }
}

/// Per speaker, each run of frames with probability above 0.5, sorted by start then speaker.
fn turns(probs: &[f32], speakers: usize) -> Vec<Turn> {
  let frames = probs.len() / speakers;
  let mut out = Vec::new();
  for speaker in 0..speakers {
    let mut run = None;
    for t in 0..=frames {
      let on = t < frames && probs[t * speakers + speaker] > 0.5;
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
    // Three frames of two speakers: 0 talks in frames 0–1, 1 in frames 1 and 3.
    let probs = [0.9, 0.1, 0.6, 0.7, 0.5, 0.2, 0.1, 0.51];
    let turns: Vec<_> =
      super::turns(&probs, 2).iter().map(|t| (t.speaker, t.start_ms, t.end_ms)).collect();
    assert_eq!(turns, [(0, 0, 160), (1, 80, 160), (1, 240, 320)]);
    assert!(super::turns(&[], 4).is_empty());
  }
}
