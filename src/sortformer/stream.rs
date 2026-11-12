// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The streaming state on the CPU, as parakeet.cpp's `src/diarization_streaming.cpp` ports
//! NeMo's `SortformerModules` (batch 1): after each chunk its frames go through a FIFO into the
//! speaker cache, which is compressed to the frames that best represent each speaker (the
//! Arrival-Order Speaker Cache), with the model's silence embedding in place of empty slots.
//! Embeddings are pre-encode outputs `[frames][d]`, predictions `[frames][speakers]`.

/// Predictions are clamped to at least this before their logarithms.
const PRED_SCORE_THRESHOLD: f32 = 0.25;
/// Added to the scores of the frames that arrived since the last compression.
const SCORES_BOOST_LATEST: f32 = 0.05;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;

/// The chunking, in embedding frames.
pub struct Params {
  /// Frames per chunk.
  pub chunk: usize,
  /// Frames of the FIFO and of the speaker cache.
  pub fifo: usize,
  pub cache: usize,
  /// Frames moved from the FIFO to the cache at a time.
  pub period: usize,
  /// Frames of each speaker's share of the compressed cache kept for silence.
  pub silence: usize,
}

pub struct State {
  /// Embedding size and speakers.
  d: usize,
  n: usize,
  cache: Vec<f32>,
  /// Predictions of the cache's frames, kept from the first compression on.
  cache_preds: Option<Vec<f32>>,
  fifo: Vec<f32>,
  fifo_preds: Vec<f32>,
  /// The embedding of an empty slot.
  silence: Vec<f32>,
}

impl State {
  pub fn new(n: usize, silence: Vec<f32>) -> Self {
    let (cache, fifo, fifo_preds) = (Vec::new(), Vec::new(), Vec::new());
    Self { d: silence.len(), n, cache, cache_preds: None, fifo, fifo_preds, silence }
  }

  /// The speaker cache then the FIFO, `[frames][d]`: the rows before a chunk's.
  pub fn context(&self) -> impl Iterator<Item = &f32> {
    self.cache.iter().chain(&self.fifo)
  }

  /// Takes a chunk's embeddings `[c][d]` and the predictions `[context + c][n]` of the forward
  /// over the context and them: `streaming_update` (sync).
  pub fn update(&mut self, p: &Params, embs: &[f32], preds: &[f32]) {
    let (d, n) = (self.d, self.n);
    let (s, f) = (self.cache.len() / d, self.fifo.len() / d);
    let c = embs.len() / d;
    self.fifo_preds = preds[s * n..].to_vec();
    self.fifo.extend_from_slice(embs);

    if f + c > p.fifo {
      let pop = p.period.max(f + c - p.fifo).min(f + c);
      let pop_embs: Vec<f32> = self.fifo.drain(..pop * d).collect();
      let pop_preds: Vec<f32> = self.fifo_preds.drain(..pop * n).collect();
      self.cache.extend_from_slice(&pop_embs);
      if let Some(cache_preds) = &mut self.cache_preds {
        cache_preds.extend_from_slice(&pop_preds);
      }

      if self.cache.len() / d > p.cache {
        // The first compression takes the cache's predictions from this forward.
        self.cache_preds.get_or_insert_with(|| [&preds[..s * n], &pop_preds].concat());
        self.compress(p.cache, p.silence);
      }
    }
  }

  /// `_compress_spkcache`: the cache down to `len` frames, per speaker its best-scored ones and
  /// `silence` of silence, in time order.
  fn compress(&mut self, len: usize, silence: usize) {
    let (d, n) = (self.d, self.n);
    let preds = self.cache_preds.take().unwrap();
    let frames = preds.len() / n;
    let per_speaker = (len / n - silence) as f32;
    let [strong, weak, min_pos] = [STRONG_BOOST_RATE, WEAK_BOOST_RATE, MIN_POS_SCORES_RATE]
      .map(|rate| (per_speaker * rate).floor() as usize);

    let mut scores = log_pred_scores(&preds, n);
    disable_low_scores(&preds, &mut scores, n, min_pos);
    // The frames since the last compression; -inf stays -inf.
    scores[len * n..].iter_mut().for_each(|s| *s += SCORES_BOOST_LATEST);
    boost_top(&mut scores, n, strong, 2.0);
    boost_top(&mut scores, n, weak, 1.0);

    // Speaker-major with `silence` frames of +inf after each speaker's, the best `len` (by score,
    // then index), in index order; -inf picks and the silence frames are empty.
    let padded = frames + silence;
    let score = |f: usize| match f % padded {
      i if i < frames => scores[i * n + f / padded],
      _ => f32::INFINITY,
    };
    let mut flat: Vec<usize> = (0..n * padded).collect();
    flat.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap().then(a.cmp(&b)));
    let mut picks: Vec<Option<usize>> = (flat.iter().take(len))
      .map(|&f| (score(f) != f32::NEG_INFINITY).then_some(f))
      .chain(std::iter::repeat(None))
      .take(len)
      .collect();
    picks.sort_by_key(|p| p.unwrap_or(usize::MAX));

    let (mut cache, mut kept) = (Vec::with_capacity(len * d), Vec::with_capacity(len * n));
    for pick in picks {
      match pick.map(|f| f % padded).filter(|&i| i < frames) {
        Some(i) => {
          cache.extend_from_slice(&self.cache[i * d..(i + 1) * d]);
          kept.extend_from_slice(&preds[i * n..(i + 1) * n]);
        }
        None => {
          cache.extend_from_slice(&self.silence);
          kept.extend(std::iter::repeat_n(0.0, n));
        }
      }
    }

    self.cache = cache;
    self.cache_preds = Some(kept);
  }
}

/// `_get_log_pred_scores`: per frame and speaker `ln p − ln(1 − p) + Σ ln(1 − p) − ln ½`, each p
/// and 1 − p clamped to at least `PRED_SCORE_THRESHOLD`.
fn log_pred_scores(preds: &[f32], n: usize) -> Vec<f32> {
  let ln = |x: f32| x.max(PRED_SCORE_THRESHOLD).ln();
  let half = 0.5f32.ln();
  let mut scores = Vec::with_capacity(preds.len());
  for row in preds.chunks(n) {
    let rest = row.iter().fold(0.0f32, |s, &p| s + ln(1.0 - p));
    scores.extend(row.iter().map(|&p| ln(p) - ln(1.0 - p) + rest - half));
  }

  scores
}

/// `_disable_low_scores`: non-speech (p ≤ 0.5) to -inf, and for a speaker with at least `min_pos`
/// positive scores, its speech frames that aren't positive too.
fn disable_low_scores(preds: &[f32], scores: &mut [f32], n: usize, min_pos: usize) {
  for (s, p) in scores.iter_mut().zip(preds) {
    if *p <= 0.5 {
      *s = f32::NEG_INFINITY;
    }
  }

  for k in 0..n {
    let positive = scores.iter().skip(k).step_by(n).filter(|&&s| s > 0.0).count();
    if positive >= min_pos {
      for (s, p) in scores.iter_mut().zip(preds).skip(k).step_by(n) {
        if *p > 0.5 && *s <= 0.0 {
          *s = f32::NEG_INFINITY;
        }
      }
    }
  }
}

/// `_boost_topk_scores`: per speaker, the `k` best frames (the lower frame first on a tie) get
/// `−scale · ln ½` more.
fn boost_top(scores: &mut [f32], n: usize, k: usize, scale: f32) {
  let delta = scale * 0.5f32.ln();
  let frames = scores.len() / n;
  for s in 0..n {
    let mut order: Vec<usize> = (0..frames).collect();
    let score = |i: usize| scores[i * n + s];
    order.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap().then(a.cmp(&b)));
    for i in order.into_iter().take(k) {
      scores[i * n + s] -= delta;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn scores_and_boosts() {
    // One frame, two speakers: p = 0.9 and 0.1, each p and 1 − p clamped to at least 0.25.
    let s = log_pred_scores(&[0.9, 0.1], 2);
    let ln = |x: f32| x.max(0.25).ln();
    let rest = ln(1.0 - 0.9) + ln(1.0 - 0.1);
    let want = |p: f32| ln(p) - ln(1.0 - p) + rest - 0.5f32.ln();
    assert_eq!(s, [want(0.9), want(0.1)]);

    // Non-speech goes; speech that isn't positive goes once a speaker has enough positives.
    let preds = [0.9, 0.2, 0.6, 0.9, 0.9, 0.4];
    let mut scores = [1.0, 1.0, -1.0, 2.0, 3.0, 1.0];
    disable_low_scores(&preds, &mut scores, 2, 2);
    let inf = f32::NEG_INFINITY;
    assert_eq!(scores, [1.0, inf, inf, 2.0, 3.0, inf]);
    let mut scores = [1.0, 1.0, -1.0, 2.0, 3.0, 1.0];
    disable_low_scores(&preds, &mut scores, 2, 3);
    assert_eq!(scores, [1.0, inf, -1.0, 2.0, 3.0, inf]);

    // The top two per speaker, ties to the lower frame.
    let mut scores = [1.0, 5.0, 3.0, 5.0, 3.0, inf];
    boost_top(&mut scores, 2, 2, 1.0);
    let b = -0.5f32.ln();
    assert_eq!(scores, [1.0, 5.0 + b, 3.0 + b, 5.0 + b, 3.0 + b, inf]);
  }

  #[test]
  fn fifo_cache_and_compression() {
    let p = Params { chunk: 4, fifo: 2, cache: 5, period: 4, silence: 3 };
    let mut st = State::new(1, vec![9.0]);
    // Frame i has embedding i; the speaker talks in even frames.
    let preds = |frames: std::ops::Range<usize>| -> Vec<f32> {
      frames.map(|i| if i % 2 == 0 { 0.9 } else { 0.0 }).collect()
    };
    let embs = |frames: std::ops::Range<usize>| frames.map(|i| i as f32).collect::<Vec<_>>();

    // Past the FIFO's 2 frames, 4 move to the cache.
    st.update(&p, &embs(0..4), &preds(0..4));
    assert_eq!((st.cache.clone(), st.fifo.len()), (vec![0.0, 1.0, 2.0, 3.0], 0));
    assert!(st.cache_preds.is_none());

    // The forward saw [cache | chunk]. 8 frames pass the cache's 5: per speaker 5 − 3 = 2 frames
    // stay, the best after boosts (6, the latest, then 0 of the equal 0, 2, 4), then 3 of silence.
    assert_eq!(st.context().count(), 4);
    st.update(&p, &embs(4..8), &preds(0..8));
    assert_eq!(st.cache, [0.0, 6.0, 9.0, 9.0, 9.0]);
    assert_eq!(st.cache_preds.as_deref(), Some(&[0.9, 0.9, 0.0, 0.0, 0.0][..]));
  }
}
