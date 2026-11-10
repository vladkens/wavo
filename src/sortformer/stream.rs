// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The streaming state on the CPU, as transcribe.cpp's `src/arch/sortformer/stream.cpp` ports
//! NeMo's `SortformerModules` (batch 1): after each chunk its frames go through a FIFO into the
//! speaker cache, which is compressed to the frames that best represent each speaker (the
//! Arrival-Order Speaker Cache), with the mean embedding of silent frames in place of empty
//! slots. Embeddings are pre-encode outputs `[frames][d]`, predictions `[frames][speakers]`.

/// Frames of each speaker's share of the compressed cache kept for silence.
pub const SIL_FRAMES_PER_SPEAKER: usize = 3;
/// A frame whose predictions sum below this is silence.
const SIL_THRESHOLD: f32 = 0.2;
/// Predictions are clamped to at least this before their logarithms.
const PRED_SCORE_THRESHOLD: f32 = 0.25;
/// Added to the scores of the frames that arrived since the last compression.
const SCORES_BOOST_LATEST: f32 = 0.05;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;

/// The chunking, in 80 ms frames: the GGUF's `stt.sortformer.stream.*`.
#[derive(Clone, Copy)]
pub struct Params {
  /// Frames per chunk.
  pub chunk: usize,
  /// Frames of the FIFO and of the speaker cache.
  pub fifo: usize,
  pub cache: usize,
  /// Frames moved from the FIFO to the cache at a time.
  pub period: usize,
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
  /// The mean embedding of the silent frames seen so far, and their count.
  sil_mean: Vec<f32>,
  sil_frames: u64,
  /// The predictions of the chunks so far.
  pub preds: Vec<f32>,
}

impl State {
  pub fn new(d: usize, n: usize) -> Self {
    Self {
      d,
      n,
      cache: Vec::new(),
      cache_preds: None,
      fifo: Vec::new(),
      fifo_preds: Vec::new(),
      sil_mean: vec![0.0; d],
      sil_frames: 0,
      preds: Vec::new(),
    }
  }

  /// The speaker cache then the FIFO, `[frames][d]`: the rows before a chunk's.
  pub fn context(&self) -> impl Iterator<Item = &f32> {
    self.cache.iter().chain(&self.fifo)
  }

  /// Takes a chunk's embeddings `[lc + c + rc][d]` and the predictions `[context + chunk][n]` of
  /// the forward over the context and them: `streaming_update` (sync).
  pub fn update(&mut self, p: &Params, embs: &[f32], preds: &[f32], [lc, rc]: [usize; 2]) {
    let (d, n) = (self.d, self.n);
    let (s, f) = (self.cache.len() / d, self.fifo.len() / d);
    let c = embs.len() / d - lc - rc;
    self.fifo_preds = preds[s * n..(s + f) * n].to_vec();
    let chunk_preds = &preds[(s + f + lc) * n..(s + f + lc + c) * n];
    self.fifo.extend_from_slice(&embs[lc * d..(lc + c) * d]);
    self.fifo_preds.extend_from_slice(chunk_preds);

    if f + c > p.fifo {
      let pop = p.period.max(f + c - p.fifo).min(f + c);
      let pop_embs: Vec<f32> = self.fifo.drain(..pop * d).collect();
      let pop_preds: Vec<f32> = self.fifo_preds.drain(..pop * n).collect();
      self.silence(&pop_embs, &pop_preds);
      self.cache.extend_from_slice(&pop_embs);
      if let Some(cache_preds) = &mut self.cache_preds {
        cache_preds.extend_from_slice(&pop_preds);
      }

      if self.cache.len() / d > p.cache {
        // The first compression takes the cache's predictions from this forward.
        self.cache_preds.get_or_insert_with(|| [&preds[..s * n], &pop_preds].concat());
        self.compress(p.cache);
      }
    }

    self.preds.extend_from_slice(chunk_preds);
  }

  /// `_get_silence_profile`: the silent frames of `embs` join the running mean.
  fn silence(&mut self, embs: &[f32], preds: &[f32]) {
    let silent: Vec<bool> =
      preds.chunks(self.n).map(|p| p.iter().fold(0.0f32, |s, x| s + x) < SIL_THRESHOLD).collect();
    let count = silent.iter().filter(|&&s| s).count() as u64;
    if count == 0 {
      return;
    }

    let (old, new) = (self.sil_frames, self.sil_frames + count);
    for (i, mean) in self.sil_mean.iter_mut().enumerate() {
      let rows = embs.chunks(self.d).zip(&silent).filter(|r| *r.1);
      let sum = rows.fold(0.0f32, |s, (row, _)| s + row[i]);
      *mean = (*mean * old as f32 + sum) / new.max(1) as f32;
    }

    self.sil_frames = new;
  }

  /// `_compress_spkcache`: the cache down to `len` frames, per speaker its best-scored ones and a
  /// few of silence, in time order.
  fn compress(&mut self, len: usize) {
    let (d, n) = (self.d, self.n);
    let preds = self.cache_preds.take().unwrap();
    let frames = preds.len() / n;
    let per_speaker = (len / n - SIL_FRAMES_PER_SPEAKER) as f32;
    let [strong, weak, min_pos] = [STRONG_BOOST_RATE, WEAK_BOOST_RATE, MIN_POS_SCORES_RATE]
      .map(|rate| (per_speaker * rate).floor() as usize);

    let mut scores = log_pred_scores(&preds, n);
    disable_low_scores(&preds, &mut scores, n, min_pos);
    // The frames since the last compression; -inf stays -inf.
    scores[len * n..].iter_mut().for_each(|s| *s += SCORES_BOOST_LATEST);
    boost_top(&mut scores, n, strong, 2.0);
    boost_top(&mut scores, n, weak, 1.0);

    // Speaker-major with SIL_FRAMES_PER_SPEAKER frames of +inf after each speaker's, the best
    // `len` (by score, then index), in index order; -inf picks and the silence frames are empty.
    let padded = frames + SIL_FRAMES_PER_SPEAKER;
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
          cache.extend_from_slice(&self.sil_mean);
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
    let p = Params { chunk: 4, fifo: 2, cache: 5, period: 4 };
    let mut st = State::new(1, 1);
    // Frame i has embedding i; the speaker talks in even frames.
    let preds = |frames: std::ops::Range<usize>| -> Vec<f32> {
      frames.map(|i| if i % 2 == 0 { 0.9 } else { 0.0 }).collect()
    };
    let embs = |frames: std::ops::Range<usize>| frames.map(|i| i as f32).collect::<Vec<_>>();

    // Past the FIFO's 2 frames, 4 move to the cache; 1 and 3 are silence.
    st.update(&p, &embs(0..4), &preds(0..4), [0, 0]);
    assert_eq!((st.cache.clone(), st.fifo.len()), (vec![0.0, 1.0, 2.0, 3.0], 0));
    assert_eq!((st.sil_frames, st.sil_mean[0]), (2, 2.0));
    assert!(st.cache_preds.is_none());

    // The forward saw [cache | chunk]. 8 frames pass the cache's 5: per speaker 5 − 3 = 2 frames
    // stay, the best after boosts (6, the latest, then 0 of the equal 0, 2, 4), then 3 of silence,
    // whose mean is now (1 + 3 + 5 + 7) / 4.
    assert_eq!(st.context().count(), 4);
    st.update(&p, &embs(4..8), &preds(0..8), [0, 0]);
    assert_eq!(st.cache, [0.0, 6.0, 4.0, 4.0, 4.0]);
    assert_eq!(st.cache_preds.as_deref(), Some(&[0.9, 0.9, 0.0, 0.0, 0.0][..]));
    assert_eq!(st.preds, preds(0..8));
  }
}
