// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The CPU side of decoding, as transcribe.cpp's `src/arch/whisper/model.cpp` runs it by default:
//! greedy at temperature 0 with segment timestamps, the suppression lists and timestamp rules on
//! the raw logits, and HF's `_retrieve_segment` to cut a window's tokens into segments and find
//! where the next window starts. Temperature fallback is not ported.

use crate::error::Result;

/// Tokens generated per window at most, and the decoder's context.
const MAX_NEW: usize = 256;
const CONTEXT: usize = 448;
/// The first timestamp is at most this many 20 ms steps in (1.0 s).
const MAX_INITIAL: usize = 50;

pub struct Ids {
  pub eot: usize,
  pub no_timestamps: usize,
  /// The first timestamp, `<|0.00|>`; timestamps run to the end of the vocabulary.
  pub begin: usize,
  pub suppress: Vec<usize>,
  pub begin_suppress: Vec<usize>,
}

impl Ids {
  fn timestamp(&self, id: usize) -> bool {
    id >= self.begin
  }
}

/// Greedy decoding of one window, from the logits of the prompt's last token at position
/// `prompt - 1`. `step(token, position)` runs the decoder and returns the logits. Returns the
/// generated tokens, without EOT, and their average log-probability when `logprobs` is set (else
/// 0).
pub fn decode(
  ids: &Ids,
  mut logits: Vec<f32>,
  prompt: usize,
  logprobs: bool,
  mut step: impl FnMut(usize, usize) -> Result<Vec<f32>>,
) -> Result<(Vec<usize>, f32)> {
  let mut generated = Vec::new();
  let (mut sum, mut n) = (0f64, 0);
  let mut pick = |logits: &mut [f32], generated: &[usize]| {
    for &id in &ids.suppress {
      logits[id] = f32::NEG_INFINITY;
    }
    if generated.is_empty() {
      for &id in &ids.begin_suppress {
        logits[id] = f32::NEG_INFINITY;
      }
    }
    timestamp_rules(ids, logits, generated);
    let (id, lp) = argmax(logits, logprobs);
    (sum, n) = (sum + lp as f64, n + 1);
    id
  };
  let mut next = pick(&mut logits, &generated);
  for pos in prompt..prompt + MAX_NEW {
    if next == ids.eot {
      break;
    }
    generated.push(next);
    if pos + 1 > CONTEXT {
      break;
    }
    logits = step(next, pos)?;
    next = pick(&mut logits, &generated);
  }
  Ok((generated, if logprobs { (sum / n as f64) as f32 } else { 0.0 }))
}

/// HF's `WhisperTimeStampLogitsProcessor` as `apply_whisper_timestamp_rules` (model.cpp): no
/// `<|notimestamps|>`; timestamps in pairs; never decreasing; the first token a timestamp of at
/// most 1 s; and only timestamps when their total probability beats every other token.
fn timestamp_rules(ids: &Ids, logits: &mut [f32], generated: &[usize]) {
  let begin = ids.begin;
  logits[ids.no_timestamps] = f32::NEG_INFINITY;
  let n = generated.len();
  let last = n > 0 && ids.timestamp(generated[n - 1]);
  let penultimate = n < 2 || ids.timestamp(generated[n - 2]);
  if last {
    let range = if penultimate { begin..logits.len() } else { 0..ids.eot };
    logits[range].fill(f32::NEG_INFINITY);
  }
  if let Some(&ts) = generated.iter().rev().find(|&&id| ids.timestamp(id)) {
    let low = if last && !penultimate { ts } else { ts + 1 };
    logits[begin..low].fill(f32::NEG_INFINITY);
  }
  if generated.is_empty() {
    logits[..begin].fill(f32::NEG_INFINITY);
    let last = (begin + MAX_INITIAL + 1).min(logits.len());
    logits[last..].fill(f32::NEG_INFINITY);
  }
  let max = logits[begin..].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
  let mut mass = f32::NEG_INFINITY;
  if max.is_finite() {
    let sum: f64 =
      logits[begin..].iter().filter(|v| v.is_finite()).map(|&v| ((v - max) as f64).exp()).sum();
    if sum > 0.0 {
      mass = max + sum.ln() as f32;
    }
  }
  let text = logits[..begin].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
  if mass > text {
    logits[..begin].fill(f32::NEG_INFINITY);
  }
}

/// The first maximum and, with `logprob`, its log-softmax: f32 `l − max`, f64 exponentials, as
/// `sample_argmax_and_logprob` (model.cpp).
fn argmax(logits: &[f32], logprob: bool) -> (usize, f32) {
  let mut best = 0;
  for (i, &v) in logits.iter().enumerate() {
    if v > logits[best] {
      best = i;
    }
  }
  if !logprob {
    return (best, 0.0);
  }
  let max = logits.iter().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, |m, &v| m.max(v));
  let sum: f64 = logits.iter().filter(|v| v.is_finite()).map(|&v| ((v - max) as f64).exp()).sum();
  (best, logits[best] - (max + sum.ln() as f32))
}

/// The softmax probability of `id`, as the reference computes the no-speech probability.
pub fn probability(logits: &[f32], id: usize) -> f32 {
  let max = logits.iter().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, |m, &v| m.max(v));
  let e = |v: f32| if v.is_finite() { ((v - max) as f64).exp() } else { 0.0 };
  let sum: f64 = logits.iter().map(|&v| e(v)).sum();
  (e(logits[id]) / sum) as f32
}

/// A segment of a window: its start in 20 ms timestamp steps and its text tokens.
#[derive(Debug, PartialEq)]
pub struct Segment {
  pub start: usize,
  pub text: Vec<usize>,
}

/// HF's `_retrieve_segment` as `whisper_retrieve_segment` (model.cpp): the segments between
/// consecutive timestamps (with no such pair, one segment from the window's start), and how many
/// mel frames the next window moves on: to the last closed timestamp, or `frames` (the window) when
/// the tokens end in one timestamp or have no pair. Segments may be empty; the caller drops those.
pub fn retrieve(ids: &Ids, generated: &[usize], frames: usize) -> (Vec<Segment>, usize) {
  let ts = |i: usize| ids.timestamp(generated[i]);
  let text =
    |r: std::ops::Range<usize>| generated[r].iter().copied().filter(|&id| id < ids.eot).collect();
  let n = generated.len();
  let mut slices: Vec<usize> =
    (0..n.saturating_sub(1)).filter(|&i| ts(i) && ts(i + 1)).map(|i| i + 1).collect();
  if slices.is_empty() {
    let segments = if n > 0 { vec![Segment { start: 0, text: text(0..n) }] } else { Vec::new() };
    return (segments, frames);
  }
  let single = n >= 2 && !ts(n - 2) && ts(n - 1);
  if single {
    slices.push(n);
  } else {
    *slices.last_mut().unwrap() += 1;
  }
  let (mut segments, mut last) = (Vec::new(), 0);
  for (k, &end) in slices.iter().enumerate() {
    let closing = if k + 1 < slices.len() || single { end - 1 } else { end - 2 };
    if end > last && ts(last) && closing >= last && closing < n && ts(closing) {
      segments.push(Segment { start: generated[last] - ids.begin, text: text(last..end) });
    }
    last = end;
  }
  let advance = match last.checked_sub(2) {
    _ if single => frames,
    Some(i) if i < n && ts(i) => (generated[i] - ids.begin) * 2,
    _ => frames,
  };
  (segments, advance)
}

#[cfg(test)]
mod tests {
  use super::*;

  // A vocabulary of 20: text 0..10, EOT 10, notimestamps 12, timestamps 13..20.
  const T: usize = 13;

  fn ids() -> Ids {
    Ids { eot: 10, no_timestamps: 12, begin: T, suppress: vec![11], begin_suppress: vec![1, 10] }
  }

  #[test]
  fn timestamp_rules() {
    let ids = ids();
    let rules = |generated: &[usize]| {
      let mut l = vec![0f32; 20];
      super::timestamp_rules(&ids, &mut l, generated);
      l.iter().map(|v| v.is_finite()).collect::<Vec<_>>()
    };
    let allowed = |set: &[usize]| (0..20).map(|i| set.contains(&i)).collect::<Vec<_>>();
    // The first token is a timestamp (and at most 1 s, all of these).
    assert_eq!(rules(&[]), allowed(&(T..20).collect::<Vec<_>>()));
    // After text then a timestamp: EOT or a later (or the same) timestamp; with all logits equal
    // the timestamps' mass wins, so only they remain.
    assert_eq!(rules(&[T, 3, T + 2]), allowed(&(T + 2..20).collect::<Vec<_>>()));
    // After two timestamps: no timestamps; text wins.
    assert_eq!(rules(&[T, 3, T + 2, T + 2]), allowed(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]));
    // After text: timestamps strictly after the last one, which outweigh text.
    assert_eq!(rules(&[T + 1, 3]), allowed(&(T + 2..20).collect::<Vec<_>>()));
  }

  #[test]
  fn decode_stops_at_eot_and_counts_logprobs() {
    let ids = ids();
    // Scripted: the prompt picks T, then 3, then T + 4, then EOT.
    let script = [3, T + 4, 10];
    let one_hot = |id: usize| (0..20).map(|i| if i == id { 5.0 } else { 0.0 }).collect::<Vec<_>>();
    let mut calls = Vec::new();
    let (out, lp) = decode(&ids, one_hot(T), 3, true, |token, pos| {
      calls.push((token, pos));
      Ok(one_hot(script[calls.len() - 1]))
    })
    .unwrap();
    assert_eq!(out, [T, 3, T + 4]);
    assert_eq!(calls, [(T, 3), (3, 4), (T + 4, 5)]);
    assert!(lp < 0.0 && lp > -1.0);
  }

  #[test]
  fn retrieve_segments() {
    let ids = ids();
    let seg = |start: usize, text: &[usize]| Segment { start, text: text.to_vec() };
    // No pair: one segment from the window start, full advance.
    assert_eq!(retrieve(&ids, &[T, 1, 2, T + 3], 3000), (vec![seg(0, &[1, 2])], 3000));
    // A pair and a single closing timestamp: both segments, full advance.
    let g = [T, 1, T + 2, T + 2, 3, T + 5];
    assert_eq!(retrieve(&ids, &g, 3000), (vec![seg(0, &[1]), seg(2, &[3])], 3000));
    // Ending in two timestamps: the last pair closes, the window moves to it.
    let g = [T, 1, T + 2, T + 3, 4, T + 5, T + 5];
    assert_eq!(retrieve(&ids, &g, 3000), (vec![seg(0, &[1]), seg(3, &[4])], 10));
    // An unfinished tail after the last pair is dropped, the next window starts at that pair.
    let g = [T, 1, T + 2, T + 3, 4];
    assert_eq!(retrieve(&ids, &g, 3000), (vec![seg(0, &[1])], 4));
    assert_eq!(retrieve(&ids, &[], 3000), (vec![], 3000));
  }
}
