// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Long audio split at its quietest moments, by frame energy without a VAD model, and the
//! segments' transcripts joined back into one.

use std::ops::Range;

use wavo::{Token, Transcript};

/// Samples per energy frame (10 ms); cuts fall on frame starts.
const FRAME: usize = 160;
/// Frames in the window that picks a pause (300 ms): long enough to prefer pauses between
/// sentences to gaps between words.
const WINDOW: usize = 30;
/// The last segment is at least this many samples (1 s).
const MIN_TAIL: usize = 16000;

/// Contiguous ranges of at most `max` samples covering `pcm`; one range when it fits. Each cut
/// lies between `max / 2` and `max` into its segment (and 1 s before the end): in the latest
/// 300 ms window there with at most twice the least energy, at its quietest frame. Only relative
/// energy counts, so the level and steady noise don't matter, and without a pause the cut still
/// lands on a quiet 10 ms. `max` must be at least 5 s.
pub fn segments(pcm: &[f32], max: usize) -> Vec<Range<usize>> {
  let energy: Vec<f32> = pcm.chunks(FRAME).map(|f| f.iter().map(|x| x * x).sum()).collect();
  let (mut out, mut start) = (Vec::new(), 0);
  while pcm.len() - start > max {
    let first = (start + max / 2) / FRAME;
    let end = (start + max).min(pcm.len() - MIN_TAIL) / FRAME;
    let sums: Vec<f32> =
      (first..=end - WINDOW).map(|f| energy[f..f + WINDOW].iter().sum()).collect();
    let least = sums.iter().copied().fold(f32::INFINITY, f32::min);
    let window = first + sums.iter().rposition(|&s| s <= 2.0 * least).unwrap();
    let frames = (window..window + WINDOW).rev();
    let cut = frames.min_by(|&a, &b| energy[a].total_cmp(&energy[b])).unwrap() * FRAME;
    out.push(start..cut);
    start = cut;
  }
  out.push(start..pcm.len());
  out
}

/// Joins the transcripts of segments starting at the given samples: texts with a space, token
/// times shifted to the whole audio. Also returns the index of each segment's first token.
pub fn join(parts: impl IntoIterator<Item = (usize, Transcript)>) -> (Transcript, Vec<usize>) {
  let (mut out, mut starts) = (Transcript::default(), Vec::new());
  for (start, part) in parts {
    starts.push(out.tokens.len());
    if !part.text.is_empty() {
      out.text += if out.text.is_empty() { "" } else { " " };
      out.text += &part.text;
    }
    let offset = (start / 16) as u32; // a multiple of 160 samples: whole ms
    let shift = |t: Token| Token { start_ms: t.start_ms + offset, ..t };
    out.tokens.extend(part.tokens.into_iter().map(shift));
  }
  (out, starts)
}

#[cfg(test)]
mod tests {
  use std::f32::consts::TAU;

  use super::*;

  const SEC: usize = 16000;

  /// Speech-like tone bursts at 0.5 with the given pauses (start and length in ms) filled by
  /// `floor`-level noise.
  fn audio(ms: usize, pauses: &[(usize, usize)], floor: f32) -> Vec<f32> {
    let mut x = 1u32;
    let mut noise = move || {
      x = x.wrapping_mul(1664525).wrapping_add(1013904223);
      floor * ((x >> 8) as f32 / (1 << 23) as f32 - 1.0)
    };
    (0..ms * 16)
      .map(|i| {
        let t = i / 16;
        if pauses.iter().any(|&(s, n)| (s..s + n).contains(&t)) {
          noise()
        } else {
          // a 180 Hz voice whose level swings at 4 Hz, like syllables, never reaching zero
          let level = 0.3 + 0.2 * (TAU * 4.0 * i as f32 / 16000.0).sin();
          level * (TAU * 180.0 * i as f32 / 16000.0).sin() + noise()
        }
      })
      .collect()
  }

  fn check_ranges(ranges: &[Range<usize>], len: usize, max: usize) {
    assert_eq!(ranges.first().unwrap().start, 0);
    assert_eq!(ranges.last().unwrap().end, len);
    for pair in ranges.windows(2) {
      assert_eq!(pair[0].end, pair[1].start, "contiguous");
      assert_eq!(pair[0].end % FRAME, 0, "cut on a frame start");
      assert!(pair[0].len() >= max / 2 - FRAME, "{:?}", pair[0]);
    }
    assert!(ranges.iter().all(|r| r.len() <= max), "{ranges:?}");
    assert!(ranges.last().unwrap().len() >= MIN_TAIL, "{ranges:?}");
  }

  #[test]
  fn short_audio_is_one_range() {
    let one = |pcm: &[f32], max| match &segments(pcm, max)[..] {
      [range] => range.clone(),
      ranges => panic!("{ranges:?}"),
    };
    assert_eq!(one(&audio(25_000, &[(10_000, 500)], 0.0), 25 * SEC), 0..25 * SEC);
    assert_eq!(one(&audio(60_000, &[], 0.0), usize::MAX), 0..60 * SEC);
    assert_eq!(one(&[], 25 * SEC), 0..0);
  }

  #[test]
  fn cuts_in_the_pause_nearest_the_limit() {
    // The first segment may end between 12.5 and 25 s: the pause at 8 s is too early and the
    // one at 21 s beats the one at 14 s. The second may end between ~34 s and 46.4 s (1 s before
    // the end), so in the pause at 45 s, not the one at 33 s.
    let pauses = [(8_000, 600), (14_000, 400), (21_000, 500), (33_000, 300), (45_000, 400)];
    for floor in [0.0, 0.01] {
      let pcm = audio(47_400, &pauses, floor);
      let ranges = segments(&pcm, 25 * SEC);
      check_ranges(&ranges, pcm.len(), 25 * SEC);
      let cuts: Vec<usize> = ranges[1..].iter().map(|r| r.start / 16).collect();
      assert_eq!(cuts.len(), 2, "{cuts:?}");
      assert!((21_000..21_500).contains(&cuts[0]), "{cuts:?}");
      assert!((45_000..45_400).contains(&cuts[1]), "{cuts:?}");
      let quiet = segments(&pcm.iter().map(|x| x * 0.01).collect::<Vec<_>>(), 25 * SEC);
      assert_eq!(quiet, ranges, "the level doesn't matter");
    }
  }

  #[test]
  fn without_pauses_cuts_stay_in_range() {
    let pcm = audio(130_000, &[], 0.001);
    let ranges = segments(&pcm, 25 * SEC);
    check_ranges(&ranges, pcm.len(), 25 * SEC);
    assert!(ranges.len() >= 6, "{ranges:?}");
    let pcm = audio(60_000, &[], 0.0); // no noise: windows repeat exactly
    let ranges = segments(&pcm, 5 * SEC);
    check_ranges(&ranges, pcm.len(), 5 * SEC);
  }

  #[test]
  fn join_shifts_tokens_and_skips_empty_texts() {
    let token = |id, start_ms| Token { id, piece: format!("▁{id}"), start_ms };
    let part = |text: &str, tokens| Transcript { text: text.into(), tokens };
    let (t, starts) = join([
      (0, part("a b", vec![token(1, 40), token(2, 80)])),
      (16 * 20_000, part("", vec![])),
      (16 * 30_010, part("c", vec![token(3, 0)])),
    ]);
    assert_eq!(t.text, "a b c");
    let times: Vec<(u32, u32)> = t.tokens.iter().map(|t| (t.id, t.start_ms)).collect();
    assert_eq!(times, [(1, 40), (2, 80), (3, 30_010)]);
    assert_eq!(starts, [0, 2, 2]);
    let one = part("x", vec![token(7, 120)]);
    assert_eq!(join([(0, one.clone())]).0, one, "a single segment from 0 is unchanged");
  }
}
