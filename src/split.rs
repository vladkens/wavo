// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Long audio split at its quietest moments, by frame energy without a VAD model, and the pieces'
//! transcripts joined back into one.

use std::ops::Range;

use crate::Transcript;

/// Samples per energy frame (10 ms); cuts fall on frame starts.
const FRAME: usize = 160;
/// Frames in the window that picks a pause (300 ms): long enough to prefer pauses between
/// sentences to gaps between words.
const WINDOW: usize = 30;
/// The last piece is at least this many samples (1 s).
const MIN_TAIL: usize = 16000;
/// The shortest piece `split` makes, in ms.
const MIN_MS: u32 = 5000;

/// Splits 16 kHz audio at its quietest moments into pieces of at most `max_ms` milliseconds, and
/// returns their sample ranges: contiguous, covering all of `pcm`, one range when it fits.
///
/// Each cut lies between `max_ms / 2` and `max_ms` into its piece (and 1 s before the end): in the
/// latest 300 ms window there with at most twice the least energy, at its quietest 10 ms. Only
/// relative energy counts, so the level and steady noise don't matter, and without a pause the
/// cut still lands on a quiet moment. It is no voice activity detection: silence stays in the
/// pieces. `max_ms` below 5000 counts as 5000; `u32::MAX` (49 days) keeps any real recording in
/// one piece. It never fails: empty audio gives one empty range.
///
/// [`Model::transcribe_long`](crate::Model::transcribe_long) splits with it; use it directly to
/// transcribe the pieces in your own way, and shift each piece's token `start_ms` by its start
/// (`range.start / 16` ms; cuts fall on whole milliseconds).
///
/// # Examples
///
/// ```
/// let pcm = vec![0.0f32; 16_000 * 40]; // 40 s
/// let pieces = wavo::split(&pcm, 15_000);
/// assert_eq!(pieces.len(), 3);
/// assert!(pieces.iter().all(|r| r.len() <= 16 * 15_000));
/// assert_eq!((pieces[0].start, pieces[2].end), (0, pcm.len()));
/// ```
pub fn split(pcm: &[f32], max_ms: u32) -> Vec<Range<usize>> {
  let max = (max_ms.max(MIN_MS) as usize).saturating_mul(16);
  // Audio that fits takes no energy pass.
  let fits = pcm.len() <= max;
  let frames = pcm.chunks(FRAME).take(if fits { 0 } else { usize::MAX });
  let energy: Vec<f32> = frames.map(|f| f.iter().map(|x| x * x).sum()).collect();
  let (mut out, mut start) = (Vec::new(), 0);
  while pcm.len() - start > max {
    let first = (start + max / 2) / FRAME;
    let end = (start + max).min(pcm.len() - MIN_TAIL) / FRAME;
    let sums: Vec<f32> =
      (first..=end - WINDOW).map(|f| energy[f..f + WINDOW].iter().sum()).collect();
    let least = sums.iter().copied().fold(f32::INFINITY, f32::min);
    // None only when every sum is NaN: then the latest window.
    let quiet = sums.iter().rposition(|&s| s <= 2.0 * least);
    let window = first + quiet.unwrap_or(sums.len() - 1);
    let frames = (window..window + WINDOW).rev();
    let cut = frames.min_by(|&a, &b| energy[a].total_cmp(&energy[b])).unwrap() * FRAME;
    out.push(start..cut);
    start = cut;
  }
  out.push(start..pcm.len());
  out
}

/// Appends the transcript of the next piece, its token times already from the start of the
/// audio: texts joined with a space, an empty one skipped.
pub(crate) fn append(out: &mut Transcript, part: Transcript) {
  if !part.text.is_empty() {
    out.text += if out.text.is_empty() { "" } else { " " };
    out.text += &part.text;
  }

  out.tokens.extend(part.tokens);
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
    let one = |pcm: &[f32], max| match &split(pcm, max)[..] {
      [range] => range.clone(),
      ranges => panic!("{ranges:?}"),
    };
    assert_eq!(one(&audio(25_000, &[(10_000, 500)], 0.0), 25_000), 0..25 * SEC);
    assert_eq!(one(&audio(60_000, &[], 0.0), u32::MAX), 0..60 * SEC);
    assert_eq!(one(&[], 25_000), 0..0);
  }

  #[test]
  fn cuts_in_the_pause_nearest_the_limit() {
    // The first piece may end between 12.5 and 25 s: the pause at 8 s is too early and the one
    // at 21 s beats the one at 14 s. The second may end between ~34 s and 46.4 s (1 s before the
    // end), so in the pause at 45 s, not the one at 33 s.
    let pauses = [(8_000, 600), (14_000, 400), (21_000, 500), (33_000, 300), (45_000, 400)];
    for floor in [0.0, 0.01] {
      let pcm = audio(47_400, &pauses, floor);
      let ranges = split(&pcm, 25_000);
      check_ranges(&ranges, pcm.len(), 25 * SEC);
      let cuts: Vec<usize> = ranges[1..].iter().map(|r| r.start / 16).collect();
      assert_eq!(cuts.len(), 2, "{cuts:?}");
      assert!((21_000..21_500).contains(&cuts[0]), "{cuts:?}");
      assert!((45_000..45_400).contains(&cuts[1]), "{cuts:?}");
      let quiet = split(&pcm.iter().map(|x| x * 0.01).collect::<Vec<_>>(), 25_000);
      assert_eq!(quiet, ranges, "the level doesn't matter");
    }
  }

  #[test]
  fn without_pauses_cuts_stay_in_range() {
    let pcm = audio(130_000, &[], 0.001);
    let ranges = split(&pcm, 25_000);
    check_ranges(&ranges, pcm.len(), 25 * SEC);
    assert!(ranges.len() >= 6, "{ranges:?}");
    let pcm = audio(60_000, &[], 0.0); // no noise: windows repeat exactly
    let ranges = split(&pcm, 5_000);
    check_ranges(&ranges, pcm.len(), 5 * SEC);
    assert_eq!(split(&pcm, 1_000), ranges, "under 5 s counts as 5 s");
    let nan = vec![f32::NAN; 12 * SEC];
    check_ranges(&split(&nan, 5_000), nan.len(), 5 * SEC);
  }

  #[test]
  fn append_joins_texts_and_skips_empty_ones() {
    let part = |text: &str, ids: &[u32]| Transcript {
      text: text.into(),
      tokens: ids.iter().map(|&id| crate::Token { id, ..Default::default() }).collect(),
    };
    let mut t = Transcript::default();
    for p in [part("a b", &[1, 2]), part("", &[]), part("c", &[3])] {
      append(&mut t, p);
    }

    assert_eq!(t.text, "a b c");
    assert_eq!(t.tokens.iter().map(|t| t.id).collect::<Vec<_>>(), [1, 2, 3]);
  }
}
