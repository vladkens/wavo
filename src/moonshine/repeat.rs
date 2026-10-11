// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The stop rule for a greedy decoder stuck repeating itself, as transcribe.cpp's
//! `src/transcribe-repetition-guard.h`: a block of tokens repeating at the tail counts as a loop
//! once it has enough copies, many more for short blocks, so emphatic repetition survives.

/// When a block repeating at the tail counts as a loop: `copies` copies spanning at least
/// `min_tokens`, but once that span passes `max_tokens` it only has to cover `max_tokens`, with at
/// least `min_copies` copies.
struct Bar {
  max_block: usize,
  copies: usize,
  min_tokens: usize,
  max_tokens: usize,
  min_copies: usize,
}

/// Stop mid-decode: a 1-token block needs 64 copies, an 8-token block 8.
const STOP: Bar = Bar { max_block: 128, copies: 8, min_tokens: 64, max_tokens: 192, min_copies: 4 };
/// Trim at a stop on the token budget: 3 copies spanning at least 32 tokens.
const TRIM: Bar = Bar { max_block: 256, copies: 3, min_tokens: 32, max_tokens: 768, min_copies: 3 };

/// The length of the block repeating at the end of `ids` under `bar`, or 0.
fn tail_block(ids: &[usize], bar: &Bar) -> usize {
  let n = ids.len();
  (1..=bar.max_block)
    .find(|&block| {
      let span = (bar.copies * block).clamp(bar.min_tokens, bar.max_tokens);
      let span = block * bar.min_copies.max(span.div_ceil(block));
      span <= n && (n - span + block..n).all(|i| ids[i] == ids[i - block])
    })
    .unwrap_or(0)
}

/// Drops every copy of the tail block but the first; whether there was one.
fn trim(ids: &mut Vec<usize>, bar: &Bar) -> bool {
  let block = tail_block(ids, bar);
  while block > 0
    && ids.len() >= 2 * block
    && ids[ids.len() - block..] == ids[ids.len() - 2 * block..ids.len() - block]
  {
    ids.truncate(ids.len() - block);
  }
  block > 0
}

/// After each token: on a loop, trims `ids` to one copy and returns true, so decoding stops.
pub fn stop(ids: &mut Vec<usize>) -> bool {
  trim(ids, &STOP)
}

/// Once, when decoding reached the token budget before the end: drops a repeating tail.
pub fn trim_at_budget(ids: &mut Vec<usize>) {
  trim(ids, &TRIM);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn loops_stop_and_repeats_stay() {
    // 63 copies of one token are emphasis, 64 a loop.
    let mut ids = vec![5; 63];
    assert!(!stop(&mut ids));
    ids.push(5);
    assert!(stop(&mut ids));
    assert_eq!(ids, [5]);
    // An 8-token block needs 8 copies, after a prefix that stays.
    let block: Vec<usize> = (10..18).collect();
    let mut ids = [vec![1, 2], block.repeat(7)].concat();
    assert!(!stop(&mut ids));
    ids.extend(&block);
    assert!(stop(&mut ids));
    assert_eq!(ids, [vec![1, 2], block.clone()].concat());
    // A 48-token block stops at 4 copies (192 tokens).
    let block: Vec<usize> = (100..148).collect();
    let mut ids = block.repeat(3);
    assert!(!stop(&mut ids));
    ids.extend(&block);
    assert!(stop(&mut ids));
    assert_eq!(ids, block);
    // At the budget 3 copies of a block of 11 (33 tokens) go, 3 of a block of 10 (30) stay.
    let block: Vec<usize> = (20..31).collect();
    let mut ids = [vec![1], block.repeat(3)].concat();
    trim_at_budget(&mut ids);
    assert_eq!(ids, [vec![1], block].concat());
    let mut ids = (20..30).collect::<Vec<_>>().repeat(3);
    trim_at_budget(&mut ids);
    assert_eq!(ids.len(), 30);
  }
}
