// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The stop rule for a greedy decoder stuck repeating itself, as transcribe.cpp's
//! `src/transcribe-repetition-guard.h`: a block of tokens repeating at the tail counts as a loop
//! once it has enough copies, many more for short blocks, so emphatic repetition survives.

/// When a block repeating at the tail counts as a loop: `copies` copies spanning at least
/// `min_tokens`, but once that span passes `max_tokens` it only has to cover `max_tokens`, with at
/// least `min_copies` copies. As `[max_block, copies, min_tokens, max_tokens, min_copies]`: on
/// every token (a 1-token block needs 64 copies, an 8-token block 8), and once at a stop on the
/// token budget (3 copies spanning at least 32 tokens).
const STOP: [usize; 5] = [128, 8, 64, 192, 4];
const TRIM: [usize; 5] = [256, 3, 32, 768, 3];

/// Drops every copy but the first of a block repeating at the end of `ids` under `bar`; whether
/// there was one.
fn trim(
  ids: &mut Vec<usize>,
  [max_block, copies, min_tokens, max_tokens, min_copies]: [usize; 5],
) -> bool {
  let n = ids.len();
  let looped = (1..=max_block).find(|&block| {
    let span = (copies * block).clamp(min_tokens, max_tokens);
    let span = block * min_copies.max(span.div_ceil(block));
    span <= n && (n - span + block..n).all(|i| ids[i] == ids[i - block])
  });
  let Some(block) = looped else { return false };
  while ids.len() >= 2 * block
    && ids[ids.len() - block..] == ids[ids.len() - 2 * block..ids.len() - block]
  {
    ids.truncate(ids.len() - block);
  }
  true
}

/// After each token: on a loop, trims `ids` to one copy and returns true, so decoding stops.
pub fn stop(ids: &mut Vec<usize>) -> bool {
  trim(ids, STOP)
}

/// Once, when decoding reached the token budget before the end: drops a repeating tail.
pub fn trim_at_budget(ids: &mut Vec<usize>) {
  trim(ids, TRIM);
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
