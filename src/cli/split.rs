// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Long audio through `Model::transcribe_long`, with where each piece starts for subtitles, for
//! `wavo run` and `wavo serve`.

use std::ops::ControlFlow;

use wavo::{Model, Options, Transcript};

/// `Model::transcribe_long` with pieces of up to `segment_ms` (0: one pass, else the library's
/// default) and `each(done, n)` after each piece. The index of each piece's first token comes
/// back too, as subtitles start a word there.
pub fn transcribe(
  model: &Model,
  pcm: &[f32],
  segment_ms: Option<u32>,
  options: &Options,
  mut each: impl FnMut(usize, usize),
) -> wavo::Result<(Transcript, Vec<usize>)> {
  let mut options = options.clone();
  options.window_ms = segment_ms.map(|ms| if ms == 0 { u32::MAX } else { ms });
  let (mut starts, mut tokens) = (Vec::new(), 0);
  let transcript = model.transcribe_long(pcm, &options, |piece| {
    starts.push(tokens);
    tokens += piece.transcript.tokens.len();
    each(piece.index + 1, piece.count);
    ControlFlow::Continue(())
  })?;
  Ok((transcript, starts))
}
