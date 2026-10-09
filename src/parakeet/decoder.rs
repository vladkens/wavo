// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! TDT greedy decoding on the CPU: a predictor of stacked LSTM layers over the last emitted token,
//! and a joint `out(relu(enc + pred(h)))` whose outputs are the token logits, then one logit per
//! duration (how many frames to advance). The encoder side of the joint is projected on the GPU.

use crate::cpu::{matmul, rows4};
use crate::error::{Result, bail};
use crate::gguf::Gguf;

pub struct Decoder {
  hidden: usize,
  joint: usize,
  /// Token classes, blank included.
  classes: usize,
  blank: usize,
  durations: Vec<usize>,
  max_symbols: usize,
  /// `[classes][hidden]`
  embed: Vec<f32>,
  lstm: Vec<Lstm>,
  /// `[joint][hidden]`, rows padded to a multiple of 4.
  pred_w: Vec<f32>,
  pred_b: Vec<f32>,
  /// `[classes + durations][joint]`, rows padded to a multiple of 4.
  out_w: Vec<f32>,
  out_b: Vec<f32>,
}

struct Lstm {
  /// `[4·hidden][hidden]`, gates i, f, g, o.
  wx: Vec<f32>,
  wh: Vec<f32>,
  /// `bias_ih + bias_hh`
  bias: Vec<f32>,
}

impl Decoder {
  pub fn new(g: &Gguf, classes: usize, blank: usize) -> Result<Self> {
    let u = |key: &str| g.get::<u32>(&format!("stt.parakeet.{key}")).map(|v| v as usize);
    let (hidden, joint) = (u("predictor.hidden")?, u("joint.hidden")?);
    let durations = g.array::<i32>("stt.parakeet.tdt.durations")?;
    let max_symbols = u("tdt.max_symbols")?;
    g.check("stt.parakeet.joint.activation", &["relu"])?;
    g.check("stt.parakeet.predictor.vocab", &[classes as u32])?;
    if u("joint.num_extra_outputs")? != durations.len() || durations.iter().any(|&d| d < 0) {
      bail!("TDT durations {durations:?} don't match the joint's extra outputs");
    }
    if max_symbols == 0 {
      bail!("stt.parakeet.tdt.max_symbols is 0");
    }
    let t = |name: &str, dims: &[usize]| g.tensor(name, dims).and_then(|t| t.to_f32());
    let lstm = (0..u("predictor.n_layers")?)
      .map(|l| {
        Ok(Lstm {
          wx: t(&format!("pred.lstm.{l}.Wx"), &[hidden, 4 * hidden])?,
          wh: t(&format!("pred.lstm.{l}.Wh"), &[hidden, 4 * hidden])?,
          bias: t(&format!("pred.lstm.{l}.bias"), &[4 * hidden])?,
        })
      })
      .collect::<Result<_>>()?;
    let outputs = classes + durations.len();
    Ok(Self {
      hidden,
      joint,
      classes,
      blank,
      durations: durations.into_iter().map(|d| d as usize).collect(),
      max_symbols,
      embed: t("pred.embed.weight", &[hidden, classes])?,
      lstm,
      pred_w: rows4(t("joint.pred.weight", &[hidden, joint])?, hidden),
      pred_b: t("joint.pred.bias", &[joint])?,
      out_w: rows4(t("joint.out.weight", &[joint, outputs])?, joint),
      out_b: t("joint.out.bias", &[outputs])?,
    })
  }

  /// Greedy search over `enc` `[frames][joint]` (encoder projection with bias). Returns
  /// `(token, frame)` pairs. The first predictor input and state are zeros.
  pub fn decode(&self, enc: &[f32]) -> Vec<(u32, u32)> {
    let j = self.joint;
    let mut state = vec![0.0; 2 * self.lstm.len() * self.hidden];
    let (mut next, mut pred) = self.predict(None, &state);
    greedy(enc.len() / j, self.blank, &self.durations, self.max_symbols, |t, emitted| {
      if let Some(token) = emitted {
        state = std::mem::take(&mut next);
        (next, pred) = self.predict(Some(token), &state);
      }
      self.joint(&enc[t * j..][..j], &pred)
    })
  }

  /// One step of the LSTM layers from `state` (per layer h, then c) on the embedding of `token`,
  /// or zeros. Returns the next state and the joint's predictor projection of the top h.
  fn predict(&self, token: Option<usize>, state: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = self.hidden;
    let mut x = token.map_or(vec![0.0; n], |t| self.embed[t * n..][..n].to_vec());
    let mut next = vec![0.0; state.len()];
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    for ((l, s), out) in
      self.lstm.iter().zip(state.chunks_exact(2 * n)).zip(next.chunks_exact_mut(2 * n))
    {
      let (h, c) = s.split_at(n);
      let (gx, gh) = (matmul(&l.wx, &x, n), matmul(&l.wh, h, n));
      let gate = |i: usize| gx[i] + gh[i] + l.bias[i];
      let (nh, nc) = out.split_at_mut(n);
      for k in 0..n {
        let (i, f, g, o) = (gate(k), gate(n + k), gate(2 * n + k), gate(3 * n + k));
        nc[k] = sigmoid(f) * c[k] + sigmoid(i) * g.tanh();
        nh[k] = sigmoid(o) * nc[k].tanh();
      }
      x = nh.to_vec();
    }
    let pred = matmul(&self.pred_w, &x, n).iter().zip(&self.pred_b).map(|(d, b)| d + b).collect();
    (next, pred)
  }

  /// The joint's token and duration index for one frame (each argmax takes the first maximum).
  fn joint(&self, enc: &[f32], pred: &[f32]) -> (usize, usize) {
    let z: Vec<f32> = enc.iter().zip(pred).map(|(e, p)| (e + p).max(0.0)).collect();
    let d = matmul(&self.out_w, &z, self.joint);
    let logits: Vec<f32> = d.iter().zip(&self.out_b).map(|(d, b)| d + b).collect();
    let (tokens, durations) = logits.split_at(self.classes);
    (argmax(tokens), argmax(durations))
  }
}

fn argmax(v: &[f32]) -> usize {
  let mut best = 0;
  for (i, &x) in v.iter().enumerate() {
    if x > v[best] {
      best = i;
    }
  }
  best
}

/// TDT greedy search over `frames`, as `decode_tdt_greedy` in transcribe.cpp's
/// `src/arch/parakeet/decoder.cpp`. `joint(t, emitted)` scores frame t and returns the token and
/// duration index; `emitted` is the token of the previous decision if it emitted one, for the
/// predictor to step on. A token is emitted at the current frame; then the frame advances by the
/// duration. Decisions on one frame are capped at `max_symbols`, and a blank of duration 0 moves
/// on one frame.
fn greedy(
  frames: usize,
  blank: usize,
  durations: &[usize],
  max_symbols: usize,
  mut joint: impl FnMut(usize, Option<usize>) -> (usize, usize),
) -> Vec<(u32, u32)> {
  let (mut out, mut t, mut symbols, mut emitted) = (Vec::new(), 0, 0, None);
  while t < frames {
    let (token, d) = joint(t, emitted.take());
    if token != blank {
      out.push((token as u32, t as u32));
      emitted = Some(token);
    }
    t += durations[d];
    symbols += 1;
    if durations[d] != 0 {
      symbols = 0;
    } else if symbols >= max_symbols || token == blank {
      t += 1;
      symbols = 0;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  const BLANK: usize = 9;

  /// Per decision: the frame, and the token passed to the predictor.
  type Calls = Vec<(usize, Option<usize>)>;

  /// `greedy` over scripted decisions.
  fn run(
    frames: usize,
    durations: &[usize],
    script: &[(usize, usize)],
  ) -> (Vec<(u32, u32)>, Calls) {
    let mut calls = Vec::new();
    let mut script = script.iter();
    let out = super::greedy(frames, BLANK, durations, 10, |t, emitted| {
      calls.push((t, emitted));
      *script.next().expect("script too short")
    });
    assert!(script.next().is_none(), "script too long");
    (out, calls)
  }

  #[test]
  fn blank_moves_on_and_tokens_stay() {
    // Duration indices map to the metadata values: 2 → 3 frames.
    let script = [(BLANK, 0), (5, 0), (6, 2), (BLANK, 1), (7, 1)];
    let (out, calls) = run(6, &[0, 1, 3], &script);
    assert_eq!(out, [(5, 1), (6, 1), (7, 5)]);
    assert_eq!(calls, [(0, None), (1, None), (1, Some(5)), (4, Some(6)), (5, None)]);
  }

  #[test]
  fn ten_decisions_per_frame() {
    // Frame 0: ten tokens of duration 0, then the cap moves on. Frame 1: nine tokens and a blank
    // of duration 0, which is the tenth decision.
    let mut script = vec![(5, 0); 19];
    script.extend([(BLANK, 0), (6, 1)]);
    let (out, calls) = run(3, &[0, 1], &script);
    let frames: Vec<u32> = out.iter().map(|&(_, t)| t).collect();
    assert_eq!(frames, [vec![0; 10], vec![1; 9], vec![2]].concat());
    assert_eq!(calls.iter().filter(|c| c.0 == 1).count(), 10);
  }
}
