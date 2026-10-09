// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! RNN-T greedy decoding on the CPU: embedding + one LSTM layer as the predictor, joint network
//! `out(relu(enc + pred(h)))`. The encoder side of the joint is projected on the GPU.

use super::dot;
use crate::error::Result;
use crate::gguf::Gguf;

/// At most this many tokens per encoder frame.
const MAX_SYMBOLS: usize = 10;

pub struct Decoder {
  hidden: usize,
  joint: usize,
  blank: usize,
  /// `[vocab][hidden]`
  embed: Vec<f32>,
  /// `[4·hidden][hidden]`, gates i, f, g, o.
  wx: Vec<f32>,
  wh: Vec<f32>,
  /// `bias_ih + bias_hh`
  bias: Vec<f32>,
  /// `[joint][hidden]`
  pred_w: Vec<f32>,
  pred_b: Vec<f32>,
  /// `[vocab][joint]`
  out_w: Vec<f32>,
  out_b: Vec<f32>,
}

struct State {
  h: Vec<f32>,
  c: Vec<f32>,
}

impl Decoder {
  pub fn new(g: &Gguf, hidden: usize, joint: usize, vocab: usize, blank: usize) -> Result<Self> {
    let t = |name: &str, dims: &[usize]| g.tensor(name, dims).map(|t| t.to_f32());
    Ok(Self {
      hidden,
      joint,
      blank,
      embed: t("pred.embed.weight", &[hidden, vocab])?,
      wx: t("pred.lstm.0.Wx", &[hidden, 4 * hidden])?,
      wh: t("pred.lstm.0.Wh", &[hidden, 4 * hidden])?,
      bias: t("pred.lstm.0.bias", &[4 * hidden])?,
      pred_w: t("joint.pred.weight", &[hidden, joint])?,
      pred_b: t("joint.pred.bias", &[joint])?,
      out_w: t("joint.out.weight", &[joint, vocab])?,
      out_b: t("joint.out.bias", &[vocab])?,
    })
  }

  /// Greedy search over `enc` `[frames][joint]` (encoder projection with bias). Returns
  /// `(token, frame)` pairs. The first predictor input is zero; the predictor state advances
  /// only when a token is emitted.
  pub fn decode(&self, enc: &[f32]) -> Vec<(u32, u32)> {
    let zero = State { h: vec![0.0; self.hidden], c: vec![0.0; self.hidden] };
    let mut next = self.lstm(&vec![0.0; self.hidden], &zero);
    let mut pred = self.predict(&next.h);
    let mut out = Vec::new();
    for (t, frame) in enc.chunks_exact(self.joint).enumerate() {
      for _ in 0..MAX_SYMBOLS {
        let token = self.joint(frame, &pred);
        if token == self.blank {
          break;
        }
        out.push((token as u32, t as u32));
        let x = &self.embed[token * self.hidden..][..self.hidden];
        next = self.lstm(x, &next);
        pred = self.predict(&next.h);
      }
    }
    out
  }

  fn lstm(&self, x: &[f32], s: &State) -> State {
    let n = self.hidden;
    let gate =
      |i: usize| self.bias[i] + dot(&self.wx[i * n..][..n], x) + dot(&self.wh[i * n..][..n], &s.h);
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let mut next = State { h: vec![0.0; n], c: vec![0.0; n] };
    for j in 0..n {
      let (i, f, g, o) = (gate(j), gate(n + j), gate(2 * n + j), gate(3 * n + j));
      next.c[j] = sigmoid(f) * s.c[j] + sigmoid(i) * g.tanh();
      next.h[j] = sigmoid(o) * next.c[j].tanh();
    }
    next
  }

  fn predict(&self, h: &[f32]) -> Vec<f32> {
    let n = self.hidden;
    (0..self.joint).map(|j| self.pred_b[j] + dot(&self.pred_w[j * n..][..n], h)).collect()
  }

  /// Argmax of the joint logits; ties go to the lowest id.
  fn joint(&self, enc: &[f32], pred: &[f32]) -> usize {
    let z: Vec<f32> = enc.iter().zip(pred).map(|(e, p)| (e + p).max(0.0)).collect();
    let mut best = (0, f32::NEG_INFINITY);
    for (v, w) in self.out_w.chunks_exact(self.joint).enumerate() {
      let logit = dot(w, &z) + self.out_b[v];
      if logit > best.1 {
        best = (v, logit);
      }
    }
    best.0
  }
}
