// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! RNN-T greedy decoding on the CPU: embedding + one LSTM layer as the predictor, joint network
//! `out(relu(enc + pred(h)))`. The encoder side of the joint is projected on the GPU.

use super::matmul;
use crate::error::Result;
use crate::gguf::Gguf;

/// At most this many tokens per encoder frame.
const MAX_SYMBOLS: usize = 10;
/// Frames scored at once while the predictor output stays fixed.
const SPAN: usize = 2;

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
  /// `[joint][hidden]`, rows padded to a multiple of 4.
  pred_w: Vec<f32>,
  pred_b: Vec<f32>,
  /// `[vocab][joint]`, rows padded to a multiple of 4.
  out_w: Vec<f32>,
  out_b: Vec<f32>,
}

struct State {
  h: Vec<f32>,
  c: Vec<f32>,
}

impl Decoder {
  pub fn new(g: &Gguf, hidden: usize, joint: usize, vocab: usize, blank: usize) -> Result<Self> {
    let t = |name: &str, dims: &[usize]| g.tensor(name, dims).and_then(|t| t.to_f32());
    // `matmul` takes rows in multiples of 4: pad with zero rows, whose outputs are ignored.
    let rows4 = |mut w: Vec<f32>, k: usize| {
      w.resize((w.len() / k).next_multiple_of(4) * k, 0.0);
      w
    };
    Ok(Self {
      hidden,
      joint,
      blank,
      embed: t("pred.embed.weight", &[hidden, vocab])?,
      wx: t("pred.lstm.0.Wx", &[hidden, 4 * hidden])?,
      wh: t("pred.lstm.0.Wh", &[hidden, 4 * hidden])?,
      bias: t("pred.lstm.0.bias", &[4 * hidden])?,
      pred_w: rows4(t("joint.pred.weight", &[hidden, joint])?, hidden),
      pred_b: t("joint.pred.bias", &[joint])?,
      out_w: rows4(t("joint.out.weight", &[joint, vocab])?, joint),
      out_b: t("joint.out.bias", &[vocab])?,
    })
  }

  /// Greedy search over `enc` `[frames][joint]` (encoder projection with bias). Returns
  /// `(token, frame)` pairs. The first predictor input is zero; the predictor state advances
  /// only when a token is emitted, so frames are scored `SPAN` at a time against one predictor
  /// output, and the frames after a token are scored again.
  pub fn decode(&self, enc: &[f32]) -> Vec<(u32, u32)> {
    let (n, j) = (self.hidden, self.joint);
    let zero = State { h: vec![0.0; n], c: vec![0.0; n] };
    let mut next = self.lstm(&vec![0.0; n], &zero);
    let mut pred = self.predict(&next.h);
    let mut out = Vec::new();
    let frames = enc.len() / j;
    let mut t = 0;
    while t < frames {
      let span = self.joint(&enc[t * j..(t + SPAN).min(frames) * j], &pred);
      let Some(skip) = span.iter().position(|&token| token != self.blank) else {
        t += span.len();
        continue;
      };
      t += skip;
      let mut token = span[skip];
      for _ in 0..MAX_SYMBOLS {
        out.push((token as u32, t as u32));
        next = self.lstm(&self.embed[token * n..][..n], &next);
        pred = self.predict(&next.h);
        token = self.joint(&enc[t * j..][..j], &pred)[0];
        if token == self.blank {
          break;
        }
      }
      t += 1;
    }
    out
  }

  fn lstm(&self, x: &[f32], s: &State) -> State {
    let n = self.hidden;
    let (gx, gh) = (matmul(&self.wx, x, n), matmul(&self.wh, &s.h, n));
    let gate = |i: usize| self.bias[i] + gx[i] + gh[i];
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
    let d = matmul(&self.pred_w, h, self.hidden);
    d.iter().zip(&self.pred_b).map(|(d, b)| b + d).collect()
  }

  /// Argmax of the joint logits for each frame in `enc`; ties go to the lowest id.
  fn joint(&self, enc: &[f32], pred: &[f32]) -> Vec<usize> {
    let z: Vec<f32> = (enc.chunks_exact(self.joint))
      .flat_map(|e| e.iter().zip(pred).map(|(e, p)| (e + p).max(0.0)))
      .collect();
    let logits = matmul(&self.out_w, &z, self.joint);
    (logits.chunks_exact(self.out_w.len() / self.joint))
      .map(|l| {
        let mut best = (0, f32::NEG_INFINITY);
        for (v, (d, b)) in l.iter().zip(&self.out_b).enumerate() {
          let logit = d + b;
          if logit > best.1 {
            best = (v, logit);
          }
        }
        best.0
      })
      .collect()
  }
}
