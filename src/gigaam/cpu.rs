// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoder on the CPU, as `encoder.rs` runs it on the GPU: the two conv1d + ReLU subsampling,
//! the Conformer blocks and the head's linear layer.

use super::Config;
use crate::cpu::conformer::Conformer;
use crate::cpu::gemm::{Linear, gemm};
use crate::cpu::ops::im2col;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gpu::{Epilogue, half};

pub struct Encoder {
  mels: usize,
  theta: f64,
  conv0: Linear,
  conv2: Linear,
  conformer: Conformer,
  head: Linear,
}

impl Encoder {
  pub fn new(g: &Gguf, cfg: Config) -> Result<Self> {
    let (d, mels) = (cfg.conformer.d, cfg.mels);
    let linear = |name: &str, dims: &[usize]| {
      let bias = g.tensor(&format!("{name}.bias"), &dims[dims.len() - 1..])?.to_f32()?;
      Linear::new(&[g.tensor(&format!("{name}.weight"), dims)?], bias)
    };
    let (name, dims) = &cfg.head;
    Ok(Self {
      mels,
      theta: cfg.theta,
      conv0: linear("enc.pre_encode.conv.0", &[5, mels, d])?,
      conv2: linear("enc.pre_encode.conv.2", &[5, d, d])?,
      head: linear(name, dims)?,
      conformer: Conformer::load(g, cfg.conformer)?,
    })
  }

  /// Encodes `mel` `[mel frames][mels]` and returns the head's outputs `[frames][outputs]`.
  pub fn run(&self, mel: &[f32]) -> Vec<f32> {
    let c = &self.conformer.cfg;
    let (d, hd) = (c.d, c.head_dim());
    let mel_frames = mel.len() / self.mels;
    let (t1, t) = (half(mel_frames), half(half(mel_frames)));
    let mut h = vec![0.0; t1 * d];
    gemm(&im2col(mel, mel_frames, self.mels, [5, 2, 2]), &self.conv0, &mut h, t1, Epilogue::Relu);
    let mut x = vec![0.0; t * d];
    gemm(&im2col(&h, t1, d, [5, 2, 2]), &self.conv2, &mut x, t, Epilogue::Relu);
    let x = self.conformer.run(x, t, &super::rope(t, hd, self.theta));
    let mut out = vec![0.0; t * self.head.n];
    gemm(&x, &self.head, &mut out, t, Epilogue::Bias);
    out
  }
}
