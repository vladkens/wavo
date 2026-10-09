// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Speech-to-text for the GGUF ASR models published by transcribe.cpp. The encoder runs on the GPU
//! through wgpu; the frontend and decoder run on the CPU.

mod error;
mod gguf;
mod gigaam;
mod gpu;

use std::path::Path;

pub use error::{Error, Result};

pub struct Model(gigaam::Gigaam);

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
  pub text: String,
  pub tokens: Vec<Token>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
  pub id: u32,
  /// SentencePiece piece, `▁` marks a word start.
  pub piece: String,
  /// Encoder frame where the token was emitted (40 ms per frame for GigaAM).
  pub frame: u32,
}

impl Model {
  pub fn load(path: impl AsRef<Path>) -> Result<Self> {
    let g = gguf::Gguf::open(path.as_ref())?;
    Ok(Self(gigaam::Gigaam::load(&g)?))
  }

  /// Transcribes 16 kHz mono PCM samples in [-1, 1].
  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    self.0.transcribe(pcm)
  }
}
