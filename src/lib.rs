// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Speech-to-text for the GGUF ASR models published by transcribe.cpp. The encoder runs on the GPU
//! through wgpu; the frontend and decoder run on the CPU.

mod conformer;
mod cpu;
mod error;
mod gguf;
mod gigaam;
mod gpu;
mod parakeet;

use std::path::Path;

use error::bail;
pub use error::{Error, Result};

pub struct Model(Family);

enum Family {
  Gigaam(Box<gigaam::Gigaam>),
  Parakeet(Box<parakeet::Parakeet>),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
  pub text: String,
  pub tokens: Vec<Token>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
  pub id: u32,
  /// Vocabulary piece: a SentencePiece piece (`▁` marks a word start) or a character.
  pub piece: String,
  /// Start time in milliseconds: the encoder frame where the token was emitted times the frame
  /// length (40 ms for GigaAM, 80 ms for Parakeet).
  pub start_ms: u32,
}

impl Model {
  pub fn load(path: impl AsRef<Path>) -> Result<Self> {
    let g = gguf::Gguf::open(path.as_ref())?;
    Ok(Self(match g.get::<&str>("general.architecture")? {
      "gigaam" => Family::Gigaam(Box::new(gigaam::Gigaam::load(&g)?)),
      "parakeet" => Family::Parakeet(Box::new(parakeet::Parakeet::load(&g)?)),
      arch => bail!("general.architecture is {arch:?}, only gigaam and parakeet are supported"),
    }))
  }

  /// Transcribes 16 kHz mono PCM samples in [-1, 1].
  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    match &self.0 {
      Family::Gigaam(m) => m.transcribe(pcm),
      Family::Parakeet(m) => m.transcribe(pcm),
    }
  }

  /// The longest audio, in milliseconds, the model was trained on. Longer audio still runs, but
  /// accuracy degrades, so split it into segments up to this length. `None`: no such window.
  pub fn max_audio_ms(&self) -> Option<u32> {
    match &self.0 {
      // No GGUF key holds GigaAM's window; transcribe.cpp hardcodes the same 25 s.
      Family::Gigaam(_) => Some(25_000),
      Family::Parakeet(_) => None,
    }
  }
}
