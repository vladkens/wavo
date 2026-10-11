// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// The hub links resolve to the module with the `hub` feature and to docs.rs without it; with the
// feature they name what rustdoc would find anyway, hence the allow.
#![allow(rustdoc::redundant_explicit_links)]
#![cfg_attr(
  feature = "hub",
  doc = "[hub-mod]: crate::hub\n[hub-models]: crate::hub::MODELS\n[hub-hub]: crate::hub::Hub\n[hub-pull]: crate::hub::Hub::pull"
)]
#![cfg_attr(
  not(feature = "hub"),
  doc = "[hub-mod]: https://docs.rs/wavo/latest/wavo/hub/\n[hub-models]: https://docs.rs/wavo/latest/wavo/hub/static.MODELS.html\n[hub-hub]: https://docs.rs/wavo/latest/wavo/hub/struct.Hub.html\n[hub-pull]: https://docs.rs/wavo/latest/wavo/hub/struct.Hub.html#method.pull"
)]
//! Speech-to-text on the GPU, in pure Rust.
//!
//! wavo runs the open ASR models that [transcribe.cpp] publishes as GGUF files on Hugging Face:
//! NVIDIA's Parakeet, Sber's GigaAM, OpenAI's Whisper and Useful Sensors' Moonshine, with the same
//! text and timestamps as transcribe.cpp. The encoder runs on the GPU through wgpu (Metal on Apple
//! Silicon, Vulkan on Linux and Windows) with wavo's own kernels; the audio frontend runs on the
//! CPU, and so does the decoder except Whisper's and Moonshine's. It builds with a plain `cargo
//! build`: no C++ toolchain, Python or ONNX Runtime.
//!
//! [transcribe.cpp]: https://github.com/handy-computer/transcribe.cpp
//!
//! # Features
//!
//! - `cli` (default): the `wavo` command-line tool. Leave it out when you use wavo as a library:
//!   `cargo add wavo --no-default-features`.
//! - `hub`: the [`hub`][hub-mod] module, which downloads and manages models in the Hugging Face
//!   cache. It adds an HTTP client (`ureq` with rustls, whose `ring` compiles some C); without it
//!   the library is pure Rust and never uses the network.
//!
//! # Models
//!
//! [`Model::load`] takes the Q8_0 GGUF file of one of these models. With the `hub` feature,
//! [`hub::MODELS`][hub-models] lists them, and [`hub::Hub`][hub-hub] downloads them by short or
//! full name.
//!
//! | Name | Full name | Size | Languages | Output |
//! |---|---|---|---|---|
//! | `parakeet-v3` | `parakeet-tdt-0.6b-v3` | 0.74 GB | 25 European languages | cased, punctuated |
//! | `parakeet-v2` | `parakeet-tdt-0.6b-v2` | 0.73 GB | English | cased, punctuated |
//! | `gigaam-v3` | `gigaam-v3-e2e-rnnt` | 0.27 GB | Russian | cased, punctuated |
//! | `gigaam-v3-e2e-ctc` | `gigaam-v3-e2e-ctc` | 0.27 GB | Russian | cased, punctuated |
//! | `gigaam-v3-rnnt` | `gigaam-v3-rnnt` | 0.27 GB | Russian | lowercase, no punctuation |
//! | `gigaam-v3-ctc` | `gigaam-v3-ctc` | 0.27 GB | Russian | lowercase, no punctuation |
//! | `whisper-turbo` | `whisper-large-v3-turbo` | 0.89 GB | 100 languages | cased, punctuated |
//! | `whisper-tiny` | `whisper-tiny` | 0.05 GB | 99 languages | cased, punctuated |
//! | `whisper-base` | `whisper-base` | 0.08 GB | 99 languages | cased, punctuated |
//! | `whisper-small` | `whisper-small` | 0.27 GB | 99 languages | cased, punctuated |
//! | `whisper-medium` | `whisper-medium` | 0.83 GB | 99 languages | cased, punctuated |
//! | `whisper-large-v3` | `whisper-large-v3` | 1.67 GB | 100 languages | cased, punctuated |
//! | `whisper-tiny.en` | `whisper-tiny.en` | 0.05 GB | English | cased, punctuated |
//! | `whisper-base.en` | `whisper-base.en` | 0.08 GB | English | cased, punctuated |
//! | `whisper-small.en` | `whisper-small.en` | 0.27 GB | English | cased, punctuated |
//! | `whisper-medium.en` | `whisper-medium.en` | 0.83 GB | English | cased, punctuated |
//! | `whisper-large` | `whisper-large` | 1.67 GB | 99 languages | cased, punctuated |
//! | `whisper-large-v2` | `whisper-large-v2` | 1.67 GB | 99 languages | cased, punctuated |
//! | `breeze-asr-25` | `Breeze-ASR-25` | 1.67 GB | Taiwanese Mandarin and English | cased, punctuated |
//! | `moonshine-tiny` | `moonshine-tiny` | 0.04 GB | English | cased, punctuated |
//! | `moonshine-base` | `moonshine-base` | 0.08 GB | English | cased, punctuated |
//!
//! A model `<full name>` lives in the Hugging Face repository `handy-computer/<full name>-gguf`
//! as `<full name>-Q8_0.gguf`.
//!
//! # Transcribe
//!
//! [`Model::transcribe`] takes 16 kHz mono samples as `f32` in [-1, 1], and returns the text and
//! its tokens with start times in milliseconds:
//!
//! ```no_run
//! # fn main() -> wavo::Result<()> {
//! let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
//! let pcm = vec![0.0f32; 16_000]; // one second of silence: 16 kHz mono in [-1, 1]
//! let transcript = model.transcribe(&pcm)?;
//! println!("{}", transcript.text);
//! for token in &transcript.tokens {
//!   println!("{:>6} ms  {}", token.start_ms, token.piece);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Load a model once and call `transcribe` as often as needed; loading takes about 0.1–0.2 s on
//! an Apple M2.
//!
//! # Download a model
//!
//! With `cargo add wavo --no-default-features --features hub`, [`hub::Hub`][hub-hub] finds a model
//! in the Hugging Face cache that `hf` and `huggingface_hub` share, and downloads it when missing:
//!
//! ```no_run
//! # #[cfg(feature = "hub")]
//! # fn main() -> wavo::Result<()> {
//! use std::ops::ControlFlow;
//!
//! let hub = wavo::hub::Hub::new(); // or Hub::at(dir) for a folder of your app's own
//! let path = match hub.find("gigaam-v3") {
//!   Some(path) => path, // no network
//!   None => hub.pull("gigaam-v3", |_| ControlFlow::Continue(()))?,
//! };
//! let model = wavo::Model::load(path)?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "hub"))]
//! # fn main() {}
//! ```
//!
//! # Download on a thread, with progress and cancel
//!
//! [`Hub::pull`][hub-pull] blocks until the file is complete, and the library starts no
//! threads of its own: run it on a thread, with [`std::thread::spawn`] as here, or with tokio's
//! `spawn_blocking`. Its callback sees the bytes done and the total, and returns
//! [`ControlFlow::Break`](std::ops::ControlFlow::Break) to cancel; the partial file stays in the
//! cache, and the next pull resumes it.
//!
//! ```no_run
//! # #[cfg(feature = "hub")]
//! # fn main() {
//! use std::ops::ControlFlow;
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicBool, Ordering};
//!
//! let cancel = Arc::new(AtomicBool::new(false));
//! let stop = Arc::clone(&cancel);
//! let download = std::thread::spawn(move || {
//!   wavo::hub::Hub::new().pull("parakeet-v3", |p| {
//!     if let Some(e) = p.retrying {
//!       eprintln!("\nretrying after: {e}");
//!     }
//!     eprint!("\r{} / {} MB", p.done / 1_000_000, p.total / 1_000_000);
//!     if stop.load(Ordering::Relaxed) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
//!   })
//! });
//! // A cancel button elsewhere runs: cancel.store(true, Ordering::Relaxed);
//! match download.join().unwrap() {
//!   Ok(path) => println!("\ndownloaded {}", path.display()),
//!   Err(wavo::Error::Cancelled) => println!("\ncancelled; the next pull resumes"),
//!   Err(e) => eprintln!("\ndownload failed: {e}"),
//! }
//! # }
//! # #[cfg(not(feature = "hub"))]
//! # fn main() {}
//! ```
//!
//! # Left to the caller
//!
//! - **Decoding and resampling.** `transcribe` takes 16 kHz mono `f32` only. Decode other formats,
//!   downmix to mono and resample yourself; the `wavo` CLI uses `symphonia` and `rubato`.
//! - **Long audio.** `transcribe` takes audio of any length (Whisper runs its own 30 s windows),
//!   but accuracy drops past [`Model::max_audio_ms`]. Split longer audio at pauses, transcribe
//!   the pieces and shift their tokens' `start_ms` by each piece's offset, as `wavo run` does
//!   in [`src/cli/split.rs`](https://github.com/vladkens/wavo/blob/main/src/cli/split.rs): it
//!   cuts at the quietest moments by frame energy.
//! - **Voice activity detection and streaming.** Neither is built in: pass whole recordings or the
//!   pieces your own VAD cuts.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

mod conformer;
mod cpu;
mod error;
mod fft;
mod gguf;
mod gigaam;
mod gpu;
#[cfg(feature = "hub")]
#[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
pub mod hub;
mod moonshine;
mod parakeet;
mod transformer;
mod whisper;

use std::path::Path;

use error::bail;
pub use error::{Error, Result};

/// A speech-to-text model loaded on the GPU.
///
/// Load it once with [`Model::load`], then call [`Model::transcribe`] for each recording. It is
/// `Send` and `Sync`: share it between threads, and concurrent calls take turns on its GPU
/// buffers.
pub struct Model(Family);

enum Family {
  Gigaam(Box<gigaam::Gigaam>),
  Parakeet(Box<parakeet::Parakeet>),
  Whisper(Box<whisper::Whisper>),
  Moonshine(Box<moonshine::Moonshine>),
}

/// The text of a recording and the tokens it is made of.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Transcript {
  /// The whole text; empty when nothing was recognized.
  pub text: String,
  /// The tokens in order: the model's vocabulary pieces the text is made of, with start times.
  pub tokens: Vec<Token>,
}

/// One token of a [`Transcript`].
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Token {
  /// Id in the model's vocabulary.
  pub id: u32,
  /// Vocabulary piece: a SentencePiece piece (`▁` marks a word start), a character, or for
  /// Whisper and Moonshine's byte tokens the text the token completes (its bytes may end inside a
  /// character).
  pub piece: String,
  /// Start time in milliseconds from the start of the audio: the encoder frame where the token
  /// was emitted times the frame length (40 ms for GigaAM, 80 ms for Parakeet); for Whisper the
  /// start of its segment; 0 for Moonshine, which has no timestamps.
  pub start_ms: u32,
}

impl Model {
  /// Loads a model from its GGUF file and uploads its weights to the GPU.
  ///
  /// The file is one of the Q8_0 GGUFs transcribe.cpp publishes (see the
  /// [model table](crate#models)); the family comes from its `general.architecture`.
  ///
  /// # Errors
  ///
  /// - [`Error::Io`]: the file can't be read.
  /// - [`Error::Model`]: not a GGUF, an architecture other than `gigaam`, `parakeet`, `whisper` and
  ///   `moonshine`, or a missing or mismatched tensor or metadata key.
  /// - [`Error::Adapter`], [`Error::Device`]: no GPU, or not one the model can run on.
  ///
  /// # Examples
  ///
  /// ```no_run
  /// let model = wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf")?;
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn load(path: impl AsRef<Path>) -> Result<Self> {
    let g = gguf::Gguf::open(path.as_ref())?;
    Ok(Self(match g.get::<&str>("general.architecture")? {
      "gigaam" => Family::Gigaam(Box::new(gigaam::Gigaam::load(&g)?)),
      "parakeet" => Family::Parakeet(Box::new(parakeet::Parakeet::load(&g)?)),
      "whisper" => Family::Whisper(Box::new(whisper::Whisper::load(&g)?)),
      "moonshine" => Family::Moonshine(Box::new(moonshine::Moonshine::load(&g)?)),
      arch => bail!(
        "general.architecture is {arch:?}, only gigaam, parakeet, whisper and moonshine are supported"
      ),
    }))
  }

  /// Transcribes 16 kHz mono PCM samples in [-1, 1].
  ///
  /// Any length runs, but past [`max_audio_ms`](Self::max_audio_ms) accuracy drops: split long
  /// audio first (see [Left to the caller](crate#left-to-the-caller)). Audio too short for one
  /// encoder frame gives an empty transcript.
  ///
  /// # Errors
  ///
  /// The GPU failed: [`Error::Gpu`], [`Error::Poll`], [`Error::Readback`] or [`Error::Map`].
  ///
  /// # Examples
  ///
  /// ```no_run
  /// # let model = wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf")?;
  /// # let pcm = vec![0.0f32; 16_000];
  /// let transcript = model.transcribe(&pcm)?;
  /// let words = transcript.text.split(' ').count();
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn transcribe(&self, pcm: &[f32]) -> Result<Transcript> {
    match &self.0 {
      Family::Gigaam(m) => m.transcribe(pcm),
      Family::Parakeet(m) => m.transcribe(pcm),
      Family::Whisper(m) => m.transcribe(pcm),
      Family::Moonshine(m) => m.transcribe(pcm),
    }
  }

  /// The longest audio, in milliseconds, the model was trained on: 25 000 for GigaAM, 30 000 for
  /// Whisper and Moonshine. Longer audio still runs, but accuracy degrades, so split it into
  /// segments up to this length. `None` for Parakeet, which has no such window.
  pub fn max_audio_ms(&self) -> Option<u32> {
    match &self.0 {
      // No GGUF key holds GigaAM's window; transcribe.cpp hardcodes the same 25 s.
      Family::Gigaam(_) => Some(25_000),
      // Its window; `transcribe` still takes any length in 30 s windows, as transcribe.cpp does.
      Family::Whisper(_) => Some(30_000),
      // transcribe.cpp advises 48.5 s; 30 s keeps fast speech under the 194-token cap.
      Family::Moonshine(_) => Some(30_000),
      Family::Parakeet(_) => None,
    }
  }
}

#[cfg(test)]
mod tests {
  #[test]
  fn model_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<super::Model>();
  }
}
