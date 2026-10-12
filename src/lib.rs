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
//! Speech-to-text on the GPU or a Mac's Neural Engine, in pure Rust.
//!
//! wavo runs the open ASR models that [transcribe.cpp] publishes as GGUF files on Hugging Face:
//! NVIDIA's Parakeet, Sber's GigaAM, OpenAI's Whisper and Useful Sensors' Moonshine, with the same
//! text and timestamps as transcribe.cpp. The encoder runs on the GPU through wgpu (Metal on Apple
//! Silicon, Vulkan on Linux and Windows) with wavo's own kernels, or for Parakeet and GigaAM on
//! the Neural Engine of an Apple Silicon Mac through Core ML ([below](#neural-engine)); the audio
//! frontend runs on the CPU, and so does the decoder except Whisper's and Moonshine's. It builds
//! with a plain `cargo build`: no C++ toolchain, Python or ONNX Runtime.
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
//! # Long audio
//!
//! `transcribe` takes audio of any length (Whisper runs its own 30 s windows), but accuracy drops
//! past [`Model::max_audio_ms`]. [`Model::transcribe_long`] splits longer audio at its quietest
//! moments into pieces of up to that length, transcribes them and joins the transcripts, with
//! token times from the start of the audio. To transcribe the pieces yourself, for example with
//! progress, take them from [`split`] and shift each piece's token `start_ms` by its start.
//!
//! # Neural Engine
//!
//! On an Apple Silicon Mac with macOS 15 or later, Parakeet's and GigaAM's encoders also run on
//! the Neural Engine: on an M1 a call over 11 s of audio takes 52–70 ms there against 170–250 ms
//! on the GPU. The model needs preparing once, which writes its encoder as a Core ML program and
//! has the system compile it (15–30 s, and as much disk as the GGUF):
//! [`Hub::pull`][hub-pull] does it after the download, and [`Model::prepare`] does it for a file
//! you manage. The system keeps its Neural Engine compile per program, so each program prepares
//! for itself: in a program that prepared before, `prepare` only loads the model (under a
//! second); in another one, such as yours after `wavo pull`, it reuses the compiled model and
//! loads it once for 10–30 s. [`Model::load`] then runs the model on the Neural Engine, and on the
//! GPU in a program that hasn't prepared it; [`Model::load_with`] picks a [`Backend`], and
//! [`Model::backend`] tells which one runs.
//!
//! ```no_run
//! # fn main() -> wavo::Result<()> {
//! use std::ops::ControlFlow;
//!
//! let path = "parakeet-tdt-0.6b-v3-Q8_0.gguf";
//! wavo::Model::prepare(path, |step| {
//!   eprintln!("preparing for the Neural Engine: {step:?}");
//!   ControlFlow::Continue(())
//! })?; // `false` without a Neural Engine: the model runs on the GPU
//! let model = wavo::Model::load(path)?;
//! println!("{:?}, pieces of up to {:?} ms", model.backend(), model.max_audio_ms());
//! # Ok(())
//! # }
//! ```
//!
//! The Neural Engine runs a fixed 15 s window in F16 with 8-bit weights: shorter audio is padded,
//! longer audio split at pauses. On the test clips the words match the GPU's, while now and then a
//! comma or a capital letter differs and a few tokens start one or two encoder frames apart. The
//! GPU stays the exact path.
//!
//! # Left to the caller
//!
//! - **Decoding and resampling.** `transcribe` takes 16 kHz mono `f32` only. Decode other formats,
//!   downmix to mono and resample yourself; the `wavo` CLI uses `symphonia` and `rubato`.
//! - **Voice activity detection and streaming.** Neither is built in: pass whole recordings or the
//!   pieces your own VAD cuts.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

mod ane;
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
mod split;
mod transformer;
mod whisper;

use std::ops::ControlFlow;
use std::path::Path;

use error::bail;
pub use error::{Error, Result};
pub use split::split;

/// A speech-to-text model loaded on the GPU or the Neural Engine.
///
/// Load it once with [`Model::load`], then call [`Model::transcribe`] for each recording. It is
/// `Send` and `Sync`: share it between threads, and concurrent calls take turns.
pub struct Model {
  family: Family,
  backend: Backend,
}

enum Family {
  Gigaam(Box<gigaam::Gigaam>),
  Parakeet(Box<parakeet::Parakeet>),
  Whisper(Box<whisper::Whisper>),
  Moonshine(Box<moonshine::Moonshine>),
}

/// Where a model's encoder runs, chosen when it loads ([`Model::load_with`]). The audio frontend
/// and the decoder run on the CPU either way (Whisper's and Moonshine's decoders on the GPU).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Backend {
  /// The faster one for the model's family on this machine: the Neural Engine for Parakeet and
  /// GigaAM once [prepared](Model::prepare) in this program, else the GPU. [`Model::load`] uses
  /// it.
  ///
  /// The system caches the Neural Engine's compile of a model per program, and a load without
  /// that cache takes 10–30 s. A new build of the program or of macOS counts as a new program
  /// (it runs on the GPU until it prepares), but when the system drops its cache on its own, for
  /// example to free disk space, the next load takes that long once.
  #[default]
  Auto,
  /// The GPU through wgpu, on every platform: the same text, tokens and timestamps as
  /// transcribe.cpp.
  Gpu,
  /// Apple's Neural Engine through Core ML, on Apple Silicon with macOS 15 or later, for Parakeet
  /// and GigaAM once [prepared](Model::prepare). It runs in F16 with 8-bit weights in a fixed
  /// 15 s window: on the test clips the words match the GPU's, while now and then a comma or a
  /// capital letter differs and some tokens start one or two encoder frames apart.
  NeuralEngine,
}

/// The backend [`Backend::Auto`] prefers per `general.architecture`: on an M1 and M2 the Neural
/// Engine runs Parakeet's and GigaAM's encoder 4–5× faster than the GPU, Whisper's slower.
const AUTO: &[(&str, Backend)] = &[
  ("parakeet", Backend::NeuralEngine),
  ("gigaam", Backend::NeuralEngine),
  ("whisper", Backend::Gpu),
  ("moonshine", Backend::Gpu),
];

/// The families with a Neural Engine program, and the function that writes it.
const NEURAL_ENGINE: &[(&str, ane::Build)] =
  &[("parakeet", parakeet::ane::build), ("gigaam", gigaam::ane::build)];

/// What [`Model::prepare`] is doing, passed to its callback when each step starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrepareStep {
  /// Writing the encoder as a Core ML program from the GGUF file: seconds, and as much free disk
  /// as the model file, freed at the end.
  Build,
  /// Core ML compiling the program: seconds.
  Compile,
  /// Loading it once, which makes the system compile it for the Neural Engine: 10–30 s the first
  /// time, under a second once the system has it cached.
  Load,
}

/// The longest piece [`Model::transcribe_long`] makes for a model without a window, in ms.
const LONG_MS: u32 = 60_000;

impl Model {
  /// Loads a model from its GGUF file on the faster backend for its family ([`Backend::Auto`]).
  ///
  /// The file is one of the Q8_0 GGUFs transcribe.cpp publishes (see the
  /// [model table](crate#models)); the family comes from its `general.architecture`. Parakeet and
  /// GigaAM load on the Neural Engine (in about 0.1 s) when this program has
  /// [prepared](Model::prepare) the file on a Mac that has one, and on the GPU otherwise.
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
    Self::load_with(path, Backend::Auto)
  }

  /// Loads a model on the given backend; see [`Model::load`].
  ///
  /// # Errors
  ///
  /// Those of [`Model::load`], and [`Error::NeuralEngine`] for [`Backend::NeuralEngine`] when this
  /// machine has no Neural Engine, the family has no Neural Engine program, the file isn't
  /// [prepared](Model::prepare), or Core ML fails to load it.
  ///
  /// # Examples
  ///
  /// The GPU, whatever is prepared:
  ///
  /// ```no_run
  /// use wavo::{Backend, Model};
  ///
  /// let model = Model::load_with("parakeet-tdt-0.6b-v3-Q8_0.gguf", Backend::Gpu)?;
  /// assert_eq!(model.backend(), Backend::Gpu);
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn load_with(path: impl AsRef<Path>, backend: Backend) -> Result<Self> {
    let path = path.as_ref();
    let g = gguf::Gguf::open(path)?;
    let arch = g.get::<&str>("general.architecture")?;
    let ane = match backend {
      Backend::Gpu => None,
      Backend::NeuralEngine => Some(neural_engine(path, arch)?),
      // A file that isn't prepared in this program, or that Core ML fails to load, runs on the
      // GPU: a first load in this program would compile for the Neural Engine for 10–30 s.
      Backend::Auto => (auto(arch) == Backend::NeuralEngine && ane::prepared(path))
        .then(|| neural_engine(path, arch).ok())
        .flatten(),
    };
    let backend = if ane.is_some() { Backend::NeuralEngine } else { Backend::Gpu };
    let family = match arch {
      "gigaam" => Family::Gigaam(Box::new(gigaam::Gigaam::load(&g, ane)?)),
      "parakeet" => Family::Parakeet(Box::new(parakeet::Parakeet::load(&g, ane)?)),
      "whisper" => Family::Whisper(Box::new(whisper::Whisper::load(&g)?)),
      "moonshine" => Family::Moonshine(Box::new(moonshine::Moonshine::load(&g)?)),
      arch => bail!(
        "general.architecture is {arch:?}, only gigaam, parakeet, whisper and moonshine are supported"
      ),
    };
    Ok(Self { family, backend })
  }

  /// The backend the model runs on: [`Backend::Gpu`] or [`Backend::NeuralEngine`], never
  /// `Auto`.
  pub fn backend(&self) -> Backend {
    self.backend
  }

  /// Transcribes 16 kHz mono PCM samples in [-1, 1].
  ///
  /// Any length runs, but past [`max_audio_ms`](Self::max_audio_ms) accuracy drops: use
  /// [`transcribe_long`](Self::transcribe_long), or split long audio yourself with [`split`].
  /// On the Neural Engine, audio longer than its 15 s window is split that way here. Audio too
  /// short for one encoder frame gives an empty transcript.
  ///
  /// # Errors
  ///
  /// The GPU failed: [`Error::Gpu`], [`Error::Poll`], [`Error::Readback`] or [`Error::Map`]; or
  /// Core ML did: [`Error::NeuralEngine`].
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
    if self.backend == Backend::NeuralEngine && pcm.len() > ane::WINDOW {
      return self.transcribe_long(pcm);
    }
    match &self.family {
      Family::Gigaam(m) => m.transcribe(pcm),
      Family::Parakeet(m) => m.transcribe(pcm),
      Family::Whisper(m) => m.transcribe(pcm),
      Family::Moonshine(m) => m.transcribe(pcm),
    }
  }

  /// Transcribes audio of any length: split at pauses by [`split`] into pieces of up to
  /// [`max_audio_ms`](Self::max_audio_ms) (60 s for a model without a window), each transcribed
  /// on its own, joined with a space between the texts and token times from the start of `pcm`.
  ///
  /// # Errors
  ///
  /// Those of [`Model::transcribe`].
  ///
  /// # Examples
  ///
  /// ```no_run
  /// # let model = wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf")?;
  /// let hour = vec![0.0f32; 16_000 * 3600];
  /// let transcript = model.transcribe_long(&hour)?;
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn transcribe_long(&self, pcm: &[f32]) -> Result<Transcript> {
    let pieces = split(pcm, self.max_audio_ms().unwrap_or(LONG_MS));
    let parts = pieces.into_iter().map(|r| Ok((r.start, self.transcribe(&pcm[r])?)));
    Ok(split::join(parts.collect::<Result<Vec<_>>>()?))
  }

  /// The longest audio, in milliseconds, the model takes in one piece: 15 000 on the Neural
  /// Engine, its fixed window; else the length it was trained on, 25 000 for GigaAM and 30 000
  /// for Whisper and Moonshine. Longer audio still runs, but accuracy degrades, so split it into
  /// pieces up to this length ([`Model::transcribe_long`] does). `None` for Parakeet on the GPU,
  /// which has no such window.
  pub fn max_audio_ms(&self) -> Option<u32> {
    if self.backend == Backend::NeuralEngine {
      return Some((ane::WINDOW / 16) as u32);
    }
    match &self.family {
      // No GGUF key holds GigaAM's window; transcribe.cpp hardcodes the same 25 s.
      Family::Gigaam(_) => Some(25_000),
      // Its window; `transcribe` still takes any length in 30 s windows, as transcribe.cpp does.
      Family::Whisper(_) => Some(30_000),
      // transcribe.cpp advises 48.5 s; 30 s keeps fast speech under the 194-token cap.
      Family::Moonshine(_) => Some(30_000),
      Family::Parakeet(_) => None,
    }
  }

  /// Prepares a model file for the Neural Engine, so that [`Model::load`] runs it there: writes
  /// its encoder as a Core ML program, has Core ML compile it, and loads it once. Returns `true`
  /// when the file is prepared, `false` when there is nothing to do: a family that runs on the
  /// GPU only (Whisper, Moonshine), or a machine without a Neural Engine (not macOS, macOS before
  /// 15, an Intel Mac, a virtual machine).
  ///
  /// It blocks for 15–40 s the first time, mostly while the system compiles the program for the
  /// Neural Engine: run it on a thread of your own, as `Hub::pull` (the `hub` feature), which calls
  /// it after each download, does. The system keeps that last compile per program, so prepare in
  /// the program that loads the model: once it has prepared there, `prepare` returns after one
  /// load of under a second; when another program prepared the file (`wavo pull`, a helper that
  /// downloads), it skips the build and spends the 10–30 s on that load. The
  /// compiled model takes about as much disk as the GGUF, in `~/Library/Caches/wavo/`, named by
  /// a hash of the file (size, modification time and some content), so the same file found
  /// through another path or cache folder shares it. [`Model::unprepare`] deletes it,
  /// and `prepare` deletes compiled models no load has used for 30 days (of files deleted
  /// elsewhere). `progress` is called as each step starts; return [`ControlFlow::Break`] to
  /// cancel before the next.
  ///
  /// # Errors
  ///
  /// - [`Error::Io`]: the file can't be read, or the cache folder can't be written.
  /// - [`Error::Model`]: not a GGUF wavo supports.
  /// - [`Error::NeuralEngine`]: Core ML failed to compile or load the program.
  /// - [`Error::Cancelled`]: `progress` returned [`ControlFlow::Break`].
  ///
  /// # Examples
  ///
  /// ```no_run
  /// use std::ops::ControlFlow;
  ///
  /// let path = "parakeet-tdt-0.6b-v3-Q8_0.gguf";
  /// if wavo::Model::prepare(path, |step| {
  ///   eprintln!("{step:?}");
  ///   ControlFlow::Continue(())
  /// })? {
  ///   assert_eq!(wavo::Model::load(path)?.backend(), wavo::Backend::NeuralEngine);
  /// }
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn prepare(
    path: impl AsRef<Path>,
    progress: impl FnMut(PrepareStep) -> ControlFlow<()>,
  ) -> Result<bool> {
    let path = path.as_ref();
    let g = gguf::Gguf::open(path)?;
    let arch = g.get::<&str>("general.architecture")?;
    let Some(&(_, build)) = NEURAL_ENGINE.iter().find(|(a, _)| *a == arch) else {
      return Ok(false);
    };
    ane::prepare(path, &g, build, progress)
  }

  /// Whether a model file is [prepared](Model::prepare) for the Neural Engine in this program:
  /// `Some(true)` when [`Model::load`] runs it there, `Some(false)` when `prepare` would make it
  /// so, `None` when the family or this machine has no Neural Engine path (or the file can't be
  /// read). Another program's prepare doesn't count: the system caches the Neural Engine's
  /// compile per program, so `prepare` in this one is fast (no rebuild) but loads once for
  /// 10–30 s.
  pub fn prepared(path: impl AsRef<Path>) -> Option<bool> {
    let path = path.as_ref();
    let g = gguf::Gguf::open(path).ok()?;
    let arch = g.get::<&str>("general.architecture").ok()?;
    (ane::available() && NEURAL_ENGINE.iter().any(|(a, _)| *a == arch)).then(|| ane::prepared(path))
  }

  /// Deletes what [`Model::prepare`] made for a model file, and returns the bytes freed (0 when
  /// it wasn't prepared). The file itself stays.
  ///
  /// # Errors
  ///
  /// [`Error::Io`]: the file can't be read, or deleting failed.
  pub fn unprepare(path: impl AsRef<Path>) -> Result<u64> {
    ane::unprepare(path.as_ref())
  }
}

fn auto(arch: &str) -> Backend {
  AUTO.iter().find(|(a, _)| *a == arch).map_or(Backend::Gpu, |(_, b)| *b)
}

fn neural_engine(path: &Path, arch: &str) -> Result<ane::Model> {
  if !NEURAL_ENGINE.iter().any(|(a, _)| *a == arch) {
    return Err(Error::NeuralEngine(format!("{arch} models have no Neural Engine program")));
  }
  ane::load(path)
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn model_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Model>();
  }

  #[test]
  fn every_family_has_a_preference() {
    for arch in ["gigaam", "parakeet", "whisper", "moonshine"] {
      assert!(AUTO.iter().any(|(a, _)| *a == arch), "{arch}");
    }
    for (arch, _) in NEURAL_ENGINE {
      assert_eq!(auto(arch), Backend::NeuralEngine, "{arch} has a program Auto never uses");
    }
  }
}
