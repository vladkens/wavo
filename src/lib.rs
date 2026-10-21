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
//! Silicon, Vulkan on Linux and Windows, DX12 on Windows without a Vulkan driver) with wavo's own
//! kernels; the audio frontend runs on the CPU, and so does the decoder except Whisper's and
//! Moonshine's. It builds with a plain `cargo build`: no C++ toolchain, Python or ONNX Runtime.
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
//! | `parakeet-ultra` | `parakeet-ultra` | 0.74 GB | 25 European languages | cased, punctuated |
//! | `parakeet-unified` | `parakeet-unified-en-0.6b` | 0.73 GB | English | cased, punctuated |
//! | `parakeet-he` | `parakeet-tdt-0.6b-v3-he` | 0.74 GB | Hebrew | cased, punctuated |
//! | `gigaam-v3` | `gigaam-v3-e2e-rnnt` | 0.27 GB | Russian | cased, punctuated |
//! | `gigaam-v3-e2e-ctc` | `gigaam-v3-e2e-ctc` | 0.27 GB | Russian | cased, punctuated |
//! | `gigaam-v3-rnnt` | `gigaam-v3-rnnt` | 0.27 GB | Russian | lowercase, no punctuation |
//! | `gigaam-v3-ctc` | `gigaam-v3-ctc` | 0.27 GB | Russian | lowercase, no punctuation |
//! | `whisper-turbo` | `whisper-large-v3-turbo` | 0.89 GB | 100 languages | cased, punctuated |
//! | `whisper-turbo-he` | `ivrit-whisper-large-v3-turbo` | 0.89 GB | Hebrew | cased, punctuated |
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
//! A model `<full name>` lives in the Hugging Face repository `<owner>/<full name>-gguf` as
//! `<full name>-Q8_0.gguf`. The owner is `handy-computer`, transcribe.cpp's account, except for the
//! Hebrew models: transcribe.cpp doesn't publish them, so their conversions are under `vladkens`.
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
//!
//! println!("{}", transcript.text);
//!
//! for token in &transcript.tokens {
//!   println!("{:>6} ms  {}", token.start_ms, token.piece);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`Model::transcribe_with`] takes [`Options`] as well: the language of the speech, for Whisper
//! to transcribe instead of detecting one. Both run the audio through the model as it is, in one
//! pass: for recordings longer than the model's window, see [Long audio](#long-audio).
//!
//! Load a model once and call `transcribe` as often as needed; loading takes about 0.1–0.2 s on
//! an Apple M2. [`Model::device`] names the GPU and API it runs on, for a log line or a settings
//! screen, and [`devices`] lists the GPUs before any model is loaded.
//!
//! # Long audio
//!
//! Models are trained on pieces of up to [`Model::max_audio_ms`]: 25 s for GigaAM, 30 s for
//! Whisper and Moonshine. Past that, `transcribe` loses accuracy, and GigaAM and Moonshine return
//! [`Error::AudioTooLong`] past 1.5 times it; Whisper runs its own 30 s windows and Parakeet takes
//! any length in one pass. [`Model::transcribe_long`] takes audio of any length: it cuts it at its
//! quietest moments with [`split`] into pieces of up to the model's window (60 s for Parakeet, or
//! [`Options::window_ms`]), transcribes each, and joins the transcripts with token times from the
//! start of the audio.
//!
//! ```no_run
//! # fn main() -> wavo::Result<()> {
//! let model = wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf")?;
//! let pcm = vec![0.0f32; 16_000 * 600]; // ten minutes
//! let transcript = model.transcribe_long(&pcm, &wavo::Options::default())?;
//! # Ok(())
//! # }
//! ```
//!
//! To show progress or cancel between pieces, cut with [`split`] and call `transcribe_with` per
//! piece yourself, shifting each piece's token `start_ms` by its start (`range.start / 16`).
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
//!
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
mod sentencepiece;
mod split;
mod transformer;
mod whisper;

use std::path::Path;

use error::bail;
pub use error::{Error, Result};
pub use gpu::{Api, Device, DeviceKind, Kernels, devices};
pub use split::split;

/// The code path of the CPU matrix products (`NEON`, `AVX2+FMA` or `portable`), for `wavo debug`;
/// not part of the API.
#[doc(hidden)]
pub fn cpu_path() -> &'static str {
  cpu::path()
}

/// A speech-to-text model loaded on the GPU.
///
/// Load it once with [`Model::load`], then call [`Model::transcribe`] for each recording. It is
/// `Send` and `Sync`: share it between threads, and concurrent calls take turns on its GPU
/// buffers.
pub struct Model {
  family: Family,
  /// The GGUF's `general.languages`.
  languages: Vec<String>,
}

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

/// Options for [`Model::transcribe_with`] and [`Model::transcribe_long`]. Start from
/// `Options::default()`, which [`Model::transcribe`] uses, and set the fields to change.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Options {
  /// The language of the speech, as one of the codes the model lists (its GGUF's
  /// `general.languages`, which `hub::ModelInfo::langs` repeats): `en`, `he`, `zh`. `None`, the
  /// default, leaves it to the model.
  ///
  /// A multilingual Whisper model then transcribes in this language instead of detecting one
  /// from the first 30 s: on short or accented speech detection can pick a close language, such
  /// as Arabic for Hebrew or Russian for Ukrainian. Told a language other than the speech's,
  /// Whisper writes that language anyway, often as a rough translation. A model of one language
  /// (GigaAM, Parakeet v2, Moonshine, Whisper `.en`) takes its own code and runs as without it.
  /// Parakeet v3 has no way to be told a language, so any code fails with it, as does a code the
  /// model doesn't list: see [`Error::Language`].
  pub language: Option<String>,
  /// The longest piece, in milliseconds, [`Model::transcribe_long`] cuts the audio into with
  /// [`split`]. `None`, the default, takes the model's [`max_audio_ms`](Model::max_audio_ms), or
  /// 60 000 for Parakeet, which has none. Below 5000 counts as 5000; `u32::MAX` runs the whole
  /// audio in one piece, as `transcribe_with` does. GigaAM and Moonshine fail on pieces past 1.5
  /// times their window: see [`Error::AudioTooLong`]. `transcribe_with` ignores it.
  pub window_ms: Option<u32>,
}

/// One token of a [`Transcript`].
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Token {
  /// Id in the model's vocabulary.
  pub id: u32,
  /// Vocabulary piece: a SentencePiece piece (`▁` marks a word start), a character, or for byte
  /// tokens (Whisper's, and SentencePiece's `<0xHH>` fallback) the text the token completes (its
  /// bytes may end inside a character, leaving the piece empty).
  pub piece: String,
  /// Start time in milliseconds from the start of the audio: the encoder frame where the token
  /// was emitted times the frame length (40 ms for GigaAM, 80 ms for Parakeet); for Whisper the
  /// start of its segment; 0 for Moonshine, which has no timestamps.
  pub start_ms: u32,
}

/// The longest piece [`Model::transcribe_long`] cuts by default for a model without a window, in
/// ms: Parakeet, whose time per second of audio grows past a minute (its attention is quadratic).
const WINDOW_MS: u32 = 60_000;

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
    let languages = g.optional_array::<&str>("general.languages")?;
    let languages = languages.into_iter().map(str::to_string).collect();
    let family = match g.get::<&str>("general.architecture")? {
      "gigaam" => Family::Gigaam(Box::new(gigaam::Gigaam::load(&g)?)),
      "parakeet" => Family::Parakeet(Box::new(parakeet::Parakeet::load(&g)?)),
      "whisper" => Family::Whisper(Box::new(whisper::Whisper::load(&g)?)),
      "moonshine" => Family::Moonshine(Box::new(moonshine::Moonshine::load(&g)?)),
      arch => bail!(
        "general.architecture is {arch:?}, only gigaam, parakeet, whisper and moonshine are supported"
      ),
    };
    Ok(Self { family, languages })
  }

  /// Transcribes 16 kHz mono PCM samples in [-1, 1], passed to the model as they are, in one pass.
  ///
  /// For audio longer than [`max_audio_ms`](Self::max_audio_ms), use
  /// [`transcribe_long`](Self::transcribe_long) (see [Long audio](crate#long-audio)): GigaAM and
  /// Moonshine run it in one pass, which loses text past the window and fails past 1.5 times it.
  /// Whisper and Parakeet take any length. Audio too short for one encoder frame gives an empty
  /// transcript.
  ///
  /// # Errors
  ///
  /// - [`Error::AudioTooLong`]: GigaAM or Moonshine, and the audio is longer than 1.5 times
  ///   [`max_audio_ms`](Self::max_audio_ms) (37.5 s for GigaAM, 45 s for Moonshine).
  /// - The GPU failed: [`Error::Gpu`], [`Error::Poll`], [`Error::Readback`] or [`Error::Map`].
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
    self.transcribe_with(pcm, &Options::default())
  }

  /// [`transcribe`](Self::transcribe) with [`Options`]; [`Options::window_ms`] doesn't apply.
  ///
  /// # Errors
  ///
  /// As [`transcribe`](Self::transcribe), and [`Error::Language`] when [`Options::language`] is
  /// set and the model can't take it.
  ///
  /// # Examples
  ///
  /// ```no_run
  /// let model = wavo::Model::load("whisper-large-v3-turbo-Q8_0.gguf")?;
  /// # let pcm = vec![0.0f32; 16_000];
  /// let mut options = wavo::Options::default();
  /// options.language = Some("he".into()); // Hebrew, not detected
  /// let transcript = model.transcribe_with(&pcm, &options)?;
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn transcribe_with(&self, pcm: &[f32], options: &Options) -> Result<Transcript> {
    let tokens = matches!(&self.family, Family::Whisper(m) if m.multilingual());
    let language = match &options.language {
      Some(code) => language(code, &self.languages, tokens)?,
      None => None,
    };

    // One pass over the whole audio: a little past the window still matches the reference (the
    // 34-35 s fixtures), from ~45 s the tail is dropped or made up, and much longer audio
    // overruns the GPU's 65 535 workgroups per dispatch.
    if let (Family::Gigaam(_) | Family::Moonshine(_), Some(window)) =
      (&self.family, self.max_audio_ms())
      && pcm.len() > window as usize * 3 / 2 * 16
    {
      let ms = (pcm.len() as u64).div_ceil(16);
      return Err(Error::AudioTooLong { ms, max_ms: window * 3 / 2 });
    }

    match &self.family {
      Family::Gigaam(m) => m.transcribe(pcm),
      Family::Parakeet(m) => m.transcribe(pcm),
      Family::Whisper(m) => m.transcribe(pcm, language),
      Family::Moonshine(m) => m.transcribe(pcm),
    }
  }

  /// The longest audio, in milliseconds, the model was trained on: 25 000 for GigaAM, 30 000 for
  /// Whisper and Moonshine. [`transcribe_long`](Self::transcribe_long) splits long audio into
  /// pieces of up to this length: Whisper runs longer audio in windows of it, GigaAM and Moonshine
  /// lose text past it and [`transcribe`](Self::transcribe) rejects 1.5 times it. `None` for
  /// Parakeet, which has no such window.
  pub fn max_audio_ms(&self) -> Option<u32> {
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

  /// Transcribes 16 kHz mono PCM samples in [-1, 1] of any length: cut at pauses by [`split`]
  /// into pieces of up to [`Options::window_ms`] (by default the model's
  /// [`max_audio_ms`](Self::max_audio_ms), or 60 s for Parakeet), each transcribed by
  /// [`transcribe_with`](Self::transcribe_with) on its own. The texts are joined with a space
  /// (an empty one is skipped), and the tokens follow each other with `start_ms` from the start of
  /// `pcm`. Audio that fits in one piece gives the same transcript as `transcribe_with`.
  ///
  /// Each piece's first token starts a word, though with a character vocabulary it has no `▁` or
  /// space: GigaAM `ctc` and `rnnt` spell words letter by letter with a space token between them,
  /// and none falls between two pieces.
  ///
  /// The pieces run one after another on the calling thread; to report progress or cancel
  /// between them, cut with [`split`] and transcribe the pieces yourself.
  ///
  /// # Errors
  ///
  /// As [`transcribe_with`](Self::transcribe_with), from the first piece that fails: also
  /// [`Error::AudioTooLong`] when [`Options::window_ms`] is past 1.5 times the window of GigaAM or
  /// Moonshine.
  ///
  /// # Examples
  ///
  /// ```no_run
  /// let model = wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf")?;
  /// # let pcm = vec![0.0f32; 16_000 * 3600];
  /// let transcript = model.transcribe_long(&pcm, &wavo::Options::default())?; // an hour
  ///
  /// let mut options = wavo::Options::default();
  /// options.window_ms = Some(15_000); // shorter pieces
  /// let transcript = model.transcribe_long(&pcm, &options)?;
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn transcribe_long(&self, pcm: &[f32], options: &Options) -> Result<Transcript> {
    let window = options.window_ms.or(self.max_audio_ms()).unwrap_or(WINDOW_MS);
    let pieces = split(pcm, window).into_iter();
    let parts = pieces.map(|r| Ok((r.start, self.transcribe_with(&pcm[r], options)?)));
    Ok(split::join(parts.collect::<Result<Vec<_>>>()?))
  }

  /// The GPU the model runs on, and the API it reaches it through, as chosen at load.
  ///
  /// [`Model::load`] looks for an adapter through Vulkan and Metal first (Metal on macOS, Vulkan
  /// on Linux and Windows), and through DX12 only when neither has one (Windows without a Vulkan
  /// driver). It prefers the high-performance adapter, usually the discrete one on a laptop with
  /// two GPUs, though the driver decides. wgpu's `WGPU_BACKEND` environment variable (`vulkan`,
  /// `metal` or `dx12`) overrides the API. The adapter may be a
  /// [`SoftwareGpu`](DeviceKind::SoftwareGpu), such as WARP or llvmpipe, when no hardware driver
  /// is installed.
  ///
  /// One model runs on one device for its lifetime. To use several GPUs, load several models;
  /// choosing the device is not supported yet.
  ///
  /// # Examples
  ///
  /// ```no_run
  /// let model = wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf")?;
  /// println!("running on {}", model.device()); // running on Apple M2 (Metal, integrated)
  /// if model.device().kind == wavo::DeviceKind::SoftwareGpu {
  ///   eprintln!("no GPU driver found: transcription will be slow");
  /// }
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn device(&self) -> &Device {
    match &self.family {
      Family::Gigaam(m) => m.gpu.info(),
      Family::Parakeet(m) => m.gpu.info(),
      Family::Whisper(m) => m.gpu.info(),
      Family::Moonshine(m) => m.gpu.info(),
    }
  }
}

/// The index in `languages` of `code`, the language to force on a model with a token per
/// language (`tokens`: multilingual Whisper); `None` for a model of that language alone.
fn language(code: &str, languages: &[String], tokens: bool) -> Result<Option<usize>> {
  let fail = |reason: String| Err(Error::Language(format!("language {code:?}: {reason}")));
  let Some(i) = languages.iter().position(|l| l == code) else {
    return fail(match languages.len() {
      0 => "the model lists no languages".into(),
      _ => format!("not one of the model's languages: {}", languages.join(", ")),
    });
  };
  match (tokens, languages.len()) {
    (true, _) => Ok(Some(i)),
    (false, 1) => Ok(None),
    (false, n) => fail(format!(
      "the model detects its {n} languages itself and can't be told one; multilingual Whisper \
       models can"
    )),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn language_codes() {
    let list = |codes: &[&str]| codes.iter().map(|c| c.to_string()).collect::<Vec<_>>();
    let whisper = list(&["en", "zh", "de", "he"]);
    assert_eq!(language("he", &whisper, true).unwrap(), Some(3));
    assert_eq!(language("en", &whisper, true).unwrap(), Some(0));
    let err = language("xx", &whisper, true).unwrap_err().to_string();
    assert_eq!(err, r#"language "xx": not one of the model's languages: en, zh, de, he"#);
    // Codes are case-sensitive, as in the GGUF.
    assert!(language("HE", &whisper, true).is_err());
    // A model of one language takes its own code and nothing else.
    assert_eq!(language("ru", &list(&["ru"]), false).unwrap(), None);
    assert!(language("en", &list(&["ru"]), false).is_err());
    // Parakeet v3: no language token, so not even a listed code.
    let err = language("de", &list(&["en", "de", "uk"]), false).unwrap_err().to_string();
    assert!(err.contains("detects its 3 languages itself"), "{err}");
    assert!(matches!(language("de", &[], false), Err(Error::Language(_))));
  }

  #[test]
  fn model_and_device_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<super::Model>();
    check::<super::Device>();
  }
}
