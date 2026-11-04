// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo

/// Everything that can fail in wavo: loading a model, running it on the GPU and, with the `hub`
/// feature, managing downloads.
///
/// It is `#[non_exhaustive]` because the `hub` feature adds variants, and features unify across a
/// build: keep a `_` arm when matching on it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// Reading the model file, or the cache's files and folders with the `hub` feature, failed.
  #[error(transparent)]
  Io(#[from] std::io::Error),
  /// The file is not a model wavo supports: not a GGUF, an architecture other than `gigaam`,
  /// `parakeet`, `whisper` and `moonshine`, or a missing or mismatched tensor or metadata key.
  #[error("invalid model: {0}")]
  Model(String),
  /// No GPU adapter (Metal, Vulkan, or DX12 on Windows) is available.
  #[error("no GPU adapter: {0}")]
  Adapter(#[from] wgpu::RequestAdapterError),
  /// The GPU adapter refused a device with the limits the model needs.
  #[error("GPU device: {0}")]
  Device(#[from] wgpu::RequestDeviceError),
  /// Reading results back from the GPU failed.
  #[error("GPU readback: {0}")]
  Readback(#[from] wgpu::BufferAsyncError),
  /// Mapping a GPU buffer for reading failed.
  #[error("GPU readback: {0}")]
  Map(#[from] wgpu::MapRangeError),
  /// Waiting for the GPU failed, for example after a device loss.
  #[error("GPU poll: {0}")]
  Poll(#[from] wgpu::PollError),
  /// An error the GPU reported during a call (out of memory, a lost device), or work the driver
  /// dropped unfinished: some drivers cancel GPU work that runs for more than about 20 s.
  #[error("GPU: {0}")]
  Gpu(String),
  /// The audio is longer than 1.5 times [`Model::max_audio_ms`](crate::Model::max_audio_ms) for
  /// a model that runs it in one pass (GigaAM, Moonshine): split it first, or call
  /// [`Model::transcribe_long`](crate::Model::transcribe_long).
  #[error("audio is {ms} ms, longer than the model's {max_ms} ms: split it at pauses")]
  AudioTooLong {
    /// The audio's length, in milliseconds, rounded up.
    ms: u64,
    /// The longest audio the model takes, in milliseconds: 1.5 times
    /// [`max_audio_ms`](crate::Model::max_audio_ms).
    max_ms: u32,
  },
  /// [`Options::language`](crate::Options::language) names a language the model can't take: one
  /// it doesn't list, or any for a model that detects among several languages itself (Parakeet
  /// v3). The message lists the model's codes when the code isn't one of them.
  /// [`Model::languages`](crate::Model::languages) and
  /// [`Model::takes_language`](crate::Model::takes_language) tell beforehand.
  #[error("{0}")]
  Language(String),
  /// The Neural Engine can't run the model: this machine has none (it needs Apple Silicon and
  /// macOS 15 or later), the family has no Neural Engine program, the model isn't
  /// [prepared](crate::Model::prepare), or Core ML failed to compile, load or run it.
  #[error("Neural Engine: {0}")]
  NeuralEngine(String),
  /// A progress callback returned [`ControlFlow::Break`](std::ops::ControlFlow::Break): the one
  /// of [`Model::prepare`](crate::Model::prepare), of
  /// [`Model::transcribe_long`](crate::Model::transcribe_long), or with the `hub` feature of
  /// `Hub::pull`, whose partial download stays in the cache for the next pull of the model to
  /// resume.
  #[error("cancelled")]
  Cancelled,
  /// A name that is neither a short nor a full name in [`hub::MODELS`](crate::hub::MODELS).
  #[cfg(feature = "hub")]
  #[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
  #[error("unknown model {name:?}{} Known models: {}", crate::hub::hint(*suggestion), crate::hub::names())]
  UnknownModel {
    /// The name as given.
    name: String,
    /// The closest known name, when one is a likely typo of `name`.
    suggestion: Option<&'static str>,
  },
  /// A request to the Hugging Face Hub failed, including a download that broke off midway.
  /// [`Hub::pull`](crate::hub::Hub::pull) has already retried what may pass on another try.
  #[cfg(feature = "hub")]
  #[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
  #[error("{url}: {message}")]
  Http {
    /// The URL requested.
    url: String,
    /// The HTTP status of an error response (404: no such file); `None` when no response came
    /// or the body broke off.
    status: Option<u16>,
    /// What went wrong.
    message: String,
  },
  /// The Hub's answer is not as expected: a response without the commit, ETag or size headers,
  /// or a download of the wrong size.
  #[cfg(feature = "hub")]
  #[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
  #[error("{0}")]
  Hub(String),
}

impl Error {
  /// Whether the GPU or Core ML failed rather than the request: [`Error::Gpu`], [`Error::Poll`],
  /// [`Error::Readback`], [`Error::Map`], [`Error::Device`] or [`Error::NeuralEngine`]. After a
  /// call on a [`Model`](crate::Model) fails so, the model may stay unusable (a lost device, out
  /// of memory, work the driver dropped): drop it and load it again, which may land on another
  /// backend, as [`Backend::Auto`](crate::Backend::Auto) takes the GPU when Core ML fails to load.
  /// False for the rest, after which the model takes the next call as before: the audio or the
  /// options don't fit ([`Error::AudioTooLong`], [`Error::Language`]) or a callback cancelled
  /// ([`Error::Cancelled`]). The errors of loading a file and of the `hub` feature are false too,
  /// except [`Error::Device`] and [`Error::NeuralEngine`], which a load also returns.
  ///
  /// # Examples
  ///
  /// ```no_run
  /// # let pcm = vec![0.0f32; 16_000];
  /// let path = "gigaam-v3-e2e-rnnt-Q8_0.gguf";
  /// let mut model = wavo::Model::load(path)?;
  /// let transcript = match model.transcribe(&pcm) {
  ///   Err(e) if e.is_fatal() => {
  ///     model = wavo::Model::load(path)?;
  ///     model.transcribe(&pcm)?
  ///   }
  ///   result => result?,
  /// };
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn is_fatal(&self) -> bool {
    matches!(
      self,
      Self::Gpu(_)
        | Self::Poll(_)
        | Self::Readback(_)
        | Self::Map(_)
        | Self::Device(_)
        | Self::NeuralEngine(_)
    )
  }
}

/// `Result` with wavo's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Shorthand for `Error::Model` with a formatted message.
macro_rules! bail {
  ($($arg:tt)*) => {
    return Err($crate::error::Error::Model(format!($($arg)*)))
  };
}
pub(crate) use bail;
