// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Download and manage the supported models in the Hugging Face cache (the `hub` feature).
//!
//! [`MODELS`] lists the models wavo knows by name: Q8_0 GGUF files in the Hugging Face repositories
//! `<owner>/<full name>-gguf`, transcribe.cpp's own under `handy-computer` and a few more
//! conversions under `vladkens`. A [`Hub`] finds, downloads and removes them in a cache folder laid
//! out as the
//! [Hub local cache spec](https://huggingface.co/docs/hub/local-cache) says, so `wavo`, the `hf`
//! CLI, Python's `huggingface_hub` and your app all find what any of them downloaded.
//!
//! Only [`Hub::pull`] uses the network. It blocks until the file is complete, so run it on a thread
//! of your own (the library starts none): [`std::thread::spawn`], or tokio's `spawn_blocking`.
//!
//! ```no_run
//! use std::ops::ControlFlow;
//! use wavo::hub::{self, Hub};
//!
//! # fn main() -> wavo::Result<()> {
//! for m in hub::MODELS {
//!   println!("{:<18} {:>5} MB  {}", m.name, m.size / 1_000_000, m.languages);
//! }
//! let hub = Hub::new();
//! let path = match hub.find("gigaam-v3") {
//!   Some(path) => path,
//!   None => hub.pull("gigaam-v3", |_| ControlFlow::Continue(()))?,
//! };
//! let model = wavo::Model::load(path)?;
//! # Ok(())
//! # }
//! ```

mod cache;

use std::fmt;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use crate::{Error, PrepareStep, Result};

/// How many more times [`Hub::pull`] resumes a download that failed on the network, after 1, 2
/// and 4 s.
pub const RETRIES: u32 = 3;

/// A model wavo can download by name: one entry of [`MODELS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelInfo {
  /// Short name, such as `parakeet-v3`.
  pub name: &'static str,
  /// The Hugging Face account that publishes its GGUF: `handy-computer` (transcribe.cpp) for most
  /// models, `vladkens` for conversions transcribe.cpp doesn't publish.
  pub owner: &'static str,
  /// The name it is published under, such as `parakeet-tdt-0.6b-v3`; the same as `name` for most
  /// models. Every function that takes a name accepts this one too.
  pub full_name: &'static str,
  /// Size of the Q8_0 GGUF file in bytes.
  pub size: u64,
  /// The languages it transcribes, for people: `English`, `Russian`, `25 European languages`, …
  pub languages: &'static str,
  /// The languages it is meant for, for programs: `["he"]` for the Hebrew fine-tunes, `["zh",
  /// "en"]` for Breeze-ASR-25, every language for the multilingual models. The codes are short
  /// BCP-47, ISO 639-1 where one exists (`en`, `ru`), else ISO 639-3 (`haw`, `yue`); Whisper's
  /// keep its own `jw` for Javanese (ISO `jv`).
  ///
  /// A fine-tune takes more codes in [`Options::language`](crate::Options::language) than it is
  /// meant for: those of its base model, which its GGUF lists in `general.languages`.
  /// `whisper-turbo-he` takes Whisper's 100. Check a code against that list, not this one.
  ///
  /// # Examples
  ///
  /// The models that transcribe Russian, and those made for Hebrew alone:
  ///
  /// ```
  /// use wavo::hub::{MODELS, ModelInfo};
  ///
  /// let names = |f: fn(&&ModelInfo) -> bool| MODELS.iter().filter(f).map(|m| m.name).collect();
  /// let ru: Vec<&str> = names(|m| m.langs.contains(&"ru"));
  /// assert!(ru.contains(&"gigaam-v3") && ru.contains(&"parakeet-v3"));
  /// assert!(!ru.contains(&"parakeet-v2"));
  ///
  /// let he: Vec<&str> = names(|m| m.langs == ["he"]);
  /// assert_eq!(he, ["parakeet-he", "whisper-turbo-he"]);
  /// ```
  pub langs: &'static [&'static str],
  /// Whether the text comes cased and punctuated; `false` is lowercase without punctuation.
  pub punctuated: bool,
  /// The model family, as the crate docs name it: `Parakeet`, `GigaAM`, `Whisper` or `Moonshine`.
  pub family: &'static str,
  /// Who made the model, such as `NVIDIA` or `ivrit.ai`; for a fine-tune, who tuned it.
  pub maker: &'static str,
  /// The URL of the original model's card, such as
  /// `https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3`: its license, languages and benchmarks.
  pub card: &'static str,
}

impl ModelInfo {
  /// The Hugging Face repository `<owner>/<full name>-gguf`, such as
  /// `handy-computer/parakeet-tdt-0.6b-v3-gguf`.
  pub fn repo(&self) -> String {
    format!("{}/{}-gguf", self.owner, self.full_name)
  }

  /// The file name in [`repo`](Self::repo), such as `parakeet-tdt-0.6b-v3-Q8_0.gguf`.
  pub fn file(&self) -> String {
    format!("{}-Q8_0.gguf", self.full_name)
  }
}

/// transcribe.cpp's account, which publishes most models.
const HC: &str = "handy-computer";
/// wavo's author's account, for conversions transcribe.cpp doesn't publish.
const VK: &str = "vladkens";

static EN: &[&str] = &["en"];
static HE: &[&str] = &["he"];
static RU: &[&str] = &["ru"];
static ZH_EN: &[&str] = &["zh", "en"];
/// transcribe.cpp's `V3_LANGUAGES`, in the model card's order.
static PARAKEET_V3: &[&str] = &[
  "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
  "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "uk",
];
/// Whisper's language tokens sorted as strings, hence `haw` (`<|haw|>`) before `ha`.
static WHISPER: &[&str] = &[
  "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da",
  "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "haw", "ha", "he", "hi",
  "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
  "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
  "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
  "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
  "yi", "yo", "zh",
];
/// [`WHISPER`] and Cantonese, which large-v3 added.
static WHISPER_V3: &[&str] = &[
  "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da",
  "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "haw", "ha", "he", "hi",
  "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
  "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
  "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
  "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
  "yi", "yo", "yue", "zh",
];

/// Every model wavo can download by name, as in the crate's [model table](crate#models).
#[rustfmt::skip]
pub static MODELS: &[ModelInfo] = &[
  ModelInfo { name: "parakeet-v3", owner: HC, full_name: "parakeet-tdt-0.6b-v3", size: 739_508_576, languages: "25 European languages", langs: PARAKEET_V3, punctuated: true, family: "Parakeet", maker: "NVIDIA", card: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3" },
  ModelInfo { name: "parakeet-v2", owner: HC, full_name: "parakeet-tdt-0.6b-v2", size: 729_574_912, languages: "English", langs: EN, punctuated: true, family: "Parakeet", maker: "NVIDIA", card: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2" },
  ModelInfo { name: "parakeet-ultra", owner: HC, full_name: "parakeet-ultra", size: 740_363_168, languages: "25 European languages", langs: PARAKEET_V3, punctuated: true, family: "Parakeet", maker: "Moondream", card: "https://huggingface.co/moondream/parakeet-ultra" },
  ModelInfo { name: "parakeet-he", owner: VK, full_name: "parakeet-tdt-0.6b-v3-he", size: 740_915_360, languages: "Hebrew", langs: HE, punctuated: true, family: "Parakeet", maker: "Yakov Kolani", card: "https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he" },
  ModelInfo { name: "gigaam-v3", owner: HC, full_name: "gigaam-v3-e2e-rnnt", size: 273_724_832, languages: "Russian", langs: RU, punctuated: true, family: "GigaAM", maker: "Sber", card: "https://huggingface.co/ai-sage/GigaAM-v3" },
  ModelInfo { name: "gigaam-v3-e2e-ctc", owner: HC, full_name: "gigaam-v3-e2e-ctc", size: 272_151_136, languages: "Russian", langs: RU, punctuated: true, family: "GigaAM", maker: "Sber", card: "https://huggingface.co/ai-sage/GigaAM-v3" },
  ModelInfo { name: "gigaam-v3-rnnt", owner: HC, full_name: "gigaam-v3-rnnt", size: 273_022_880, languages: "Russian", langs: RU, punctuated: false, family: "GigaAM", maker: "Sber", card: "https://huggingface.co/ai-sage/GigaAM-v3" },
  ModelInfo { name: "gigaam-v3-ctc", owner: HC, full_name: "gigaam-v3-ctc", size: 271_803_328, languages: "Russian", langs: RU, punctuated: false, family: "GigaAM", maker: "Sber", card: "https://huggingface.co/ai-sage/GigaAM-v3" },
  ModelInfo { name: "whisper-turbo", owner: HC, full_name: "whisper-large-v3-turbo", size: 886_381_760, languages: "100 languages", langs: WHISPER_V3, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-large-v3-turbo" },
  ModelInfo { name: "whisper-turbo-he", owner: VK, full_name: "ivrit-whisper-large-v3-turbo", size: 886_381_920, languages: "Hebrew", langs: HE, punctuated: true, family: "Whisper", maker: "ivrit.ai", card: "https://huggingface.co/ivrit-ai/whisper-large-v3-turbo" },
  ModelInfo { name: "whisper-tiny", owner: HC, full_name: "whisper-tiny", size: 45_981_088, languages: "99 languages", langs: WHISPER, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-tiny" },
  ModelInfo { name: "whisper-base", owner: HC, full_name: "whisper-base", size: 84_962_880, languages: "99 languages", langs: WHISPER, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-base" },
  ModelInfo { name: "whisper-small", owner: HC, full_name: "whisper-small", size: 269_751_136, languages: "99 languages", langs: WHISPER, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-small" },
  ModelInfo { name: "whisper-medium", owner: HC, full_name: "whisper-medium", size: 831_538_144, languages: "99 languages", langs: WHISPER, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-medium" },
  ModelInfo { name: "whisper-large-v3", owner: HC, full_name: "whisper-large-v3", size: 1_668_741_440, languages: "100 languages", langs: WHISPER_V3, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-large-v3" },
  ModelInfo { name: "whisper-tiny.en", owner: HC, full_name: "whisper-tiny.en", size: 45_904_544, languages: "English", langs: EN, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-tiny.en" },
  ModelInfo { name: "whisper-base.en", owner: HC, full_name: "whisper-base.en", size: 84_886_208, languages: "English", langs: EN, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-base.en" },
  ModelInfo { name: "whisper-small.en", owner: HC, full_name: "whisper-small.en", size: 269_674_144, languages: "English", langs: EN, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-small.en" },
  ModelInfo { name: "whisper-medium.en", owner: HC, full_name: "whisper-medium.en", size: 831_460_928, languages: "English", langs: EN, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-medium.en" },
  ModelInfo { name: "whisper-large", owner: HC, full_name: "whisper-large", size: 1_667_964_192, languages: "99 languages", langs: WHISPER, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-large" },
  ModelInfo { name: "whisper-large-v2", owner: HC, full_name: "whisper-large-v2", size: 1_667_964_224, languages: "99 languages", langs: WHISPER, punctuated: true, family: "Whisper", maker: "OpenAI", card: "https://huggingface.co/openai/whisper-large-v2" },
  ModelInfo { name: "breeze-asr-25", owner: HC, full_name: "Breeze-ASR-25", size: 1_667_964_224, languages: "Taiwanese Mandarin and English", langs: ZH_EN, punctuated: true, family: "Whisper", maker: "MediaTek Research", card: "https://huggingface.co/MediaTek-Research/Breeze-ASR-25" },
  ModelInfo { name: "moonshine-tiny", owner: HC, full_name: "moonshine-tiny", size: 35_466_912, languages: "English", langs: EN, punctuated: true, family: "Moonshine", maker: "Useful Sensors", card: "https://huggingface.co/UsefulSensors/moonshine-tiny" },
  ModelInfo { name: "moonshine-base", owner: HC, full_name: "moonshine-base", size: 77_476_480, languages: "English", langs: EN, punctuated: true, family: "Moonshine", maker: "Useful Sensors", card: "https://huggingface.co/UsefulSensors/moonshine-base" },
];

/// The model a short or full name stands for.
///
/// # Errors
///
/// [`Error::UnknownModel`] for any other name, with the closest known name when it looks like a
/// typo.
///
/// # Examples
///
/// ```
/// let m = wavo::hub::lookup("parakeet-tdt-0.6b-v3")?;
/// assert_eq!(m.name, "parakeet-v3");
/// assert_eq!(m.repo(), "handy-computer/parakeet-tdt-0.6b-v3-gguf");
///
/// let e = wavo::hub::lookup("parakeet3").unwrap_err();
/// assert!(matches!(e, wavo::Error::UnknownModel { suggestion: Some("parakeet-v3"), .. }));
/// # Ok::<(), wavo::Error>(())
/// ```
pub fn lookup(name: &str) -> Result<&'static ModelInfo> {
  if let Some(m) = MODELS.iter().find(|m| m.name == name || m.full_name == name) {
    return Ok(m);
  }
  let closest = (MODELS.iter().flat_map(|m| [m.name, m.full_name]))
    .map(|n| (distance(name, n), n))
    .min()
    .filter(|&(d, n)| d <= (n.len() / 3).max(2));
  Err(Error::UnknownModel { name: name.into(), suggestion: closest.map(|(_, n)| n) })
}

/// The end of [`Error::UnknownModel`]'s first sentence.
pub(crate) fn hint(suggestion: Option<&str>) -> String {
  suggestion.map_or(".".into(), |n| format!(", did you mean {n}?"))
}

pub(crate) fn names() -> String {
  MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
}

/// Levenshtein distance.
fn distance(a: &str, b: &str) -> usize {
  let b: Vec<char> = b.chars().collect();
  let mut row: Vec<usize> = (0..=b.len()).collect();
  for (i, ca) in a.chars().enumerate() {
    let mut diag = row[0];
    row[0] = i + 1;
    for (j, &cb) in b.iter().enumerate() {
      let next = (diag + (ca != cb) as usize).min(row[j] + 1).min(row[j + 1] + 1);
      diag = row[j + 1];
      row[j + 1] = next;
    }
  }
  row[b.len()]
}

/// How far [`Hub::pull`] got, passed to its callback.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Progress<'a> {
  /// Bytes of the file on disk, including a part an earlier pull left to resume.
  pub done: u64,
  /// Size of the file in bytes.
  pub total: u64,
  /// The error a try failed with, on the one call before [`Hub::pull`] waits and retries. With
  /// [`preparing`](Self::preparing), the error preparing failed with, on one last call: the file
  /// stays and runs on the GPU, and the next pull prepares again. `None` on every other call.
  pub retrying: Option<&'a Error>,
  /// The step of [preparing](crate::Model::prepare) the downloaded model for the Neural Engine,
  /// once per step after the download (`done` equals `total` then), and the failed one again when
  /// preparing fails; `None` while downloading.
  pub preparing: Option<PrepareStep>,
}

/// A Hugging Face cache folder that holds models.
///
/// It shares the cache with `hf` and Python's `huggingface_hub`: a model either of them downloaded
/// is found here, and the other way round. A model's file is at
/// `<root>/models--<owner>--<full name>-gguf/snapshots/<commit>/<full name>-Q8_0.gguf`, a
/// symlink into the repository's `blobs/`. Several processes can pull and remove at once: they
/// take turns through lock files in `<root>/.locks/`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Hub {
  root: PathBuf,
  /// The environment variable that set `root`, if any.
  from: Option<&'static str>,
  /// Whether `pull` prepares models for the Neural Engine.
  prepare: bool,
}

impl Default for Hub {
  fn default() -> Self {
    Self::new()
  }
}

impl Hub {
  /// The cache that `hf` and Python's `huggingface_hub` use: `$HF_HUB_CACHE`, else the legacy
  /// `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
  /// `~/.cache/huggingface/hub` (the home folder from `$HOME`, else `$USERPROFILE`). Empty
  /// variables count as unset.
  pub fn new() -> Self {
    Self::from_env(|k| std::env::var_os(k).filter(|v| !v.is_empty()))
  }

  /// A cache in a folder of your own, such as your app's data folder. It is created on the first
  /// pull.
  pub fn at(root: impl Into<PathBuf>) -> Self {
    Self { root: root.into(), from: None, prepare: true }
  }

  /// Whether [`pull`](Self::pull) also [prepares](crate::Model::prepare) Parakeet and GigaAM for
  /// the Neural Engine after the download (the default) or only downloads. Preparing compiles in
  /// the calling process, with a peak of about the model's size and Core ML state that stays,
  /// and warms the Neural Engine for this program only: an app that loads models in another
  /// process (a helper, the same executable started again) downloads with `prepare(false)` and
  /// calls `Model::prepare` there.
  ///
  /// ```no_run
  /// use std::ops::ControlFlow;
  ///
  /// let hub = wavo::hub::Hub::new().prepare(false);
  /// let path = hub.pull("parakeet-v3", |_| ControlFlow::Continue(()))?;
  /// // Later, in the process that transcribes:
  /// wavo::Model::prepare(&path, |_| ControlFlow::Continue(()))?;
  /// # Ok::<(), wavo::Error>(())
  /// ```
  pub fn prepare(mut self, prepare: bool) -> Self {
    self.prepare = prepare;
    self
  }

  /// The cache folder.
  pub fn root(&self) -> &Path {
    &self.root
  }

  /// The downloaded file of a model by short or full name. Never uses the network.
  ///
  /// `None` when the model isn't downloaded, or `name` isn't in [`MODELS`].
  pub fn find(&self, name: &str) -> Option<PathBuf> {
    let m = lookup(name).ok()?;
    self.find_file(&m.repo(), &m.file())
  }

  /// The model file a name or path stands for, such as a command line's MODEL argument. Never
  /// uses the network.
  ///
  /// An argument that ends in `.gguf` or has a path separator (`/` or `\`, on every system) is
  /// a path, returned as is when the file exists. Any other argument is a model's short or full
  /// name, never a file in the current folder: `parakeet-v3` is the downloaded file of that model
  /// even beside a file named `parakeet-v3`; pass `./parakeet-v3` for the file.
  ///
  /// # Errors
  ///
  /// - [`Error::Io`] of kind [`NotFound`](std::io::ErrorKind::NotFound): a path that is not a file
  ///   (`<path>: no such file`), or a model that isn't downloaded (`<name> is not in <root>`);
  ///   [`pull`](Self::pull) it then.
  /// - [`Error::UnknownModel`]: a name that is not in [`MODELS`], with the closest name when it
  ///   looks like a typo.
  ///
  /// # Examples
  ///
  /// ```
  /// use std::io::ErrorKind;
  /// use wavo::{Error, hub::Hub};
  ///
  /// let hub = Hub::at("/no/such/cache");
  /// let missing = hub.resolve("gigaam-v3").unwrap_err();
  /// assert!(matches!(&missing, Error::Io(e) if e.kind() == ErrorKind::NotFound));
  /// assert!(matches!(hub.resolve("gigam-v3"), Err(Error::UnknownModel { .. })));
  /// assert!(matches!(hub.resolve("models/my.gguf"), Err(Error::Io(_))));
  /// ```
  pub fn resolve(&self, name_or_path: &str) -> Result<PathBuf> {
    let not_found =
      |what: String| Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, what));
    if name_or_path.ends_with(".gguf") || name_or_path.contains(['/', '\\']) {
      let path = Path::new(name_or_path);
      if path.is_file() {
        return Ok(path.into());
      }

      return Err(not_found(format!("{name_or_path}: no such file")));
    }

    let m = lookup(name_or_path)?;
    self.find(m.name).ok_or_else(|| not_found(format!("{} is not in {self}", m.name)))
  }

  /// The models of [`MODELS`] downloaded to this cache, in its order, with their files. Never uses
  /// the network.
  ///
  /// ```
  /// assert!(wavo::hub::Hub::at("/no/such/cache").downloaded().is_empty());
  /// ```
  pub fn downloaded(&self) -> Vec<(&'static ModelInfo, PathBuf)> {
    MODELS.iter().filter_map(|m| self.find(m.name).map(|path| (m, path))).collect()
  }

  /// Downloads a model by short or full name, unless its latest revision is in the cache already,
  /// and returns the path of its file.
  ///
  /// It blocks until the file is complete: run it on a thread of your own. One HEAD request asks
  /// the Hub for the latest revision; when the cache has it, nothing else is downloaded.
  /// Otherwise the file downloads into `blobs/<etag>.incomplete`, resuming a part an earlier pull
  /// left there, and moves into place once complete. A download that breaks off, or receives
  /// nothing for 30 s, is resumed up to [`RETRIES`] more times. There is no limit on the time a
  /// whole download takes.
  ///
  /// Then, for Parakeet and GigaAM on a Mac with a Neural Engine, it
  /// [prepares](crate::Model::prepare) the file for it, which takes the longest the first time.
  /// Elsewhere, for the other families and with [`prepare(false)`](Self::prepare), this step is
  /// skipped. When preparing fails, `pull` still returns the path: the file runs on the GPU,
  /// `progress` gets the error ([`Progress::retrying`]), and the next pull prepares again.
  ///
  /// `progress` is called when the download starts, after every chunk it receives (often: throttle
  /// your own UI updates), before every retry, and as each step of preparing starts
  /// ([`Progress::preparing`]). Return [`ControlFlow::Break`] to cancel: a partial file stays for
  /// the next pull to resume. While preparing, `Break` cancels before the next step starts; a
  /// step that has started (the system's compile in `Load` takes the longest) runs to its end.
  ///
  /// # Errors
  ///
  /// - [`Error::UnknownModel`]: `name` is not in [`MODELS`].
  /// - [`Error::Http`]: the Hub could not be reached within 15 s or answered with an error status,
  ///   or the download broke off or stalled more times than it retries.
  /// - [`Error::Hub`]: the Hub's answer lacks the expected headers, or the file came out at the
  ///   wrong size.
  /// - [`Error::Cancelled`]: `progress` returned [`ControlFlow::Break`].
  /// - [`Error::Io`]: writing to the cache failed, for example on a full disk.
  ///
  /// # Examples
  ///
  /// Print the percentage, and give up after 1 GB:
  ///
  /// ```no_run
  /// use std::ops::ControlFlow;
  ///
  /// let path = wavo::hub::Hub::new().pull("whisper-large-v3", |p| {
  ///   eprint!("\r{:3}%", p.done * 100 / p.total);
  ///   if p.done < 1_000_000_000 { ControlFlow::Continue(()) } else { ControlFlow::Break(()) }
  /// })?;
  /// # Ok::<(), wavo::Error>(())
  /// ```
  ///
  /// On a thread, with a cancel button: see
  /// [the crate docs](crate#download-on-a-thread-with-progress-and-cancel).
  pub fn pull(
    &self,
    name: &str,
    mut progress: impl FnMut(Progress<'_>) -> ControlFlow<()>,
  ) -> Result<PathBuf> {
    let m = lookup(name)?;
    // The repository lock, held from the download through preparing, as `remove` takes it: the
    // file stays while it is read.
    let (path, _repo) = self.pull_file(&m.repo(), &m.file(), &mut progress)?;
    if !self.prepare {
      return Ok(path);
    }

    let (size, mut last) = (std::fs::metadata(&path)?.len(), PrepareStep::Build);
    let prepared = crate::Model::prepare(&path, |step| {
      last = step;
      progress(Progress { done: size, total: size, retrying: None, preparing: Some(step) })
    });
    match prepared {
      Err(Error::Cancelled) => Err(Error::Cancelled),
      Err(e) => {
        let _ =
          progress(Progress { done: size, total: size, retrying: Some(&e), preparing: Some(last) });
        Ok(path)
      }
      Ok(_) => Ok(path),
    }
  }

  /// Deletes a model by short or full name from the cache, along with what
  /// [`Model::prepare`](crate::Model::prepare) made for it, and returns the bytes freed; `None`
  /// when it wasn't downloaded. Other files in the model's repository folder stay.
  ///
  /// Newer `hf` versions keep a file's bytes in a store shared across repositories,
  /// `<root>/blobs/`. As with `hf cache rm`, such a file is deleted only once no other repository
  /// links to it, and counts as 0 bytes freed while one still does.
  ///
  /// It waits while another process pulls the same model.
  ///
  /// # Errors
  ///
  /// - [`Error::UnknownModel`]: `name` is not in [`MODELS`].
  /// - [`Error::Io`]: deleting failed.
  pub fn remove(&self, name: &str) -> Result<Option<u64>> {
    let m = lookup(name)?;
    self.remove_file(&m.repo(), &m.file())
  }
}

/// The root, with the environment variable that set it: `/data/hub (set by HF_HOME)`.
impl fmt::Display for Hub {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.root.display())?;
    self.from.map_or(Ok(()), |v| write!(f, " (set by {v})"))
  }
}

#[cfg(test)]
mod tests {
  use std::fs;

  use super::*;

  fn name(arg: &str) -> &'static str {
    lookup(arg).unwrap().name
  }

  fn error(arg: &str) -> String {
    lookup(arg).unwrap_err().to_string()
  }

  #[test]
  fn short_and_full_names() {
    assert_eq!(name("parakeet-v3"), "parakeet-v3");
    assert_eq!(name("parakeet-tdt-0.6b-v3"), "parakeet-v3");
    assert_eq!(name("gigaam-v3"), "gigaam-v3");
    assert_eq!(name("gigaam-v3-e2e-rnnt"), "gigaam-v3");
    assert_eq!(name("gigaam-v3-ctc"), "gigaam-v3-ctc");
    let m = lookup("gigaam-v3").unwrap();
    assert_eq!(m.repo(), "handy-computer/gigaam-v3-e2e-rnnt-gguf");
    assert_eq!(m.file(), "gigaam-v3-e2e-rnnt-Q8_0.gguf");
    assert_eq!(name("whisper-large-v3-turbo"), "whisper-turbo");
    let m = lookup("whisper-turbo").unwrap();
    assert_eq!(m.repo(), "handy-computer/whisper-large-v3-turbo-gguf");
    assert_eq!(m.file(), "whisper-large-v3-turbo-Q8_0.gguf");
    assert_eq!(name("Breeze-ASR-25"), "breeze-asr-25");
    let m = lookup("breeze-asr-25").unwrap();
    assert_eq!(m.repo(), "handy-computer/Breeze-ASR-25-gguf");
    assert_eq!(m.file(), "Breeze-ASR-25-Q8_0.gguf");
    assert_eq!(name("ivrit-whisper-large-v3-turbo"), "whisper-turbo-he");
    let m = lookup("whisper-turbo-he").unwrap();
    assert_eq!(m.repo(), "vladkens/ivrit-whisper-large-v3-turbo-gguf");
    assert_eq!(m.file(), "ivrit-whisper-large-v3-turbo-Q8_0.gguf");
    let m = lookup("parakeet-he").unwrap();
    assert_eq!(m.repo(), "vladkens/parakeet-tdt-0.6b-v3-he-gguf");
    assert_eq!(m.file(), "parakeet-tdt-0.6b-v3-he-Q8_0.gguf");
  }

  #[test]
  fn names_are_unambiguous() {
    for (i, a) in MODELS.iter().enumerate() {
      for b in &MODELS[i + 1..] {
        for n in [b.name, b.full_name] {
          assert!(a.name != n && a.full_name != n, "{} and {} share {n}", a.full_name, b.full_name);
        }
      }
    }
  }

  /// `tests/models.rs` checks the codes against the GGUFs; this, that they agree with `languages`.
  #[test]
  fn langs() {
    for m in MODELS {
      let codes = m.langs;
      assert!(!codes.is_empty(), "{}", m.name);
      for c in codes {
        let ok = (2..=3).contains(&c.len()) && c.bytes().all(|b| b.is_ascii_lowercase());
        assert!(ok, "{}: {c:?}", m.name);
      }
      let mut unique = codes.to_vec();
      unique.sort();
      unique.dedup();
      assert_eq!(unique.len(), codes.len(), "{}: a code twice", m.name);
      if let Ok(n) = m.languages.split(' ').next().unwrap().parse::<usize>() {
        assert_eq!(codes.len(), n, "{}: {}", m.name, m.languages);
      }
    }
  }

  #[test]
  fn catalog_fields() {
    for m in MODELS {
      assert!(["Parakeet", "GigaAM", "Whisper", "Moonshine"].contains(&m.family), "{}", m.name);
      assert!(!m.maker.is_empty() && !m.languages.is_empty(), "{}", m.name);
      let repo: Vec<&str> =
        m.card.strip_prefix("https://huggingface.co/").unwrap_or("").split('/').collect();
      assert!(repo.len() == 2 && !repo.contains(&""), "{}: {}", m.name, m.card);
    }
  }

  #[test]
  fn resolve() {
    let hub = Hub::at("/nowhere");
    let error = |arg| hub.resolve(arg).unwrap_err().to_string();
    assert_eq!(error("gigaam-v3-e2e-rnnt"), "gigaam-v3 is not in /nowhere");
    assert_eq!(error("dir/m.gguf"), "dir/m.gguf: no such file");
    assert_eq!(error("m.gguf"), "m.gguf: no such file");
    assert_eq!(error(r"C:\models\x.GGUF"), r"C:\models\x.GGUF: no such file");
    assert!(error("gigam-v3").contains("did you mean gigaam-v3?"));
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert_eq!(hub.resolve(file.to_str().unwrap()).unwrap(), file);
    // A bare name is a model's even when a file of that name is in the current folder.
    assert!(Path::new("Cargo.toml").is_file());
    assert!(matches!(hub.resolve("Cargo.toml"), Err(Error::UnknownModel { .. })));
    assert!(hub.downloaded().is_empty());
  }

  #[test]
  fn unknown_names() {
    assert!(error("parakeet3").contains("did you mean parakeet-v3?"));
    assert!(error("gigam-v3").contains("did you mean gigaam-v3?"));
    assert!(error("parakeet-tdt-06b-v3").contains("did you mean parakeet-tdt-0.6b-v3?"));
    let e = error("whisper");
    assert!(e.starts_with("unknown model \"whisper\". Known models: parakeet-v3, "), "{e}");
    assert_eq!(distance("kitten", "sitting"), 3);
    assert_eq!(distance("", "abc"), 3);
    let hub = Hub::at("/nowhere");
    assert_eq!(hub.find("whisper"), None);
    assert!(matches!(hub.remove("whisper"), Err(Error::UnknownModel { .. })));
  }

  /// Every other list of the models names exactly `MODELS`: the model tables of the crate docs
  /// and the readme (in order), `ALL_MODELS` in the Makefile and the tests in `tests/models.rs`.
  #[test]
  fn model_lists_match() {
    let read = |path: &str| fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path));
    let row = |m: &ModelInfo| {
      let output = if m.punctuated { "cased, punctuated" } else { "lowercase, no punctuation" };
      format!("{:.2} GB | {} | {output} |", m.size as f64 / 1e9, m.languages)
    };

    let docs: Vec<&str> =
      (include_str!("../lib.rs").lines()).filter_map(|l| l.strip_prefix("//! | `")).collect();
    let expected: Vec<String> =
      MODELS.iter().map(|m| format!("{}` | `{}` | {}", m.name, m.full_name, row(m))).collect();
    assert_eq!(docs, expected, "model table in src/lib.rs");

    let readme = read("readme.md").unwrap();
    let section = readme.split("\n## ").find(|s| s.lines().next().unwrap().ends_with(" Models"));
    let readme: Vec<String> = (section.expect("## Models in readme.md").lines())
      .filter_map(|l| l.strip_prefix("| `"))
      .map(|l| l.replacen(" (default)", "", 1))
      .collect();
    let expected: Vec<String> =
      MODELS.iter().map(|m| format!("{}` | {}", m.name, row(m))).collect();
    assert_eq!(readme, expected, "model table in readme.md");

    // The published crate has neither file: these two checks run in the repository only, and
    // fail there if either file is missing or unreadable.
    let (makefile, tests) = match (read("Makefile"), read("tests/models.rs")) {
      (Ok(makefile), Ok(tests)) => (makefile, tests),
      (Err(a), Err(b)) if [&a, &b].iter().all(|e| e.kind() == std::io::ErrorKind::NotFound) => {
        return;
      }
      (makefile, tests) => {
        panic!("Makefile: {:?}, tests/models.rs: {:?}", makefile.err(), tests.err())
      }
    };
    let mut full: Vec<&str> = MODELS.iter().map(|m| m.full_name).collect();
    full.sort();
    let mut make = Vec::new();
    for line in makefile.lines().skip_while(|l| !l.starts_with("ALL_MODELS := ")) {
      make.extend(
        line.trim_start_matches("ALL_MODELS := ").split_whitespace().filter(|w| *w != "\\"),
      );
      if !line.ends_with('\\') {
        break;
      }
    }
    make.sort();
    assert_eq!(make, full, "ALL_MODELS in the Makefile");

    let mut tested: Vec<&str> = (tests.lines())
      .filter_map(|l| l.trim_start().strip_prefix("check(\"")?.split('"').next())
      .collect();
    tested.sort();
    assert_eq!(tested, full, "tests in tests/models.rs");
  }
}
