// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The Neural Engine backend (macOS): a family's encoder written from its GGUF as a Core ML
//! program with one fixed 15 s window, compiled once by `prepare` and run through Core ML. The
//! frontend and the decoder stay on the CPU.
//!
//! Compiled models live in `~/Library/Caches/wavo/`, outside the Hugging Face cache:
//!
//! ```text
//! lock                                         held by prepare, prune and remove
//! <stem>.<id>.v<GRAPH>.mlmodelc                the compiled model
//! <stem>.<id>.v<GRAPH>.<program>.loaded        a program loaded it; holds its stamp
//! <stem>.<id>.v<GRAPH>.tmp                     a build in progress
//! ```
//!
//! `<stem>` is the GGUF's file name without `.gguf` (dots as `_`) and `<id>` a hash of its content.
//! A prepare drops the other ids and versions of its stem, the older revisions of the model, so
//! the cache holds one compiled model per model name, plus those of files deleted elsewhere until
//! 30 days without a load.

pub mod conformer;
#[cfg_attr(not(target_os = "macos"), path = "unsupported.rs")]
mod coreml;
mod mil;
mod proto;

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, UNIX_EPOCH};

pub use conformer::Graph;
pub use coreml::Model;
pub use mil::{Program, Var};

use crate::gguf::Gguf;
use crate::gpu::Gpu;
use crate::{Api, Device, DeviceKind, Error, Kernels, PrepareStep, Result};

/// The audio window of every program, in samples (15 s): most dictation is shorter, and longer
/// audio is split at pauses into pieces of up to this.
pub const WINDOW: usize = 15 * 16000;
/// Bumped whenever a program changes, so models compiled by an older wavo are rebuilt.
const GRAPH: u32 = 1;

/// Writes a family's encoder into the program and returns its outputs.
pub type Build = fn(&Gguf, &mut Program) -> Result<Vec<Var>>;

/// What `Model::device` reports on the Neural Engine: `Apple M2 Neural Engine (Core ML)`, with
/// the OS version as its driver.
pub fn device() -> &'static Device {
  static DEVICE: OnceLock<Device> = OnceLock::new();
  DEVICE.get_or_init(|| Device {
    name: format!("{} Neural Engine", coreml::chip()).trim_start().to_string(),
    api: Api::CoreMl,
    kind: DeviceKind::NeuralEngine,
    driver: coreml::os_version(),
    kernels: Kernels::CoreMl,
  })
}

/// Where a family's encoder `E` runs: the GPU, or the Neural Engine's program.
pub enum Backend<E> {
  Gpu(Box<(Gpu, E)>),
  Ane(Model),
}

impl<E> Backend<E> {
  /// `ane` when given, else a GPU and the encoder `encoder` loads on it.
  pub fn new(ane: Option<Model>, encoder: impl FnOnce(&Gpu) -> Result<E>) -> Result<Self> {
    let Some(model) = ane else {
      let gpu = Gpu::new(true)?;
      let e = encoder(&gpu)?;
      return Ok(Self::Gpu(Box::new((gpu, e))));
    };
    Ok(Self::Ane(model))
  }

  pub fn device(&self) -> &Device {
    match self {
      Self::Gpu(gpu) => gpu.0.info(),
      Self::Ane(_) => device(),
    }
  }
}

fn unsupported(why: &str) -> Error {
  Error::NeuralEngine(why.into())
}

/// Whether this Mac has a Neural Engine Core ML can use; probed once.
pub fn available() -> bool {
  static AVAILABLE: OnceLock<bool> = OnceLock::new();
  *AVAILABLE.get_or_init(coreml::available)
}

fn cache() -> Result<PathBuf> {
  let home = std::env::var_os("HOME").filter(|h| !h.is_empty());
  let home = home.ok_or_else(|| unsupported("HOME is not set"))?;
  Ok(PathBuf::from(home).join("Library/Caches/wavo"))
}

/// Holds the cache's one lock until the file is dropped, creating the folder.
fn lock(dir: &Path) -> Result<File> {
  fs::create_dir_all(dir)?;
  let lock = File::options().create(true).truncate(false).write(true).open(dir.join("lock"))?;
  lock.lock()?;
  Ok(lock)
}

/// `<stem>.<id>`, what names the GGUF's compiled models: `<id>` is FNV-1a over the file's size and
/// its first and last MiB (the header names every tensor with its offset, the tail holds weights),
/// so copies of a file share them.
fn name(path: &Path) -> Result<String> {
  const PART: u64 = 1 << 20;
  let stem = path.file_stem().unwrap_or_default().to_string_lossy();
  let stem: String = stem.chars().map(|c| if c == '.' { '_' } else { safe(c) }).collect();
  let mut f = File::open(path)?;
  let len = f.metadata()?.len();
  let mut h = FNV;
  fnv(&mut h, &len.to_le_bytes());
  let mut buf = vec![0; PART.min(len) as usize];
  f.read_exact(&mut buf)?;
  fnv(&mut h, &buf);
  f.seek(SeekFrom::Start(len - buf.len() as u64))?;
  f.read_exact(&mut buf)?;
  fnv(&mut h, &buf);
  Ok(format!("{stem}.{h:016x}"))
}

/// `c` where a cache name takes it, else `_`.
fn safe(c: char) -> char {
  if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }
}

/// FNV-1a's 64-bit offset basis, and the hash of `h` extended by `bytes`.
const FNV: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(h: &mut u64, bytes: &[u8]) {
  for &b in bytes {
    *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
  }
}

/// The compiled model of a GGUF file, whether it exists or not.
fn compiled(path: &Path) -> Result<PathBuf> {
  Ok(cache()?.join(format!("{}.v{GRAPH}.mlmodelc", name(path)?)))
}

/// This program's marker next to a compiled model, `<compiled stem>.<program>.loaded`, and the
/// stamp it holds. The system caches its Neural Engine compile per executable and drops it on OS
/// updates, and a load without that cache takes 10–30 s, so the stamp is the executable's size
/// and time and the OS version: a rebuilt or updated program or OS finds another one.
fn marker(compiled: &Path) -> (PathBuf, &'static str) {
  static STAMP: OnceLock<(String, String)> = OnceLock::new();
  let (program, stamp) = STAMP.get_or_init(|| {
    let exe = std::env::current_exe().ok();
    let name = exe.as_deref().and_then(Path::file_stem).map(|s| s.to_string_lossy().into_owned());
    let meta = exe.and_then(|p| fs::metadata(p).ok());
    let time = meta.as_ref().and_then(|m| m.modified().ok());
    let time = time.and_then(|t| t.duration_since(UNIX_EPOCH).ok());
    let stamp = format!("{:?} {time:?} {}", meta.map(|m| m.len()), coreml::os_version());
    (name.unwrap_or_default().chars().map(safe).collect(), stamp)
  });
  let marker = format!("{}.{program}.loaded", compiled.with_extension("").display());
  (PathBuf::from(marker), stamp)
}

/// Whether this program has loaded `compiled` before: then a load takes about 0.1 s.
fn loaded(compiled: &Path) -> bool {
  let (marker, stamp) = marker(compiled);
  compiled.is_dir() && fs::read(marker).is_ok_and(|s| s == stamp.as_bytes())
}

/// The GGUF's compiled model when it exists and this program has loaded it before. It doesn't
/// probe Core ML.
pub fn prepared(path: &Path) -> Option<PathBuf> {
  compiled(path).ok().filter(|c| loaded(c))
}

/// Loads the GGUF's compiled model.
pub fn load(path: &Path) -> Result<Model> {
  let compiled = compiled(path)?;
  if !compiled.is_dir() && available() {
    return Err(unsupported(
      "the model is not prepared for the Neural Engine: run `wavo pull` or wavo::Model::prepare",
    ));
  }
  load_compiled(&compiled)
}

/// Loads a compiled model and marks it loaded by this program.
pub fn load_compiled(compiled: &Path) -> Result<Model> {
  if !available() {
    return Err(unsupported(
      "no Neural Engine: it needs an Apple Silicon Mac with macOS 15 or later",
    ));
  }
  let model = Model::load(compiled)?;
  let _ = mark(compiled); // a load without its marker still works
  Ok(model)
}

fn mark(compiled: &Path) -> Result<()> {
  let (marker, stamp) = marker(compiled);
  Ok(fs::write(marker, stamp)?)
}

/// Builds and compiles the GGUF's program unless it is in the cache, then loads it once so the
/// system compiles and caches it for the Neural Engine, unless this program has loaded it before.
/// `false` on a Mac without one.
pub fn prepare(
  path: &Path,
  g: &Gguf,
  build: Build,
  mut progress: impl FnMut(PrepareStep) -> ControlFlow<()>,
) -> Result<bool> {
  if !available() {
    return Ok(false);
  }
  let target = compiled(path)?;
  if loaded(&target) {
    return Ok(true);
  }

  let mut step = |s| if progress(s).is_break() { Err(Error::Cancelled) } else { Ok(()) };
  let dir = target.parent().unwrap();
  // One prepare at a time, also across threads (`wavo pull` pulls in parallel): a first one peaks
  // at about the model's size in memory.
  let _lock = lock(dir)?;
  prune(dir, &target);
  if !target.is_dir() {
    step(PrepareStep::Build)?;
    let tmp = target.with_extension("tmp");
    let _ = fs::remove_dir_all(&tmp);
    let built = (|| {
      let package = tmp.join("model.mlpackage");
      let mut p = Program::new(&package.join("Data/com.apple.CoreML/weights/weight.bin"))?;
      let outputs = build(g, &mut p)?;
      p.finish(&outputs.iter().collect::<Vec<_>>(), &package)?;
      step(PrepareStep::Compile)?;
      let out = coreml::compile(&package)?;
      fs::remove_dir_all(&package)?;
      move_dir(&out, &target)
    })();
    let _ = fs::remove_dir_all(&tmp);
    built?;
  }
  step(PrepareStep::Load)?;
  Model::load(&target)?;
  mark(&target)?;
  Ok(true)
}

/// What names the GGUF's compiled models, for `remove`; `None` off macOS, where there are none.
pub fn key(path: &Path) -> Result<Option<String>> {
  if cfg!(target_os = "macos") { name(path).map(Some) } else { Ok(None) }
}

/// Deletes the compiled models of every graph version named by `key`, waiting for a prepare under
/// way; returns the bytes freed.
pub fn remove(key: &str) -> Result<u64> {
  let dir = cache()?;
  if !dir.is_dir() {
    return Ok(0);
  }
  let _lock = lock(&dir)?;
  Ok(delete(&dir, key))
}

/// Deletes the GGUF's compiled models; returns the bytes freed.
pub fn unprepare(path: &Path) -> Result<u64> {
  key(path)?.map_or(Ok(0), |key| remove(&key))
}

/// Deletes the entries `<prefix>.*` of the cache folder; returns the bytes freed.
fn delete(dir: &Path, prefix: &str) -> u64 {
  let prefix = format!("{prefix}.");
  let entries = fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path());
  let mine =
    entries.filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&prefix)));
  mine
    .map(|p| {
      let n = size(&p);
      let gone = if p.is_dir() { fs::remove_dir_all(&p) } else { fs::remove_file(&p) };
      if gone.is_ok() { n } else { 0 }
    })
    .sum()
}

/// Deletes, under the cache lock, the compiled models that `target`, the one being prepared,
/// replaces: the other ids and graph versions of its stem (an older revision of the model, a
/// changed file). Also those no program has loaded for 30 days (by their newest marker): of GGUFs
/// deleted behind wavo's back (`hf cache rm`, another cache folder).
fn prune(dir: &Path, target: &Path) {
  const MAX_AGE: Duration = Duration::from_secs(30 * 24 * 3600);
  let paths: Vec<PathBuf> =
    fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
  let name = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().into_owned();
  let model = |p: &Path| name(p).split('.').next().unwrap_or_default().to_string();
  for compiled in paths.iter().filter(|p| p.extension().is_some_and(|e| e == "mlmodelc")) {
    let stem = name(&compiled.with_extension(""));
    let replaced = compiled != target && model(compiled) == model(target);
    let mine = paths.iter().filter(|p| name(p).starts_with(&format!("{stem}.")));
    let times = mine.filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok());
    let old = || times.max().is_some_and(|t| t.elapsed().is_ok_and(|age| age > MAX_AGE));
    if replaced || old() {
      delete(dir, &stem);
    }
  }
}

fn size(path: &Path) -> u64 {
  match fs::read_dir(path) {
    Ok(entries) => entries.flatten().map(|e| size(&e.path())).sum(),
    Err(_) => fs::metadata(path).map_or(0, |m| m.len()),
  }
}

/// Renames a folder, or copies it where renaming fails (another volume).
fn move_dir(from: &Path, to: &Path) -> Result<()> {
  if fs::rename(from, to).is_ok() {
    return Ok(());
  }
  let tmp = to.with_extension("copy");
  copy_dir(from, &tmp)?;
  fs::rename(&tmp, to)?;
  Ok(fs::remove_dir_all(from)?)
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
  fs::create_dir_all(to)?;
  for e in fs::read_dir(from)? {
    let e = e?;
    if e.file_type()?.is_dir() {
      copy_dir(&e.path(), &to.join(e.file_name()))?;
    } else {
      fs::copy(e.path(), to.join(e.file_name()))?;
    }
  }
  Ok(())
}

/// 1 for the first `valid` of `len` frames, 0 for the padding.
pub fn mask(len: usize, valid: usize) -> Vec<f32> {
  (0..len).map(|i| (i < valid) as u32 as f32).collect()
}

/// An additive attention mask over keys: 0 for the first `valid` of `len`, −10⁴ for the padding.
pub fn key_mask(len: usize, valid: usize) -> Vec<f32> {
  (0..len).map(|i| if i < valid { 0.0 } else { -1e4 }).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wavo-ane-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  #[test]
  fn names_follow_the_content() {
    let dir = scratch("names");
    let (a, b, c) =
      (dir.join("m-0.6b.gguf"), dir.join("copy").join("m-0.6b.gguf"), dir.join("n.gguf"));
    fs::create_dir_all(b.parent().unwrap()).unwrap();
    fs::write(&a, [1; 100]).unwrap();
    fs::write(&b, [1; 100]).unwrap();
    fs::write(&c, [2; 100]).unwrap();
    let na = name(&a).unwrap();
    assert!(na.starts_with("m-0_6b.") && na.len() == 23, "{na}");
    assert_eq!(na, name(&b).unwrap(), "a copy shares the name");
    assert_ne!(na[7..], name(&c).unwrap()[2..], "other content, another id");
    fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn prune_drops_what_a_prepare_replaces() {
    let dir = scratch("prune");
    let target = dir.join("m.aaaa.v1.mlmodelc");
    let dirs = ["m.aaaa.v1", "m.bbbb.v1", "m.aaaa.v0", "n.cccc.v1", "mm.dddd.v1"];
    for d in dirs {
      fs::create_dir_all(dir.join(format!("{d}.mlmodelc"))).unwrap();
      fs::write(dir.join(format!("{d}.wavo.loaded")), "stamp").unwrap();
    }
    File::create(dir.join("lock")).unwrap();
    prune(&dir, &target);
    let mut left: Vec<String> =
      fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    left.sort();
    let mut want: Vec<String> = ["m.aaaa.v1", "n.cccc.v1", "mm.dddd.v1"]
      .iter()
      .flat_map(|d| ["mlmodelc", "wavo.loaded"].map(|e| format!("{d}.{e}")))
      .chain(["lock".into()])
      .collect();
    want.sort();
    assert_eq!(left, want);
    assert_eq!(delete(&dir, "n.cccc"), 5, "the marker's bytes");
    fs::remove_dir_all(dir).unwrap();
  }
}
