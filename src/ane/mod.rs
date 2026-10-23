// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The Neural Engine backend (macOS): a family's encoder written from its GGUF as a Core ML
//! program with one fixed 15 s window, compiled once by `prepare` and run through Core ML. The
//! frontend and the decoder stay on the CPU.
//!
//! Compiled models live in `~/Library/Caches/wavo/`, outside the Hugging Face cache:
//!
//! ```text
//! lock                                         held by prepare, prune, sweep and remove
//! <stem>.<id>.v<GRAPH>.mlmodelc                the compiled model
//! <stem>.<id>.v<GRAPH>.<program>.loaded        a program loaded it; holds its stamp
//! <stem>.<id>.v<GRAPH>.source                  the GGUF's path it was built from
//! <stem>.<id>.v<GRAPH>.tmp                     a build in progress
//! <stem>.<id>.v<GRAPH>.trash                   a compiled model being deleted
//! ```
//!
//! `<stem>` is the GGUF's file name without `.gguf` (dots as `_`) and `<id>` a hash of its content.
//! A prepare drops the other ids and versions of its stem, the older revisions of the model, so
//! the cache holds one compiled model per model name. The first load, prepare or `prepared` of a
//! process also starts a sweep in the background that deletes the compiled models whose GGUF was
//! deleted or replaced elsewhere (`hf cache rm`, Finder), and leftovers of a build or delete cut
//! short. Those without a `.source` (from before it) go after 30 days without a load.

pub mod conformer;
#[cfg_attr(not(target_os = "macos"), path = "unsupported.rs")]
mod coreml;
mod mil;
mod proto;

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};
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
const GRAPH: u32 = 2;

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
  crate::hold(&dir.join("lock"))
}

/// `<stem>.<id>`, what names the GGUF's compiled models: `<id>` is FNV-1a over the file's size,
/// modification time and first and last MiB (the header names every tensor with its offset, the
/// tail holds weights). Links to one file share it; a copy or a new download gets a new one, and
/// its prepare replaces the old one of the same stem.
fn name(path: &Path) -> Result<String> {
  const PART: u64 = 1 << 20;
  let stem = path.file_stem().unwrap_or_default().to_string_lossy();
  let stem: String = stem.chars().map(|c| if c == '.' { '_' } else { safe(c) }).collect();
  let mut f = File::open(path)?;
  let meta = f.metadata()?;
  let len = meta.len();
  let time = meta.modified()?.duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
  let mut h = FNV;
  fnv(&mut h, &len.to_le_bytes());
  fnv(&mut h, &time.to_le_bytes());
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
  sweep();
  compiled(path).ok().filter(|c| loaded(c))
}

/// Loads the GGUF's compiled model.
pub fn load(path: &Path) -> Result<Model> {
  sweep();
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

  sweep();
  let target = compiled(path)?;
  if loaded(&target) {
    let _ = record(&target, path); // a model the sweep can't place still loads
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
      let moved = fs::remove_dir_all(&package).map_err(Error::from);
      let moved = moved.and_then(|()| move_dir(&out, &target));
      let _ = fs::remove_dir_all(&out); // Core ML's temporary folder, unless it moved
      moved
    })();
    let _ = fs::remove_dir_all(&tmp);
    built?;
  }

  record(&target, path)?;
  step(PrepareStep::Load)?;
  Model::load(&target)?;
  mark(&target)?;
  Ok(true)
}

/// Writes the GGUF's path next to its compiled model, for the sweep, unless it is there: the last
/// path prepared of those sharing the model (links to one file, as a new revision's of the same
/// content), so deleting an older one elsewhere keeps it.
fn record(target: &Path, path: &Path) -> Result<()> {
  let (source, path) = (target.with_extension("source"), std::path::absolute(path)?);
  let path = path.to_string_lossy();
  if fs::read_to_string(&source).is_ok_and(|s| s == path) {
    return Ok(());
  }

  Ok(fs::write(source, path.as_bytes())?)
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
      let gone = if p.is_dir() { trash(&p) } else { fs::remove_file(&p) };
      if gone.is_ok() { n } else { 0 }
    })
    .sum()
}

/// Deletes a folder after renaming it to `.trash`, so one half deleted when the process exits is
/// never taken for a compiled model; the next sweep deletes the rest.
fn trash(dir: &Path) -> std::io::Result<()> {
  let trash = dir.with_extension("trash");
  if trash != dir {
    let _ = fs::remove_dir_all(&trash);
    fs::rename(dir, &trash)?;
  }

  fs::remove_dir_all(trash)
}

/// Starts the sweep of the cache folder on a thread of its own, once per process; off macOS,
/// where there is no cache, it does nothing.
fn sweep() {
  static SWEEP: Once = Once::new();
  if cfg!(target_os = "macos") {
    SWEEP.call_once(|| {
      let sweep = || cache().map(|dir| sweep_dir(&dir));
      let _ = std::thread::Builder::new().name("wavo-ane-sweep".into()).spawn(sweep);
    });
  }
}

/// Deletes, unless another prepare, sweep or remove holds the cache lock, the compiled models whose
/// `.source` GGUF no longer exists or now has another id, and leftover `.tmp` and `.trash` folders.
fn sweep_dir(dir: &Path) {
  let lock = File::options().create(true).truncate(false).write(true).open(dir.join("lock"));
  let Ok(lock) = lock else { return };
  if lock.try_lock().is_err() {
    return;
  }

  for p in fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()) {
    let file = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
    if file.ends_with(".tmp") || file.ends_with(".trash") {
      let _ = fs::remove_dir_all(&p);
    }

    let Some(stem) = file.strip_suffix(".source") else { continue };
    let Ok(source) = fs::read_to_string(&p) else { continue };
    let (source, id) = (Path::new(&source), stem.rsplit_once('.').map_or(stem, |(id, _)| id));
    let gone = matches!(source.try_exists(), Ok(false));
    if gone || name(source).is_ok_and(|n| n != id) {
      delete(dir, stem);
    }
  }
}

/// Deletes, under the cache lock, the compiled models that `target`, the one being prepared,
/// replaces: the other ids and graph versions of its stem (an older revision of the model, a
/// changed file, or another GGUF of the same name: one compiled model per name is the trade-off).
/// Also those without a `.source` no program has loaded for 30 days (by their newest marker): of
/// GGUFs deleted behind wavo's back before the sweep could tell.
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
    let unknown = !compiled.with_extension("source").exists();
    if replaced || (unknown && old()) {
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
  let copied = copy_dir(from, &tmp).and_then(|()| Ok(fs::rename(&tmp, to)?));
  let _ = fs::remove_dir_all(&tmp);
  copied
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
  fn names_follow_the_file() {
    let dir = scratch("names");
    let (a, b) = (dir.join("m-0.6b.gguf"), dir.join("copy").join("m-0.6b.gguf"));
    fs::create_dir_all(b.parent().unwrap()).unwrap();
    fs::write(&a, [1; 100]).unwrap();
    fs::copy(&a, &b).unwrap();
    let day = Duration::from_secs(24 * 3600);
    File::options().write(true).open(&b).unwrap().set_modified(UNIX_EPOCH + day).unwrap();
    let na = name(&a).unwrap();
    assert!(na.starts_with("m-0_6b.") && na.len() == 23, "{na}");
    assert_ne!(na, name(&b).unwrap(), "a copy of another time, another id");
    #[cfg(unix)]
    {
      let link = dir.join("link.gguf");
      std::os::unix::fs::symlink(&a, &link).unwrap();
      assert_eq!(name(&link).unwrap()[5..], na[7..], "a link shares the id");
    }

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

  #[test]
  fn sweep_drops_models_of_deleted_and_replaced_files() {
    let dir = scratch("sweep");
    let file = |n: &str, bytes: &[u8]| {
      let p = dir.join(n);
      fs::write(&p, bytes).unwrap();
      p
    };
    let entry = |gguf: &Path, source: Option<&Path>| {
      let stem = format!("{}.v{GRAPH}", name(gguf).unwrap());
      fs::create_dir_all(dir.join(format!("{stem}.mlmodelc/weights"))).unwrap();
      fs::write(dir.join(format!("{stem}.wavo.loaded")), "stamp").unwrap();
      if let Some(source) = source {
        fs::write(dir.join(format!("{stem}.source")), source.to_str().unwrap()).unwrap();
      }

      dir.join(format!("{stem}.mlmodelc"))
    };
    let (live, gone, changed) =
      (file("a.gguf", &[1; 64]), file("b.gguf", &[2; 64]), file("c.gguf", &[3; 64]));
    let old = file("d.gguf", &[4; 64]);
    let kept = [entry(&live, Some(&live)), entry(&old, None)];
    let dropped =
      [entry(&gone, Some(&gone)), entry(&changed, Some(&changed)), dir.join("x.v2.trash")];
    fs::create_dir_all(&dropped[2]).unwrap();
    fs::remove_file(gone).unwrap();
    fs::remove_file(old).unwrap();
    fs::write(&changed, [5; 64]).unwrap();

    let busy = crate::hold(&dir.join("lock")).unwrap();
    sweep_dir(&dir);
    assert!(dropped.iter().all(|p| p.exists()), "a busy lock skips the sweep");
    drop(busy);

    sweep_dir(&dir);
    assert!(kept.iter().all(|p| p.is_dir()), "a live file, and one without a source");
    assert!(dropped.iter().all(|p| !p.exists()), "a deleted file, a replaced one, the trash");
    let left = fs::read_dir(&dir)
      .unwrap()
      .filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "source"));
    assert_eq!(left.count(), 1, "the live file's source");
    fs::remove_dir_all(dir).unwrap();
  }
}
