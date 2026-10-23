// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The Neural Engine backend (macOS): a family's encoder written from its GGUF as a Core ML
//! program with one fixed 15 s window, compiled once by `prepare` and run through Core ML. The
//! frontend and the decoder stay on the CPU.
//!
//! Compiled models live in `~/Library/Caches/wavo/`, outside the Hugging Face cache, as
//! `<stem>.<id>.v<GRAPH>.mlmodelc`: the GGUF's file name without `.gguf` (dots as `_`), an id of
//! its content and the graph version. A prepare drops the other ids and versions of its stem, the
//! older revisions of the model, so the cache holds one compiled model per model name, plus those
//! of files deleted elsewhere until 30 days without a load.

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
pub use coreml::{Model, available};
pub use mil::{Program, Var};

use crate::gguf::Gguf;
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

fn unsupported(why: &str) -> Error {
  Error::NeuralEngine(why.into())
}

fn cache() -> Result<PathBuf> {
  let home = std::env::var_os("HOME").filter(|h| !h.is_empty());
  let home = home.ok_or_else(|| unsupported("HOME is not set"))?;
  Ok(PathBuf::from(home).join("Library/Caches/wavo"))
}

/// `<stem>.<id>`: what names the GGUF's compiled models. In a Hugging Face cache the file is a
/// link to `blobs/<sha256>`, and the id is the start of that hash, so copies in two caches share
/// a compiled model. Elsewhere it is FNV-1a over the file's size, modification time and first and
/// last MiB: the header names every tensor with its offset, the tail holds weights, and a file
/// rewritten in place has a new time.
fn name(path: &Path) -> Result<String> {
  let stem = path.file_stem().unwrap_or_default().to_string_lossy();
  let stem: String = stem.chars().map(|c| if c == '.' { '_' } else { safe(c) }).collect();
  let real = fs::canonicalize(path)?;
  let blob = real.file_name().unwrap_or_default().to_string_lossy();
  let in_blobs = real.parent().and_then(Path::file_name).is_some_and(|d| d == "blobs");
  if in_blobs && blob.len() == 64 && blob.bytes().all(|b| b.is_ascii_hexdigit()) {
    return Ok(format!("{stem}.{}", &blob[..16]));
  }

  const PART: u64 = 1 << 20;
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

/// The marker a load in this program touches next to a compiled model,
/// `<compiled stem>.<program>.<stamp>.loaded`. The system caches its Neural Engine compile per
/// executable (by content) and drops it on OS updates, and a load without that cache takes
/// 10–30 s, so the stamp hashes the executable's size and time and the OS version: a rebuilt or
/// updated program or OS finds no marker.
fn loaded(compiled: &Path) -> (PathBuf, String) {
  let exe = std::env::current_exe().ok();
  let name = exe.as_deref().and_then(Path::file_stem).map(|s| s.to_string_lossy().into_owned());
  let name: String = name.unwrap_or_default().chars().map(safe).collect();
  let meta = exe.and_then(|p| fs::metadata(p).ok());
  let time =
    meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok());
  let stamp = format!("{:?} {time:?} {}", meta.map(|m| m.len()), coreml::os_version());
  let mut h = FNV;
  fnv(&mut h, stamp.as_bytes());
  let prefix = format!("{}.{name}.", compiled.with_extension("").display());
  (PathBuf::from(format!("{prefix}{:08x}.loaded", h as u32)), prefix)
}

/// Touches this program's marker and deletes its markers of older stamps.
fn mark(compiled: &Path) -> Result<()> {
  let (marker, prefix) = loaded(compiled);
  File::create(&marker)?;
  for old in
    fs::read_dir(compiled.parent().unwrap()).into_iter().flatten().flatten().map(|e| e.path())
  {
    if old != marker && old.to_string_lossy().starts_with(&prefix) {
      let _ = fs::remove_file(old);
    }
  }
  Ok(())
}

/// Whether the GGUF's compiled model is in the cache and this program has loaded it before, on a
/// Mac that can run it: then a load takes about 0.1 s.
pub fn prepared(path: &Path) -> bool {
  coreml::available() && compiled(path).is_ok_and(|p| p.is_dir() && loaded(&p).0.exists())
}

/// Loads the GGUF's compiled model.
pub fn load(path: &Path) -> Result<Model> {
  if !coreml::available() {
    return Err(unsupported(
      "no Neural Engine: it needs an Apple Silicon Mac with macOS 15 or later",
    ));
  }
  let compiled = compiled(path)?;
  if !compiled.is_dir() {
    return Err(unsupported(
      "the model is not prepared for the Neural Engine: run `wavo pull` or wavo::Model::prepare",
    ));
  }
  let model = Model::load(&compiled)?;
  let _ = mark(&compiled); // a load without its marker still works
  Ok(model)
}

/// Builds and compiles the GGUF's program unless it is in the cache, then loads it once so the
/// system compiles and caches it for the Neural Engine. `false` on a Mac without one.
pub fn prepare(
  path: &Path,
  g: &Gguf,
  build: Build,
  mut progress: impl FnMut(PrepareStep) -> ControlFlow<()>,
) -> Result<bool> {
  if !coreml::available() {
    return Ok(false);
  }

  // One at a time in this process (`wavo pull` pulls models in parallel): a first prepare peaks
  // at about the model's size in memory.
  static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
  let _one = ONE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
  let mut step = |s| if progress(s).is_break() { Err(Error::Cancelled) } else { Ok(()) };
  let target = compiled(path)?;
  let dir = target.parent().unwrap();
  fs::create_dir_all(dir)?;
  prune(dir, &target);
  let lock = File::create(target.with_extension("lock"))?;
  lock.lock()?;
  if !target.is_dir() {
    step(PrepareStep::Build)?;
    let tmp = target.with_extension("tmp"); // one per model: under the lock
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

/// Deletes the compiled models of every graph version named by `key`; returns the bytes freed.
pub fn remove(key: &str) -> Result<u64> {
  let dir = cache()?;
  // `prepare`'s lock, which it creates first: none, nothing to delete. Taken before listing so that
  // a prepare under way finishes first, and left in place: a new file would let another prepare in
  // meanwhile.
  let lock = match File::options().write(true).open(dir.join(format!("{key}.v{GRAPH}.lock"))) {
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
    lock => lock?,
  };
  lock.lock()?;
  let paths: Vec<PathBuf> = (fs::read_dir(&dir)?.collect::<std::io::Result<Vec<_>>>()?.into_iter())
    .map(|e| e.path())
    .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&format!("{key}."))))
    .filter(|p| p.extension().is_none_or(|e| e != "lock"))
    .collect();
  let mut freed = 0;
  for p in paths {
    freed += size(&p);
    if p.is_dir() { fs::remove_dir_all(&p)? } else { fs::remove_file(&p)? }
  }
  Ok(freed)
}

/// Deletes the GGUF's compiled models; returns the bytes freed.
pub fn unprepare(path: &Path) -> Result<u64> {
  key(path)?.map_or(Ok(0), |key| remove(&key))
}

/// Deletes the compiled models that `target`, the one being prepared, replaces: the other ids and
/// graph versions of its stem (an older revision of the model, a changed file). Also those no
/// program has loaded for 30 days (by their newest `loaded` marker): of GGUFs deleted behind
/// wavo's back (`hf cache rm`, another cache folder).
fn prune(dir: &Path, target: &Path) {
  const MAX_AGE: Duration = Duration::from_secs(30 * 24 * 3600);
  let paths: Vec<PathBuf> =
    fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
  let name = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().into_owned();
  let model = |p: &Path| name(p).split('.').next().unwrap_or_default().to_string();
  for compiled in paths.iter().filter(|p| p.extension().is_some_and(|e| e == "mlmodelc")) {
    let stem = name(&compiled.with_extension(""));
    let replaced = compiled != target && model(compiled) == model(target);
    let mine = || paths.iter().filter(|p| name(p).starts_with(&format!("{stem}.")));
    let times = mine().chain([compiled]).filter(|p| p.extension().is_none_or(|e| e != "lock"));
    let times = times.filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok());
    // Skipped while a prepare of the same model holds its lock; lock files stay.
    let lock = fs::OpenOptions::new()
      .create(true)
      .truncate(false)
      .write(true)
      .open(compiled.with_extension("lock"));
    let free = lock.as_ref().is_ok_and(|l| l.try_lock().is_ok());
    let old = || times.max().is_some_and(|t| t.elapsed().is_ok_and(|age| age > MAX_AGE));
    if free && (replaced || old()) {
      let _ = fs::remove_dir_all(compiled);
      mine()
        .filter(|p| p.extension().is_none_or(|e| e != "lock"))
        .for_each(|p| _ = fs::remove_file(p));
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

  #[cfg(unix)]
  #[test]
  fn names_follow_the_content() {
    let dir = scratch("names");
    let sha = "0123456789abcdef".repeat(4);
    fs::create_dir_all(dir.join("blobs")).unwrap();
    fs::write(dir.join("blobs").join(&sha), [1; 100]).unwrap();
    let link = dir.join("parakeet-tdt-0.6b-v3-Q8_0.gguf");
    std::os::unix::fs::symlink(dir.join("blobs").join(&sha), &link).unwrap();
    assert_eq!(name(&link).unwrap(), "parakeet-tdt-0_6b-v3-Q8_0.0123456789abcdef");

    // Outside a Hugging Face cache: by content and time.
    let (a, b) = (dir.join("m.gguf"), dir.join("copy").join("m.gguf"));
    fs::write(&a, [1; 100]).unwrap();
    fs::create_dir_all(b.parent().unwrap()).unwrap();
    fs::write(&b, [2; 100]).unwrap();
    let (na, nb) = (name(&a).unwrap(), name(&b).unwrap());
    assert!(na.starts_with("m.") && na.len() == 18, "{na}");
    assert_ne!(na, nb);
    fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn prune_drops_what_a_prepare_replaces() {
    let dir = scratch("prune");
    let target = dir.join("m.aaaa.v1.mlmodelc");
    let dirs = ["m.aaaa.v1", "m.bbbb.v1", "m.aaaa.v0", "n.cccc.v1", "mm.dddd.v1"];
    for d in dirs {
      fs::create_dir_all(dir.join(format!("{d}.mlmodelc"))).unwrap();
      File::create(dir.join(format!("{d}.lock"))).unwrap();
      File::create(dir.join(format!("{d}.wavo.0000.loaded"))).unwrap();
    }
    prune(&dir, &target);
    let mut left: Vec<String> =
      fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    left.sort();
    let mut want: Vec<String> = ["m.aaaa.v1", "n.cccc.v1", "mm.dddd.v1"]
      .iter()
      .flat_map(|d| ["mlmodelc", "lock", "wavo.0000.loaded"].map(|e| format!("{d}.{e}")))
      .chain(["m.bbbb.v1.lock".into(), "m.aaaa.v0.lock".into()])
      .collect();
    want.sort();
    assert_eq!(left, want);
    fs::remove_dir_all(dir).unwrap();
  }
}
