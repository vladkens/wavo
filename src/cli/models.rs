// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! MODEL arguments: a name from `wavo::hub::MODELS` or a path to a `.gguf`.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Result;
use wavo::hub::{self, ModelInfo};

pub const DEFAULT: &str = "parakeet-ultra";

/// What a MODEL argument names.
pub enum Source {
  Path(PathBuf),
  Model(&'static ModelInfo),
}

/// A path when the argument ends in `.gguf` or has a path separator, else a known name.
pub fn resolve(arg: &str) -> Result<Source> {
  if arg.ends_with(".gguf") || arg.contains(['/', '\\']) {
    return Ok(Source::Path(arg.into()));
  }
  Ok(Source::Model(hub::lookup(arg)?))
}

/// The model's line in the help: its languages, then what sets it apart.
pub fn about(m: &ModelInfo) -> String {
  let lower = if m.punctuated { "" } else { ", lowercase without punctuation" };
  let ctc = if m.full_name.ends_with("-ctc") { ", CTC head" } else { "" };
  format!("{}{lower}{ctc}", m.languages)
}

/// The model file's `general.architecture` (`whisper`, `gigaam`, …), from its GGUF metadata:
/// the key-value pairs are read in order, skipping other values, until the key turns up.
pub fn architecture(path: &Path) -> Option<String> {
  let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);
  let mut magic = [0; 4];
  f.read_exact(&mut magic).ok()?;
  (magic == *b"GGUF").then_some(())?;
  let (_version, _tensors, kvs) = (int::<4>(&mut f)?, int::<8>(&mut f)?, int::<8>(&mut f)?);
  for _ in 0..kvs {
    let key = string(&mut f)?;
    let ty = int::<4>(&mut f)?;
    if key == b"general.architecture" && ty == 8 {
      return String::from_utf8(string(&mut f)?).ok();
    }
    skip(&mut f, ty)?;
  }
  None
}

/// Whether the model times its tokens: not Whisper, which times segments, nor Moonshine.
pub fn timed(path: &Path) -> bool {
  !matches!(architecture(path).as_deref(), Some("whisper" | "moonshine"))
}

/// A little-endian unsigned integer of `N` bytes.
fn int<const N: usize>(f: &mut impl Read) -> Option<u64> {
  let mut b = [0; 8];
  f.read_exact(&mut b[..N]).ok()?;
  Some(u64::from_le_bytes(b))
}

fn string(f: &mut impl Read) -> Option<Vec<u8>> {
  let mut s = vec![0; int::<8>(f)? as usize];
  f.read_exact(&mut s).ok()?;
  Some(s)
}

/// Skips one GGUF value of type `ty`: scalars 0–7 and 10–12, a string 8, an array 9.
fn skip(f: &mut impl Read, ty: u64) -> Option<()> {
  match ty {
    0 | 1 | 7 => int::<1>(f).map(drop),
    2 | 3 => int::<2>(f).map(drop),
    4..=6 => int::<4>(f).map(drop),
    10..=12 => int::<8>(f).map(drop),
    8 => string(f).map(drop),
    9 => {
      let (ty, n) = (int::<4>(f)?, int::<8>(f)?);
      (0..n).try_for_each(|_| skip(f, ty))
    }
    _ => None,
  }
}

pub fn gb(bytes: u64) -> String {
  format!("{:.2} GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn architecture_from_the_header() {
    let mut gguf = b"GGUF\x03\0\0\0".to_vec();
    gguf.extend(0u64.to_le_bytes()); // tensors
    gguf.extend(2u64.to_le_bytes()); // keys
    // A description that mentions the key comes first and must not match.
    gguf.extend(19u64.to_le_bytes());
    gguf.extend(b"general.description");
    gguf.extend(8u32.to_le_bytes());
    gguf.extend(27u64.to_le_bytes());
    gguf.extend(b"has a general.architecture!");
    gguf.extend(20u64.to_le_bytes());
    gguf.extend(b"general.architecture");
    gguf.extend(8u32.to_le_bytes());
    gguf.extend(7u64.to_le_bytes());
    gguf.extend(b"whisper");
    let path = std::env::temp_dir().join(format!("wavo-arch-{}.gguf", std::process::id()));
    std::fs::write(&path, &gguf).unwrap();
    assert_eq!(architecture(&path).as_deref(), Some("whisper"));
    std::fs::write(&path, b"GGUF").unwrap();
    assert_eq!(architecture(&path), None);
    std::fs::remove_file(path).unwrap();
  }

  #[test]
  fn names_and_help() {
    assert!(matches!(resolve("gigaam-v3-e2e-rnnt"), Ok(Source::Model(m)) if m.name == "gigaam-v3"));
    assert!(resolve("parakeet3").is_err());
    assert_eq!(hub::lookup(DEFAULT).unwrap().full_name, "parakeet-ultra");
    let about = |name| about(hub::lookup(name).unwrap());
    assert_eq!(about("gigaam-v3"), "Russian");
    assert_eq!(about("gigaam-v3-e2e-ctc"), "Russian, CTC head");
    assert_eq!(about("gigaam-v3-ctc"), "Russian, lowercase without punctuation, CTC head");
    assert_eq!(about("whisper-turbo"), "100 languages");
  }

  #[test]
  fn paths() {
    for arg in ["model.gguf", "./model", "dir/x", r"C:\models\x.GGUF", "a\\b"] {
      assert!(matches!(resolve(arg), Ok(Source::Path(p)) if p == Path::new(arg)), "{arg}");
    }
  }
}
