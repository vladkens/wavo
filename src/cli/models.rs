// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The models `wavo` knows by name: the Q8_0 GGUFs transcribe.cpp publishes as
//! `handy-computer/<variant>-gguf`, with sizes from its `catalog/<variant>.json`.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

pub struct Model {
  /// Short name; the published variant name is accepted too.
  pub name: &'static str,
  pub variant: &'static str,
  pub size: u64,
  pub about: &'static str,
}

impl Model {
  pub fn repo(&self) -> String {
    format!("handy-computer/{}-gguf", self.variant)
  }

  pub fn file(&self) -> String {
    format!("{}-Q8_0.gguf", self.variant)
  }
}

pub const DEFAULT: &str = "parakeet-v3";

#[rustfmt::skip]
pub const MODELS: &[Model] = &[
  Model { name: "parakeet-v3", variant: "parakeet-tdt-0.6b-v3", size: 739_508_576, about: "25 European languages" },
  Model { name: "parakeet-v2", variant: "parakeet-tdt-0.6b-v2", size: 729_574_912, about: "English" },
  Model { name: "gigaam-v3", variant: "gigaam-v3-e2e-rnnt", size: 273_724_832, about: "Russian" },
  Model { name: "gigaam-v3-e2e-ctc", variant: "gigaam-v3-e2e-ctc", size: 272_151_136, about: "Russian, CTC head" },
  Model { name: "gigaam-v3-rnnt", variant: "gigaam-v3-rnnt", size: 273_022_880, about: "Russian, lowercase without punctuation" },
  Model { name: "gigaam-v3-ctc", variant: "gigaam-v3-ctc", size: 271_803_328, about: "Russian, lowercase without punctuation, CTC head" },
  Model { name: "whisper-turbo", variant: "whisper-large-v3-turbo", size: 886_381_760, about: "100 languages" },
];

/// What a MODEL argument names.
pub enum Source {
  Path(PathBuf),
  Model(&'static Model),
}

pub fn find(name: &str) -> Option<&'static Model> {
  MODELS.iter().find(|m| m.name == name || m.variant == name)
}

/// A path when the argument ends in `.gguf` or has a path separator, else a known name.
pub fn resolve(arg: &str) -> Result<Source> {
  if arg.ends_with(".gguf") || arg.contains(['/', '\\']) {
    return Ok(Source::Path(arg.into()));
  }
  if let Some(model) = find(arg) {
    return Ok(Source::Model(model));
  }
  let names = MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ");
  let closest = (MODELS.iter().flat_map(|m| [m.name, m.variant]))
    .map(|n| (distance(arg, n), n))
    .min()
    .filter(|&(d, n)| d <= (n.len() / 3).max(2));
  match closest {
    Some((_, n)) => bail!("unknown model {arg:?}, did you mean {n}? Known models: {names}"),
    None => bail!("unknown model {arg:?}. Known models: {names}"),
  }
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

/// The model file's `general.architecture` (`whisper`, `gigaam`, …), read from its GGUF header:
/// the key comes first in the metadata, as a string (type 8) with a u64 length.
pub fn architecture(path: &Path) -> Option<String> {
  let mut head = Vec::new();
  std::fs::File::open(path).ok()?.take(4096).read_to_end(&mut head).ok()?;
  let key = b"general.architecture";
  let at = head.windows(key.len()).position(|w| w == key)? + key.len();
  let rest = head.get(at..)?;
  if u32::from_le_bytes(rest.get(..4)?.try_into().ok()?) != 8 {
    return None;
  }
  let len = u64::from_le_bytes(rest.get(4..12)?.try_into().ok()?) as usize;
  String::from_utf8(rest.get(12..12 + len)?.to_vec()).ok()
}

pub fn gb(bytes: u64) -> String {
  format!("{:.2} GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
  use std::path::Path;

  use super::*;

  fn name(arg: &str) -> &'static str {
    match resolve(arg).unwrap() {
      Source::Model(m) => m.name,
      Source::Path(p) => panic!("{arg} resolved to the path {p:?}"),
    }
  }

  fn error(arg: &str) -> String {
    resolve(arg).err().expect("an error").to_string()
  }

  #[test]
  fn architecture_from_the_header() {
    let mut gguf = b"GGUF\x03\0\0\0".to_vec();
    gguf.extend([0; 16]); // tensor and key counts
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
  fn short_and_full_names() {
    assert_eq!(name("parakeet-v3"), "parakeet-v3");
    assert_eq!(name("parakeet-tdt-0.6b-v3"), "parakeet-v3");
    assert_eq!(name("gigaam-v3"), "gigaam-v3");
    assert_eq!(name("gigaam-v3-e2e-rnnt"), "gigaam-v3");
    assert_eq!(name("gigaam-v3-ctc"), "gigaam-v3-ctc");
    assert_eq!(find(DEFAULT).unwrap().variant, "parakeet-tdt-0.6b-v3");
    let m = find("gigaam-v3").unwrap();
    assert_eq!(m.repo(), "handy-computer/gigaam-v3-e2e-rnnt-gguf");
    assert_eq!(m.file(), "gigaam-v3-e2e-rnnt-Q8_0.gguf");
    assert_eq!(name("whisper-large-v3-turbo"), "whisper-turbo");
    let m = find("whisper-turbo").unwrap();
    assert_eq!(m.repo(), "handy-computer/whisper-large-v3-turbo-gguf");
    assert_eq!(m.file(), "whisper-large-v3-turbo-Q8_0.gguf");
  }

  #[test]
  fn names_are_unambiguous() {
    for (i, a) in MODELS.iter().enumerate() {
      for b in &MODELS[i + 1..] {
        for n in [b.name, b.variant] {
          assert!(a.name != n && a.variant != n, "{} and {} share {n}", a.variant, b.variant);
        }
      }
    }
  }

  #[test]
  fn paths() {
    for arg in ["model.gguf", "./model", "dir/x", r"C:\models\x.GGUF", "a\\b"] {
      assert!(matches!(resolve(arg), Ok(Source::Path(p)) if p == Path::new(arg)), "{arg}");
    }
  }

  #[test]
  fn unknown_names() {
    assert!(error("parakeet3").contains("did you mean parakeet-v3?"));
    assert!(error("gigam-v3").contains("did you mean gigaam-v3?"));
    assert!(error("parakeet-tdt-06b-v3").contains("did you mean parakeet-tdt-0.6b-v3?"));
    let e = error("whisper");
    assert!(!e.contains("did you mean") && e.contains("Known models: parakeet-v3, "), "{e}");
    assert_eq!(distance("kitten", "sitting"), 3);
    assert_eq!(distance("", "abc"), 3);
  }
}
