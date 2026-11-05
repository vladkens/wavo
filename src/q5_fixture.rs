// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Moonshine Tiny requantized from its published Q8_0 file to Q5_0, run on the fixture clips.
//! Skipped when that model or the clips are not on disk, so `cargo test --lib` still passes
//! without `make models` and `make samples`.

use std::path::{Path, PathBuf};

use crate::gguf::write_q5_0;
use crate::gpu::{self, WeightBytes};
use crate::hub::Hub;
use crate::{Backend, Family, Model};

/// Deletes `path` when dropped.
struct Tmp(PathBuf);

impl Drop for Tmp {
  fn drop(&mut self) {
    let _ = std::fs::remove_file(&self.0);
  }
}

fn wav(sample: &str) -> Option<Vec<f32>> {
  let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("3rd");
  let mut path = root.join(format!("samples/{sample}.wav"));
  if !path.exists() {
    path = root.join(format!("transcribe.cpp/samples/{sample}.wav"));
  }
  if !path.exists() {
    return None;
  }
  let mut wav = hound::WavReader::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  assert_eq!((wav.spec().channels, wav.spec().sample_rate), (1, 16000), "{path:?}");
  Some(wav.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect())
}

/// Resident bytes of one packed upload, and the same padded matrix stored as f16.
fn packed_size(w: &WeightBytes) -> (u64, u64) {
  let codes = (w.n as u64) * (w.k as u64) / 32 * 24;
  let scales = (w.n as u64) * (w.k as u64) / 16;
  let stored = |n: u64| n.max(4).next_multiple_of(4);
  (stored(codes) + stored(scales), (w.n as u64) * (w.k as u64) * 2)
}

#[test]
fn moonshine_tiny_q5_0() {
  let Some(src) = Hub::new().find("moonshine-tiny") else { return };
  let clips = ["jfk", "dots", "jobs-silence"];
  let pcm: Vec<_> = clips.iter().map(|c| wav(c)).collect();
  if pcm.iter().any(|p| p.is_none()) {
    return;
  }

  let dst = std::env::temp_dir().join(format!("wavo-moonshine-q5-{}.gguf", std::process::id()));
  let _tmp = Tmp(dst.clone());
  let wrote = write_q5_0(&src, &dst).unwrap();
  assert!(wrote.dst_bytes < wrote.src_bytes, "{} -> {}", wrote.src_bytes, wrote.dst_bytes);
  let _guard = gpu::record_weights();
  let model = Model::load_with(&dst, Backend::Gpu).unwrap();
  let Family::Moonshine(m) = &model.family else {
    panic!("moonshine-tiny loaded as another family")
  };
  let weights = m.gpu.weight_bytes();

  let q5: Vec<_> = weights.iter().filter(|w| w.ty == 3).collect();
  let elements: usize = q5.iter().map(|w| w.rows * w.k).sum();
  assert_eq!(elements, wrote.elements, "Q5_0 uploads do not cover the converted weights");
  assert!(weights.iter().all(|w| w.ty != 2), "a Q8_0 layer was uploaded");
  for w in &q5 {
    let (packed, wide) = packed_size(w);
    assert_eq!(
      w.weight + w.scales,
      packed,
      "Q5_0 layer {}×{} is not the packed buffers",
      w.k,
      w.rows
    );
    assert!(packed < wide, "Q5_0 layer {}×{} occupies as much as f16", w.k, w.rows);
  }

  for (name, pcm) in clips.iter().zip(pcm) {
    let got = model.transcribe(&pcm.unwrap()).unwrap();
    assert!(!got.text.is_empty(), "{name}: empty transcript");
  }
}
