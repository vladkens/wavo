// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Moonshine Tiny requantized from its published Q8_0 file. Tensors that moved the fixture clips
//! stay Q5_0; the rest become Q4_K. Skipped when that model or the clips are not on disk, so
//! `cargo test --lib` still passes without `make models` and `make samples`.

use std::path::{Path, PathBuf};

use crate::gguf::write_q4_k;
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

/// Tensors left at Q5_0. The others that can be Q4_K are Q4_K. q, k and v of one attention are
/// one upload, so a stack is all one type. Encoder layers are `enc.blocks.0` through
/// `enc.blocks.5`.
fn keep_q5(name: &str) -> bool {
  if name == "dec.token_embd.weight" || name == "dec.blocks.0.ffn.fc1.weight" {
    return true;
  }
  if name.contains("cross_attn.out.weight") {
    return true;
  }
  let Some(rest) = name.strip_prefix("enc.blocks.") else { return false };
  let Some(layer) = rest.chars().next() else { return false };
  if !('0'..='4').contains(&layer) {
    return false;
  }
  name.contains(".attn.") || name.ends_with(".ffn.fc1.weight")
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
  let (codes, scales) = if w.ty == 7 {
    let blocks = (w.n as u64) * (w.k as u64) / 256;
    (blocks * 128, blocks * 20)
  } else {
    ((w.n as u64) * (w.k as u64) / 32 * 24, (w.n as u64) * (w.k as u64) / 16)
  };
  let stored = |n: u64| n.max(4).next_multiple_of(4);
  (stored(codes) + stored(scales), (w.n as u64) * (w.k as u64) * 2)
}

#[test]
fn moonshine_tiny_q4_k() {
  let Some(src) = Hub::new().find("moonshine-tiny") else { return };
  let clips = ["jfk", "dots", "jobs-silence"];
  let pcm: Vec<_> = clips.iter().map(|c| wav(c)).collect();
  if pcm.iter().any(|p| p.is_none()) {
    return;
  }

  let dst = std::env::temp_dir().join(format!("wavo-moonshine-q4-{}.gguf", std::process::id()));
  let _tmp = Tmp(dst.clone());
  let wrote = write_q4_k(&src, &dst, keep_q5).unwrap();
  assert!(wrote.dst_bytes < wrote.src_bytes, "{} -> {}", wrote.src_bytes, wrote.dst_bytes);
  let _guard = gpu::record_weights();
  let model = Model::load_with(&dst, Backend::Gpu).unwrap();
  let Family::Moonshine(m) = &model.family else {
    panic!("moonshine-tiny loaded as another family")
  };
  let weights = m.gpu.weight_bytes();

  let q4: Vec<_> = weights.iter().filter(|w| w.ty == 7).collect();
  let q5: Vec<_> = weights.iter().filter(|w| w.ty == 3).collect();
  let q4_elements: usize = q4.iter().map(|w| w.rows * w.k).sum();
  let q5_elements: usize = q5.iter().map(|w| w.rows * w.k).sum();
  assert_eq!((wrote.q4, wrote.q4_elements), (64, 11_446_272));
  assert_eq!((wrote.q5, wrote.q5_elements), (33, 13_916_160));
  assert_eq!(q4_elements, wrote.q4_elements, "Q4_K uploads do not cover the converted weights");
  assert_eq!(q5_elements, wrote.q5_elements, "Q5_0 uploads do not cover the tensors left wide");
  assert!(
    weights.iter().all(|w| w.ty == 0 || w.ty == 3 || w.ty == 7),
    "uploaded type outside F32, Q5_0 and Q4_K"
  );
  for w in q4.iter().chain(&q5) {
    let (packed, wide) = packed_size(w);
    assert_eq!(
      w.weight + w.scales,
      packed,
      "layer {}×{} type {} is not the packed buffers",
      w.k,
      w.rows,
      w.ty
    );
    assert!(packed < wide, "layer {}×{} type {} occupies as much as f16", w.k, w.rows, w.ty);
  }

  for (name, pcm) in clips.iter().zip(pcm) {
    let got = model.transcribe(&pcm.unwrap()).unwrap();
    assert!(!got.text.is_empty(), "{name}: empty transcript");
  }
}
