// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Transcripts must match the reference fixtures (`make fixtures`) exactly: text, every token piece
//! and its start time. Missing models or samples fail the test.

use std::path::{Path, PathBuf};

/// The Q8_0 GGUF of `name` in the Hugging Face cache (`make models`).
fn model_path(name: &str) -> PathBuf {
  let home = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(|| {
    PathBuf::from(std::env::var_os("HOME").expect("HOME is not set")).join(".cache/huggingface")
  });
  let repo = home.join(format!("hub/models--handy-computer--{name}-gguf"));
  let rev = std::fs::read_to_string(repo.join("refs/main"))
    .unwrap_or_else(|e| panic!("{name} is not downloaded, run `make models`: {e}"));
  repo.join("snapshots").join(rev.trim()).join(format!("{name}-Q8_0.gguf"))
}

fn read_wav(path: &Path) -> Vec<f32> {
  let mut wav = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  assert_eq!((wav.spec().channels, wav.spec().sample_rate), (1, 16000), "{path:?}");
  wav.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

/// Text and `(piece, start ms)` tokens from a fixture: `text: ...`, `tokens: N`, then lines like
/// `  [   0.96 ->    1.00] p=0.000 ▁В`, or only `text: (empty)`. Pieces come with `▁` as a space,
/// as the reference prints some of them.
fn read_fixture(path: &Path) -> (String, Vec<(String, u32)>) {
  let fixture = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  let mut lines = fixture.lines();
  let text = lines.next().and_then(|l| l.strip_prefix("text: ")).expect("text line");
  if text == "(empty)" {
    assert!(lines.next().is_none(), "{path:?}: tokens after an empty text");
    return (String::new(), Vec::new());
  }
  let count = lines.next().and_then(|l| l.strip_prefix("tokens: ")).expect("tokens line");
  let count: usize = count.parse().unwrap();
  let tokens: Vec<(String, u32)> = lines
    .map(|line| {
      let (times, rest) = line.trim_start().strip_prefix('[').unwrap().split_once(']').unwrap();
      let start: f64 = times.split("->").next().unwrap().trim().parse().unwrap();
      let piece = rest.trim_start().split_once(' ').unwrap().1;
      (piece.replace('▁', " "), (start * 1000.0).round() as u32)
    })
    .collect();
  assert_eq!(tokens.len(), count, "{path:?}: token count");
  (text.to_string(), tokens)
}

fn check(name: &str, samples: &[&str]) {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let model = wavo::Model::load(model_path(name)).unwrap();
  for sample in samples {
    let pcm = read_wav(&root.join(format!("3rd/transcribe.cpp/samples/{sample}.wav")));
    let (text, tokens) = read_fixture(&root.join(format!("tests/fixtures/{name}/{sample}.txt")));
    let got = model.transcribe(&pcm).unwrap();
    assert_eq!(got.text, text, "{name} {sample}: text");
    let got: Vec<(String, u32)> =
      got.tokens.into_iter().map(|t| (t.piece.replace('▁', " "), t.start_ms)).collect();
    assert_eq!(got, tokens, "{name} {sample}: tokens");
  }
}

const GIGAAM: &[&str] = &["ru", "ru-short", "ru-long"];

#[test]
fn gigaam_v3_e2e_rnnt() {
  check("gigaam-v3-e2e-rnnt", GIGAAM);
}

#[test]
fn gigaam_v3_e2e_ctc() {
  check("gigaam-v3-e2e-ctc", GIGAAM);
}

#[test]
fn gigaam_v3_rnnt() {
  check("gigaam-v3-rnnt", GIGAAM);
}

#[test]
fn gigaam_v3_ctc() {
  check("gigaam-v3-ctc", GIGAAM);
}

#[test]
fn parakeet_tdt_v2() {
  check("parakeet-tdt-0.6b-v2", &["jfk", "dots", "jobs-silence"]);
}

#[test]
fn parakeet_tdt_v3() {
  check("parakeet-tdt-0.6b-v3", &["jfk", "ru-short", "uk-short"]);
}
