// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Transcripts must match the reference fixtures (`make fixtures`) exactly: text, every token piece
//! and its start time (for Whisper every segment's text and start). Missing models or samples fail
//! the test.

use std::path::{Path, PathBuf};

/// The Q8_0 GGUF of `name` in the Hugging Face cache (`make models`), with the root resolved as
/// the CLI does (`src/cli/hfs.rs`).
fn model_path(name: &str) -> PathBuf {
  let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
  let user = || var("HOME").or_else(|| var("USERPROFILE")).unwrap_or_default();
  let cache = || var("XDG_CACHE_HOME").unwrap_or_else(|| user().join(".cache"));
  let home = || var("HF_HOME").unwrap_or_else(|| cache().join("huggingface"));
  let hub = var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE"));
  let hub = hub.unwrap_or_else(|| home().join("hub"));
  let repo = hub.join(format!("models--handy-computer--{name}-gguf"));
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
/// as the reference prints some of them. Whisper fixtures have `segments: N` and lines
/// `  [   0.00 ->   10.40] text` instead: then the pairs are segment texts and starts.
fn read_fixture(path: &Path) -> (String, Vec<(String, u32)>, bool) {
  let fixture = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  let mut lines = fixture.lines();
  let text = lines.next().and_then(|l| l.strip_prefix("text: ")).expect("text line");
  if text == "(empty)" {
    assert!(lines.next().is_none(), "{path:?}: tokens after an empty text");
    return (String::new(), Vec::new(), false);
  }
  let head = lines.next().expect("tokens or segments line");
  let (count, segments) = match head.strip_prefix("segments: ") {
    Some(count) => (count, true),
    None => (head.strip_prefix("tokens: ").expect("tokens line"), false),
  };
  let count: usize = count.parse().unwrap();
  let tokens: Vec<(String, u32)> = lines
    .map(|line| {
      let (times, rest) = line.trim_start().strip_prefix('[').unwrap().split_once(']').unwrap();
      let start: f64 = times.split("->").next().unwrap().trim().parse().unwrap();
      let piece = if segments { rest.trim() } else { rest.trim_start().split_once(' ').unwrap().1 };
      (piece.replace('▁', " "), (start * 1000.0).round() as u32)
    })
    .collect();
  assert_eq!(tokens.len(), count, "{path:?}: token count");
  (text.to_string(), tokens, segments)
}

fn check(name: &str, samples: &[&str]) {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let model = wavo::Model::load(model_path(name)).unwrap();
  for sample in samples {
    let pcm = read_wav(&root.join(format!("3rd/transcribe.cpp/samples/{sample}.wav")));
    let fixture = root.join(format!("tests/fixtures/{name}/{sample}.txt"));
    let (text, tokens, segments) = read_fixture(&fixture);
    let got = model.transcribe(&pcm).unwrap();
    assert_eq!(got.text, text, "{name} {sample}: text");
    let mut got: Vec<(String, u32)> =
      got.tokens.into_iter().map(|t| (t.piece.replace('▁', " "), t.start_ms)).collect();
    if segments {
      // The tokens of a segment share its start; their pieces make its text.
      let mut joined: Vec<(String, u32)> = Vec::new();
      for (piece, start) in got {
        match joined.last_mut() {
          Some(s) if s.1 == start => s.0 += &piece,
          _ => joined.push((piece, start)),
        }
      }
      let trim = |s: String| s.trim_matches([' ', '\t', '\n', '\r']).to_string();
      got = joined.into_iter().map(|(text, start)| (trim(text), start)).collect();
    }
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

#[test]
fn whisper_large_v3_turbo() {
  check("whisper-large-v3-turbo", &["jfk", "zh-short", "ru-long", "jobs-silence"]);
}

#[test]
fn whisper_tiny() {
  check("whisper-tiny", &["jfk", "zh-short", "ru-long", "jobs-silence"]);
}

#[test]
fn whisper_base() {
  check("whisper-base", &["jfk", "zh-short", "ru-long", "uk-short"]);
}

#[test]
fn whisper_small() {
  check("whisper-small", &["jfk", "zh-short", "ru-long", "uk-short"]);
}

#[test]
fn whisper_medium() {
  check("whisper-medium", &["jfk", "zh-short", "ru-long", "jobs-silence"]);
}

#[test]
fn whisper_large_v3() {
  check("whisper-large-v3", &["jfk", "zh-short", "ru-long", "jobs-silence"]);
}

#[test]
fn whisper_tiny_en() {
  check("whisper-tiny.en", &["jfk", "dots", "jobs-silence"]);
}
