// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Transcripts must match the reference fixtures (`make fixtures`) exactly: text, every token piece
//! and its start time (for Whisper every segment's text and start). Missing models or samples fail
//! the test.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

/// One model at a time: they share the GPU anyway, and several large ones at once overrun the
/// memory of an 8 GB card.
static GPU: Mutex<()> = Mutex::new(());

/// The Q8_0 GGUF of `name` in the Hugging Face cache (`make models`), with the root resolved as
/// `wavo::hub::Hub::new` does (`src/hub/cache.rs`).
fn model_path(name: &str) -> PathBuf {
  let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
  let user = || var("HOME").or_else(|| var("USERPROFILE")).unwrap_or_default();
  let cache = || var("XDG_CACHE_HOME").unwrap_or_else(|| user().join(".cache"));
  let home = || var("HF_HOME").unwrap_or_else(|| cache().join("huggingface"));
  let hub = var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE"));
  let hub = hub.unwrap_or_else(|| home().join("hub"));
  let repo = hub.join(format!("models--handy-computer--{name}-gguf"));
  let rev = std::fs::read_to_string(repo.join("refs/main"))
    .unwrap_or_else(|e| panic!("{name} is not downloaded, run `make models MODELS={name}`: {e}"));
  repo.join("snapshots").join(rev.trim()).join(format!("{name}-Q8_0.gguf"))
}

fn read_wav(path: &Path) -> Vec<f32> {
  let mut wav = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  assert_eq!((wav.spec().channels, wav.spec().sample_rate), (1, 16000), "{path:?}");
  wav.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

/// Text and `(piece, start ms)` tokens from the block `## <sample>` of `tests/fixtures/<name>.txt`,
/// its lines up to the next header: `text: ...`, `tokens: N`, then lines like
/// `  [   0.96 ->    1.00] p=0.000 ▁В`, or only `text: (empty)`. Pieces come with `▁` as a space,
/// as the reference prints some of them. Whisper fixtures have `segments: N` and lines
/// `  [   0.00 ->   10.40] text` instead: then the pairs are segment texts and starts. A block with
/// only a non-empty `text: ...` checks the text alone and that the pieces make it.
fn read_fixture(name: &str, sample: &str) -> (String, Option<Vec<(String, u32)>>, bool) {
  let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.txt"));
  let fixture = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
  let header = format!("## {sample}");
  let mut lines = fixture.lines().skip_while(|l| *l != header);
  assert!(lines.next().is_some(), "{path:?}: no {header}");
  let mut lines = lines.take_while(|l| !l.starts_with("## ")).filter(|l| !l.is_empty());
  let text = lines.next().and_then(|l| l.strip_prefix("text: ")).expect("text line");
  if text == "(empty)" {
    assert!(lines.next().is_none(), "{path:?}: tokens after an empty text");
    return (String::new(), Some(Vec::new()), false);
  }
  let Some(head) = lines.next() else { return (text.to_string(), None, false) };
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
  (text.to_string(), Some(tokens), segments)
}

fn check(name: &str, samples: &[&str]) {
  let _gpu = GPU.lock().unwrap_or_else(PoisonError::into_inner);
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let model = wavo::Model::load(model_path(name)).unwrap();
  for sample in samples {
    let pcm = read_wav(&root.join(format!("3rd/transcribe.cpp/samples/{sample}.wav")));
    let (text, tokens, segments) = read_fixture(name, sample);
    let got = model.transcribe(&pcm).unwrap();
    assert_eq!(got.text, text, "{name} {sample}: text");
    let Some(tokens) = tokens else {
      // Text only (Moonshine): the pieces make the text, but for one leading space.
      let joined: String = got.tokens.iter().map(|t| t.piece.replace('▁', " ")).collect();
      assert_eq!(joined.strip_prefix(' ').unwrap_or(&joined), text, "{name} {sample}: pieces");
      continue;
    };
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
  check("whisper-base", &["jfk", "zh-short", "uk-short", "dots"]);
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

#[test]
fn whisper_base_en() {
  check("whisper-base.en", &["jfk", "dots", "jobs-silence"]);
}

#[test]
fn whisper_small_en() {
  check("whisper-small.en", &["jfk", "dots", "jobs-silence"]);
}

#[test]
fn whisper_medium_en() {
  check("whisper-medium.en", &["jfk", "dots", "jobs-silence"]);
}

#[test]
fn whisper_large() {
  check("whisper-large", &["jfk", "zh-short", "ru-long", "uk-short"]);
}

#[test]
fn whisper_large_v2() {
  check("whisper-large-v2", &["jfk", "zh-short", "ru-long", "jobs-silence"]);
}

#[test]
fn breeze_asr_25() {
  check("Breeze-ASR-25", &["jfk", "zh-short", "zh-long", "jobs-silence"]);
}

#[test]
fn moonshine_tiny() {
  check("moonshine-tiny", &["jfk", "dots", "jobs-silence"]);
}

#[test]
fn moonshine_base() {
  check("moonshine-base", &["jfk", "dots", "jobs-silence"]);
}
