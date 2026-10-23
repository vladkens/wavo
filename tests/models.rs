// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Transcripts must match the reference fixtures (`make fixtures`) exactly: text, every token piece
//! and its start time (for Whisper every segment's text and start). Every GGUF must list the
//! languages `hub::MODELS` gives its model. Missing models or samples fail the test. A sample
//! `<clip>+lang=<code>` is `<clip>` transcribed with `Options::language` set to `<code>`. On macOS
//! the Neural Engine runs the Parakeet and GigaAM models too, in agreement with the fixtures rather
//! than exactly (`check_ane`).
//!
//! The tests need the `hub` feature, for `hub::MODELS`: the default features have it.
#![cfg(feature = "hub")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

/// The library's GGUF reader, to read metadata the API doesn't expose. It fails through `error`.
#[path = "../src/gguf.rs"]
#[allow(dead_code)]
mod gguf;

mod error {
  pub use wavo::Result;

  macro_rules! bail {
    ($($arg:tt)*) => {
      return Err(wavo::Error::Model(format!($($arg)*)))
    };
  }
  pub(crate) use bail;
}

/// One model at a time: they share the GPU anyway, and several large ones at once overrun the
/// memory of an 8 GB card.
static GPU: Mutex<()> = Mutex::new(());

/// The Q8_0 GGUF of `name` in the Hugging Face cache (`make models`), in its owner's repository
/// in `hub::MODELS`, with the root resolved as `wavo::hub::Hub::new` does (`src/hub/cache.rs`).
fn model_path(name: &str) -> PathBuf {
  let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
  let user = || var("HOME").or_else(|| var("USERPROFILE")).unwrap_or_default();
  let cache = || var("XDG_CACHE_HOME").unwrap_or_else(|| user().join(".cache"));
  let home = || var("HF_HOME").unwrap_or_else(|| cache().join("huggingface"));
  let hub = var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE"));
  let hub = hub.unwrap_or_else(|| home().join("hub"));
  let owner = wavo::hub::lookup(name).unwrap().owner;
  let repo = hub.join(format!("models--{owner}--{name}-gguf"));
  let rev = std::fs::read_to_string(repo.join("refs/main"))
    .unwrap_or_else(|e| panic!("{name} is not downloaded, run `make models MODELS={name}`: {e}"));
  repo.join("snapshots").join(rev.trim()).join(format!("{name}-Q8_0.gguf"))
}

/// A test clip: `3rd/samples/` has the ones transcribe.cpp's `samples/` lacks (`make samples-he`).
fn read_sample(sample: &str) -> Vec<f32> {
  let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("3rd");
  let mut path = root.join(format!("samples/{sample}.wav"));
  if !path.exists() {
    path = root.join(format!("transcribe.cpp/samples/{sample}.wav"));
  }
  read_wav(&path)
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

/// The GGUF's `general.languages` is the model's `langs` in `hub::MODELS`.
fn check_languages(name: &str, path: &Path) {
  let g = gguf::Gguf::open(path).unwrap();
  let codes: Vec<&str> = g.array("general.languages").unwrap();
  assert_eq!(codes, wavo::hub::lookup(name).unwrap().langs, "{name}: general.languages");
}

fn check(name: &str, samples: &[&str]) {
  let _gpu = GPU.lock().unwrap_or_else(PoisonError::into_inner);
  let path = model_path(name);
  check_languages(name, &path);
  let model = wavo::Model::load_with(&path, wavo::Backend::Gpu).unwrap();
  for sample in samples {
    let (clip, language) = match sample.split_once("+lang=") {
      Some((clip, code)) => (clip, Some(code.to_string())),
      None => (*sample, None),
    };
    let pcm = read_sample(clip);
    let (text, tokens, segments) = read_fixture(name, sample);
    let mut options = wavo::Options::default();
    options.language = language;
    let got = model.transcribe_with(&pcm, &options).unwrap();
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

/// A clip played twice, past 1.5 times the model's `max_audio_ms`, fails with `AudioTooLong`.
fn too_long(name: &str, sample: &str) {
  let _gpu = GPU.lock().unwrap_or_else(PoisonError::into_inner);
  let model = wavo::Model::load_with(model_path(name), wavo::Backend::Gpu).unwrap();
  let pcm = read_sample(sample).repeat(2);
  let got = model.transcribe(&pcm);
  assert!(matches!(got, Err(wavo::Error::AudioTooLong { .. })), "{name} {sample}: {got:?}");
}

/// Neural Engine agreement. F16 activations, 8-bit weights and arithmetic that differs between
/// chip generations change a word now and then, drop a comma or move a token by a frame or two,
/// so per clip, against the fixture when it fits the 15 s window, else against the GPU's
/// transcript of the same pieces (the cuts change punctuation at them):
/// - the words, case and punctuation aside, are within max(1, 2%) word edits of the expected ones;
///   a word split in two or two merged into one (equal once the space is removed) is no edit;
/// - at most one aligned word differs in punctuation or case alone (each printed);
/// - of the word tokens that align with the expected ones, at least 95% start within two encoder
///   frames of them.
///
/// Prepares the model first (15–40 s the first time), and fails on a Mac without a Neural
/// Engine. The corpus check that goes with it is `make compare BACKEND=ane` (agents.md).
#[cfg(target_os = "macos")]
fn check_ane(name: &str, frame_ms: u32, samples: &[&str]) {
  use std::ops::ControlFlow;

  use wavo::{Backend, DeviceKind, Model};
  type Tokens = Vec<(String, u32)>;

  let _gpu = GPU.lock().unwrap_or_else(PoisonError::into_inner);
  let path = model_path(name);
  let prepared = Model::prepare(&path, |_| ControlFlow::Continue(())).unwrap();
  assert!(prepared, "{name}: this Mac has no Neural Engine");
  let ane = Model::load_with(&path, Backend::NeuralEngine).unwrap();
  assert_eq!(ane.device().kind, DeviceKind::NeuralEngine);
  let window = ane.max_audio_ms().unwrap();
  let mut gpu = None;
  let tokens = |t: &wavo::Transcript| -> Tokens {
    t.tokens.iter().map(|t| (t.piece.replace('▁', " "), t.start_ms)).collect()
  };
  let bare = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
  // Words with letters or digits: (bare, as written).
  let words = |t: &str| -> Vec<(String, String)> {
    t.split_whitespace().map(|w| (bare(w), w.to_string())).filter(|w| !w.0.is_empty()).collect()
  };
  // Tokens with letters or digits, lowercased.
  let word_tokens = |v: &Tokens| -> Tokens {
    v.iter().filter(|t| !bare(&t.0).is_empty()).map(|t| (t.0.to_lowercase(), t.1)).collect()
  };
  for sample in samples {
    let at = format!("{name} {sample}");
    let pcm = read_sample(sample);
    let pieces: Vec<(String, Tokens, wavo::Transcript)> = if pcm.len() / 16 <= window as usize {
      let (text, want, _) = read_fixture(name, sample);
      vec![(text, want.unwrap(), ane.transcribe(&pcm).unwrap())]
    } else {
      let gpu = gpu.get_or_insert_with(|| Model::load_with(&path, Backend::Gpu).unwrap());
      (wavo::split(&pcm, window).into_iter())
        .map(|r| {
          let want = gpu.transcribe(&pcm[r.clone()]).unwrap();
          (want.text.clone(), tokens(&want), ane.transcribe(&pcm[r]).unwrap())
        })
        .collect()
    };

    let (mut expected, mut actual, mut offsets) = (Vec::new(), Vec::new(), Vec::new());
    for (text, want, got) in &pieces {
      expected.extend(words(text));
      actual.extend(words(&got.text));
      let (got, want) = (word_tokens(&tokens(got)), word_tokens(want));
      let pieces = |v: &Tokens| v.iter().map(|t| t.0.clone()).collect::<Vec<_>>();
      let pairs = align(&pieces(&got), &pieces(&want));
      offsets.extend(pairs.into_iter().map(|(g, w)| got[g].1.abs_diff(want[w].1)));
    }

    let bare_words = |v: &[(String, String)]| v.iter().map(|w| w.0.clone()).collect::<Vec<_>>();
    let (e, a) = (bare_words(&expected), bare_words(&actual));
    let edits = word_edits(&e, &a);
    let allowed = (e.len() * 2 / 100).max(1);
    let (want, got): (Vec<_>, Vec<_>) =
      pieces.iter().map(|(text, _, got)| (text.as_str(), got.text.as_str())).unzip();
    let (want, got) = (want.join(" / "), got.join(" / "));
    println!("{at}: {edits} word edits of {} words", e.len());
    assert!(edits <= allowed, "{at}: {edits} word edits\n{got}\n{want}");

    let restyled: Vec<_> = (align(&a, &e).into_iter())
      .filter(|&(i, j)| actual[i].1 != expected[j].1)
      .map(|(i, j)| (&actual[i].1, &expected[j].1))
      .collect();
    for (got, want) in &restyled {
      println!("{at}: {got:?} instead of {want:?}");
    }

    assert!(restyled.len() <= 1, "{at}: {} words differ in punctuation or case", restyled.len());
    let close = offsets.iter().filter(|&&d| d <= 2 * frame_ms).count();
    let worst = offsets.iter().max().copied().unwrap_or(0);
    let moved = offsets.iter().filter(|&&d| d > 0).count();
    println!("{at}: {moved}/{} aligned tokens moved, at most {worst} ms", offsets.len());
    assert!(close * 100 >= offsets.len() * 95, "{at}: offsets {offsets:?}");
  }
}

/// Index pairs of a longest common subsequence of `a` and `b`, in order.
#[cfg(target_os = "macos")]
fn align<T: PartialEq>(a: &[T], b: &[T]) -> Vec<(usize, usize)> {
  let (n, m) = (a.len(), b.len());
  // `l[i][j]`: the longest common subsequence of `a[i..]` and `b[j..]`.
  let mut l = vec![vec![0usize; m + 1]; n + 1];
  for i in (0..n).rev() {
    for j in (0..m).rev() {
      l[i][j] = if a[i] == b[j] { l[i + 1][j + 1] + 1 } else { l[i + 1][j].max(l[i][j + 1]) };
    }
  }

  let (mut i, mut j, mut out) = (0, 0, Vec::new());
  while i < n && j < m {
    if a[i] == b[j] {
      out.push((i, j));
      (i, j) = (i + 1, j + 1);
    } else if l[i + 1][j] >= l[i][j + 1] {
      i += 1;
    } else {
      j += 1;
    }
  }

  out
}

/// Word edits (substitutions, insertions, deletions) from `a` to `b`, where a word of one split
/// into two of the other, or two merged into one, costs nothing.
#[cfg(target_os = "macos")]
fn word_edits(a: &[String], b: &[String]) -> usize {
  let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
  for i in 0..=a.len() {
    for j in 0..=b.len() {
      if i == 0 || j == 0 {
        d[i][j] = i + j;
        continue;
      }

      let mut best = (d[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]))
        .min(d[i - 1][j] + 1)
        .min(d[i][j - 1] + 1);
      if j >= 2 && a[i - 1] == format!("{}{}", b[j - 2], b[j - 1]) {
        best = best.min(d[i - 1][j - 2]);
      }

      if i >= 2 && b[j - 1] == format!("{}{}", a[i - 2], a[i - 1]) {
        best = best.min(d[i - 2][j - 1]);
      }

      d[i][j] = best;
    }
  }

  d[a.len()][b.len()]
}

// ru-long (34 s) is past GigaAM's 25 s window but within the 37.5 s it takes.
const GIGAAM: &[&str] = &["ru", "ru-short", "ru-long"];

#[test]
fn gigaam_v3_e2e_rnnt() {
  check("gigaam-v3-e2e-rnnt", GIGAAM);
  too_long("gigaam-v3-e2e-rnnt", "ru-long");
}

#[test]
fn gigaam_v3_e2e_ctc() {
  check("gigaam-v3-e2e-ctc", GIGAAM);
  too_long("gigaam-v3-e2e-ctc", "ru-long");
}

#[test]
fn gigaam_v3_rnnt() {
  check("gigaam-v3-rnnt", GIGAAM);
  too_long("gigaam-v3-rnnt", "ru-long");
}

#[test]
fn gigaam_v3_ctc() {
  check("gigaam-v3-ctc", GIGAAM);
  too_long("gigaam-v3-ctc", "ru-long");
}

#[cfg(target_os = "macos")]
#[test]
fn gigaam_v3_e2e_rnnt_ane() {
  check_ane("gigaam-v3-e2e-rnnt", 40, GIGAAM);
}

#[cfg(target_os = "macos")]
#[test]
fn gigaam_v3_e2e_ctc_ane() {
  check_ane("gigaam-v3-e2e-ctc", 40, GIGAAM);
}

#[cfg(target_os = "macos")]
#[test]
fn gigaam_v3_rnnt_ane() {
  check_ane("gigaam-v3-rnnt", 40, GIGAAM);
}

#[cfg(target_os = "macos")]
#[test]
fn gigaam_v3_ctc_ane() {
  check_ane("gigaam-v3-ctc", 40, GIGAAM);
}

#[test]
fn parakeet_tdt_v2() {
  check("parakeet-tdt-0.6b-v2", &["jfk", "dots", "jobs-silence"]);
}

#[test]
fn parakeet_tdt_v3() {
  check("parakeet-tdt-0.6b-v3", &["jfk", "ru-short", "uk-short"]);
}

// uk-long (29 s) is within the 30 s that transcribe.cpp takes in one pass; past them its VAD head
// splits the audio, which wavo doesn't.
#[test]
fn parakeet_ultra() {
  check("parakeet-ultra", &["jfk", "ru-short", "uk-long", "jobs-silence"]);
}

#[cfg(target_os = "macos")]
#[test]
fn parakeet_tdt_v2_ane() {
  check_ane("parakeet-tdt-0.6b-v2", 80, &["jfk", "dots", "jobs-silence"]);
}

#[cfg(target_os = "macos")]
#[test]
fn parakeet_tdt_v3_ane() {
  check_ane("parakeet-tdt-0.6b-v3", 80, &["jfk", "ru-short", "uk-short"]);
}

#[cfg(target_os = "macos")]
#[test]
fn parakeet_ultra_ane() {
  check_ane("parakeet-ultra", 80, &["jfk", "ru-short", "uk-long", "jobs-silence"]);
}

#[test]
fn parakeet_tdt_v3_he() {
  check("parakeet-tdt-0.6b-v3-he", &["he-short", "he-long"]);
}

#[test]
fn whisper_large_v3_turbo() {
  check("whisper-large-v3-turbo", &["jfk", "zh-short", "ru-long", "jobs-silence"]);
}

#[test]
fn ivrit_whisper_large_v3_turbo() {
  check("ivrit-whisper-large-v3-turbo", &["he-short", "he-long"]);
}

#[test]
fn whisper_tiny() {
  // The last two with the language set: whisper-tiny takes uk-short for Russian, and ru-long is
  // told the wrong one across two windows.
  const SAMPLES: &[&str] =
    &["jfk", "zh-short", "ru-long", "jobs-silence", "uk-short+lang=uk", "ru-long+lang=uk"];
  check("whisper-tiny", SAMPLES);
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
  too_long("moonshine-tiny", "dots");
}

#[test]
fn moonshine_base() {
  check("moonshine-base", &["jfk", "dots", "jobs-silence"]);
  too_long("moonshine-base", "dots");
}
