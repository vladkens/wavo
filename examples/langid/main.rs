// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The spoken languages of a recording over time: `langid WAV [en,ru,...]` (16 kHz mono WAV, and
//! optionally the languages it may be in).
//!
//! - whisper-tiny: `Model::detect_language` on the whole file (up to 5 windows), then
//!   `transcribe_long` with `Options::sticky_language`, each piece in the language it keeps;
//! - parakeet-ultra, which doesn't say its language: `transcribe_long` in pieces of up to 30 s, and
//!   the language of each piece's text by whatlang among Parakeet's 25 languages (or the given).
//!
//! Each runs when its model is downloaded. Consecutive pieces of one language make a span.

#[allow(dead_code)] // `name` is for `wavo serve`
#[path = "../../src/cli/lid.rs"]
mod lid;

use std::error::Error;
use std::ops::ControlFlow;
use std::time::Instant;

use wavo::{Model, Options};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let (wav, languages) = match &args[..] {
    [wav] => (wav, Vec::new()),
    [wav, list] => (wav, list.split(',').map(str::to_string).collect()),
    _ => return Err("usage: langid WAV [en,ru,...]".into()),
  };
  let mut reader = hound::WavReader::open(wav)?;
  let spec = reader.spec();
  if (spec.channels, spec.sample_rate, spec.bits_per_sample) != (1, 16000, 16) {
    return Err("the WAV must be 16 kHz mono 16-bit".into());
  }

  let pcm: Vec<f32> =
    reader.samples::<i16>().map(|s| Ok(s? as f32 / 32768.0)).collect::<Result<_>>()?;
  let hub = wavo::hub::Hub::new();
  println!("{wav}: {}", time(pcm.len()));
  if let Some(path) = hub.find("whisper-tiny") {
    whisper(&Model::load(path)?, &pcm, &languages)?;
  }

  if let Some(path) = hub.find("parakeet-ultra") {
    parakeet(&Model::load(path)?, &pcm, &languages)?;
  }

  Ok(())
}

/// `m:ss` of a sample count.
fn time(samples: usize) -> String {
  let s = samples / 16_000;
  format!("{}:{:02}", s / 60, s % 60)
}

/// A span: start and end in samples, language, text.
type Span = (usize, usize, String, String);

/// Joins a piece to the last span when it has the same language.
fn push(spans: &mut Vec<Span>, start: usize, end: usize, lang: String, text: &str) {
  match spans.last_mut() {
    Some(last) if last.2 == lang => {
      last.1 = end;
      if !text.is_empty() {
        last.3 += if last.3.is_empty() { "" } else { " " };
        last.3 += text;
      }
    }
    _ => spans.push((start, end, lang, text.to_string())),
  }
}

fn print(spans: &[Span]) {
  for (start, end, lang, text) in spans {
    let short: String = text.chars().take(70).collect();
    let more = if short.len() < text.len() { "…" } else { "" };
    println!("  {}–{}  {lang:<3} {short}{more}", time(*start), time(*end));
  }
}

/// Where each piece of `transcribe_long` starts and ends, in samples, with its transcript.
fn pieces(
  model: &Model,
  pcm: &[f32],
  o: &Options,
) -> Result<Vec<(usize, usize, wavo::Transcript)>> {
  let mut out: Vec<(usize, usize, wavo::Transcript)> = Vec::new();
  model.transcribe_long(pcm, o, |p| {
    let start = p.start_ms as usize * 16;
    if let Some(last) = out.last_mut() {
      last.1 = start;
    }

    out.push((start, pcm.len(), p.transcript.clone()));
    ControlFlow::Continue(())
  })?;
  Ok(out)
}

fn whisper(model: &Model, pcm: &[f32], languages: &[String]) -> Result<()> {
  let mut o = Options::default();
  o.languages = languages.to_vec();
  let t = Instant::now();
  let ranked = model.detect_language(pcm, &o)?;
  let top: Vec<String> =
    ranked.iter().take(3).map(|l| format!("{} {:.2}", l.code, l.probability)).collect();
  println!("\nwhisper-tiny detect_language: {} ({} ms)", top.join(", "), t.elapsed().as_millis());
  o.sticky_language = true;
  o.window_ms = Some(30_000);
  let t = Instant::now();
  let mut spans = Vec::new();
  for (start, end, transcript) in pieces(model, pcm, &o)? {
    let lang = transcript.language.clone().unwrap_or_default();
    push(&mut spans, start, end, lang, &transcript.text);
  }

  println!("whisper-tiny per piece, sticky ({} ms):", t.elapsed().as_millis());
  print(&spans);
  Ok(())
}

fn parakeet(model: &Model, pcm: &[f32], languages: &[String]) -> Result<()> {
  let allow = if languages.is_empty() { model.languages() } else { languages };

  let mut o = Options::default();
  o.window_ms = Some(30_000);
  let t = Instant::now();
  let (mut spans, mut last) = (Vec::new(), None);
  for (start, end, transcript) in pieces(model, pcm, &o)? {
    // A piece without text keeps the language before it.
    let lang = lid::language(&transcript.text, allow).map(|l| l.0).or(last);
    last = lang;
    push(&mut spans, start, end, lang.unwrap_or("?").to_string(), &transcript.text);
  }

  println!("\nparakeet-ultra per piece + whatlang ({} ms):", t.elapsed().as_millis());
  print(&spans);
  Ok(())
}
