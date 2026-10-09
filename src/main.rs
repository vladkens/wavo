// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
use std::time::Instant;

use anyhow::{Context, Result, bail};

const USAGE: &str = "usage: wavo MODEL.gguf AUDIO.wav [--tokens]
       wavo bench MODEL.gguf AUDIO.wav [-n N]";

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let args: Vec<&str> = args.iter().map(String::as_str).collect();
  match args[..] {
    ["bench", model, audio] => bench(model, audio, 10),
    ["bench", model, audio, "-n", n] => bench(model, audio, n.parse().context("-n")?),
    [model, audio] => transcribe(model, audio, false),
    [model, audio, "--tokens"] => transcribe(model, audio, true),
    _ => bail!(USAGE),
  }
}

fn transcribe(model: &str, audio: &str, tokens: bool) -> Result<()> {
  let pcm = read_wav(audio)?;
  let model = wavo::Model::load(model).with_context(|| format!("loading {model}"))?;
  let transcript = model.transcribe(&pcm)?;
  println!("{}", transcript.text);
  if tokens {
    for t in &transcript.tokens {
      println!("{:>6} {:>6} {}", t.frame, t.id, t.piece);
    }
  }
  Ok(())
}

/// Load time, first call and warm median/min over `n` calls, in milliseconds.
fn bench(model: &str, audio: &str, n: usize) -> Result<()> {
  if n == 0 {
    bail!("-n must be at least 1");
  }
  let pcm = read_wav(audio)?;
  let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
  let t = Instant::now();
  let model = wavo::Model::load(model).with_context(|| format!("loading {model}"))?;
  let load = ms(t);
  let t = Instant::now();
  model.transcribe(&pcm)?;
  let first = ms(t);
  let mut warm = (0..n)
    .map(|_| {
      let t = Instant::now();
      model.transcribe(&pcm).map(|_| ms(t))
    })
    .collect::<Result<Vec<_>, _>>()?;
  warm.sort_by(f64::total_cmp);
  let median = (warm[(n - 1) / 2] + warm[n / 2]) / 2.0;
  println!(
    "load {load:.1} ms, first {first:.1} ms, warm median {median:.1} ms, min {:.1} ms",
    warm[0]
  );
  Ok(())
}

/// Reads mono 16 kHz PCM16 or F32 samples.
fn read_wav(path: &str) -> Result<Vec<f32>> {
  let mut wav = hound::WavReader::open(path).with_context(|| format!("reading {path}"))?;
  let spec = wav.spec();
  if spec.channels != 1 || spec.sample_rate != 16000 {
    bail!("{path}: {} channels at {} Hz, expected mono 16 kHz", spec.channels, spec.sample_rate);
  }
  let pcm = match (spec.sample_format, spec.bits_per_sample) {
    (hound::SampleFormat::Int, 16) => {
      wav.samples::<i16>().map(|s| s.map(|s| s as f32 / 32768.0)).collect::<Result<_, _>>()?
    }
    (hound::SampleFormat::Float, 32) => wav.samples::<f32>().collect::<Result<_, _>>()?,
    (format, bits) => bail!("{path}: {bits}-bit {format:?} samples, expected PCM16 or F32"),
  };
  Ok(pcm)
}
