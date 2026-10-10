// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo`: speech to text with models pulled by name into the Hugging Face cache.

mod hfs;
mod models;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use models::{MODELS, Source};

const USAGE: &str = "\
Usage:
  wavo run [MODEL] AUDIO         transcribe AUDIO, with parakeet-v3 unless MODEL is given
  wavo pull MODEL                download a model
  wavo list                      list downloaded models
  wavo rm MODEL                  delete a downloaded model
  wavo bench MODEL AUDIO [-n N]  time the model load, the first call and N more (10)

MODEL is a name below, its full name or a path to a .gguf file. AUDIO is a 16 kHz mono WAV.
Models live in the Hugging Face cache (HF_HUB_CACHE or HF_HOME), shared with the hf CLI;
only `wavo pull` uses the network.

Models:";

fn usage() -> String {
  let mut usage = USAGE.to_string();
  for m in MODELS {
    let full = if m.name == m.variant { String::new() } else { format!(" ({})", m.variant) };
    usage += &format!("\n  {:<18} {}  {}{full}", m.name, models::gb(m.size), m.about);
  }
  usage
}

fn main() -> ExitCode {
  let args: Vec<String> = std::env::args().skip(1).collect();
  match cli(&args.iter().map(String::as_str).collect::<Vec<_>>()) {
    Ok(()) => ExitCode::SUCCESS,
    Err(e) => {
      eprintln!("error: {e:#}");
      ExitCode::FAILURE
    }
  }
}

fn cli(args: &[&str]) -> Result<()> {
  match args {
    [] | ["help" | "-h" | "--help"] => println!("{}", usage()),
    ["-V" | "--version"] => println!("wavo {}", env!("CARGO_PKG_VERSION")),
    ["run", args @ ..] => run(args)?,
    ["pull", name] => pull(name)?,
    ["list" | "ls"] => list()?,
    ["rm", name] => rm(name)?,
    ["bench", model, audio] => bench(model, audio, 10)?,
    ["bench", model, audio, "-n", n] => bench(model, audio, n.parse().context("-n")?)?,
    _ => bail!("unknown command: wavo {}\n\n{}", args.join(" "), usage()),
  }
  Ok(())
}

/// A downloaded model by name, or a `.gguf` path. Never touches the network.
fn model_path(arg: &str) -> Result<PathBuf> {
  match models::resolve(arg)? {
    Source::Path(path) if path.is_file() => Ok(path),
    Source::Path(path) => bail!("{}: no such file", path.display()),
    Source::Model(m) => hfs::Cache::new().find(&m.repo(), &m.file()).ok_or_else(|| {
      anyhow!("{} is not downloaded; run: wavo pull {} ({})", m.name, m.name, models::gb(m.size))
    }),
  }
}

fn load(model: &str) -> Result<wavo::Model> {
  let path = model_path(model)?;
  wavo::Model::load(&path).with_context(|| format!("loading {}", path.display()))
}

fn run(args: &[&str]) -> Result<()> {
  let (model, audio) = match *args {
    [arg] if models::find(arg).is_some() && !Path::new(arg).exists() => {
      bail!("no audio file given: wavo run {arg} AUDIO")
    }
    [audio] => (models::DEFAULT, audio),
    [model, audio] => (model, audio),
    _ => bail!("usage: wavo run [MODEL] AUDIO"),
  };
  let model = load(model)?;
  println!("{}", model.transcribe(&read_wav(audio)?)?.text);
  Ok(())
}

/// A model by name, for the commands that manage the cache.
fn named(arg: &str) -> Result<&'static models::Model> {
  match models::resolve(arg)? {
    Source::Model(m) => Ok(m),
    Source::Path(_) => bail!("expected a model name, not a path: {arg}"),
  }
}

fn pull(arg: &str) -> Result<()> {
  let m = named(arg)?;
  let path = hfs::Cache::new().pull(&m.repo(), &m.file());
  println!("{}", path.with_context(|| format!("pulling {}", m.name))?.display());
  Ok(())
}

fn list() -> Result<()> {
  let cache = hfs::Cache::new();
  let found: Vec<_> =
    MODELS.iter().filter_map(|m| cache.find(&m.repo(), &m.file()).map(|path| (m, path))).collect();
  if found.is_empty() {
    eprintln!("no models downloaded; try: wavo pull {}", models::DEFAULT);
  } else {
    println!("{:<18} {:>7}  PATH", "NAME", "SIZE");
  }
  for (m, path) in found {
    println!("{:<18} {}  {}", m.name, models::gb(std::fs::metadata(&path)?.len()), path.display());
  }
  Ok(())
}

fn rm(arg: &str) -> Result<()> {
  let m = named(arg)?;
  let freed = hfs::Cache::new().remove(&m.repo(), &m.file());
  match freed.with_context(|| format!("removing {}", m.name))? {
    Some(freed) => eprintln!("removed {} ({} freed)", m.name, models::gb(freed)),
    None => bail!("{} is not downloaded", m.name),
  }
  Ok(())
}

/// Load time, first call and warm median/min over `n` calls, in milliseconds.
fn bench(model: &str, audio: &str, n: usize) -> Result<()> {
  if n == 0 {
    bail!("-n must be at least 1");
  }
  let pcm = read_wav(audio)?;
  let path = model_path(model)?;
  let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
  let t = Instant::now();
  let model = wavo::Model::load(&path).with_context(|| format!("loading {}", path.display()))?;
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
