// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo`: speech to text with models pulled by name into the Hugging Face cache.

mod audio;
mod hfs;
mod models;
mod output;
mod split;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use models::{MODELS, Source};

const USAGE: &str = "\
Usage:
  wavo run [MODEL] AUDIO [--json | --srt] [--segment SECS]
                                 transcribe AUDIO, with parakeet-v3 unless MODEL is given;
                                 --json adds tokens with start times, --srt prints subtitles;
                                 longer audio is split at pauses into segments of up to SECS
                                 (25 for GigaAM, 30 for Whisper, else 60; 0 for one pass)
  wavo pull MODEL...             download models, several at once
  wavo list                      list downloaded models
  wavo rm MODEL                  delete a downloaded model
  wavo bench MODEL AUDIO [-n N]  time the model load, the first call and N more (10)

MODEL is a name below, its full name or a path to a .gguf file. AUDIO is wav, mp3, m4a, flac
or ogg (Vorbis), downmixed to mono and resampled to 16 kHz.
Models live in the Hugging Face cache (HF_HUB_CACHE or HF_HOME), shared with the hf CLI;
only `wavo pull` uses the network.

Models:";

/// `wavo run` splits audio into segments of up to this for models without a window of their own.
const SEGMENT_MS: u32 = 60_000;

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
  let help = matches!(args, [] | ["help"]) || args.iter().any(|a| matches!(*a, "-h" | "--help"));
  match args {
    _ if help => println!("{}", usage()),
    ["-V" | "--version"] => println!("wavo {}", env!("CARGO_PKG_VERSION")),
    ["run", args @ ..] => run(args)?,
    ["pull", names @ ..] if !names.is_empty() => pull(names)?,
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
    Source::Model(m) => {
      let cache = hfs::Cache::new();
      cache.find(&m.repo(), &m.file()).ok_or_else(|| {
        let (name, size, root) = (m.name, models::gb(m.size), cache.describe());
        anyhow!("{name} is not in {root}; run: wavo pull {name} ({size})")
      })
    }
  }
}

fn load(path: &Path) -> Result<wavo::Model> {
  wavo::Model::load(path).with_context(|| format!("loading {}", path.display()))
}

fn run(args: &[&str]) -> Result<()> {
  const USAGE: &str = "usage: wavo run [MODEL] AUDIO [--json | --srt] [--segment SECS]";
  let (mut format, mut segment, mut rest) = (None, None, Vec::new());
  let mut args = args.iter();
  while let Some(&arg) = args.next() {
    match arg {
      "--json" | "--srt" if format.is_none() => format = Some(arg),
      "--segment" if segment.is_none() => {
        let secs = args.next().and_then(|s| s.parse::<u32>().ok()).filter(|&s| s == 0 || s >= 5);
        let secs =
          secs.context("--segment takes whole seconds: 0 for one pass, else at least 5")?;
        segment = Some(secs.saturating_mul(1000));
      }
      _ if arg.starts_with("--") => bail!(USAGE),
      _ => rest.push(arg),
    }
  }
  let (model, audio) = match *rest {
    [arg] if models::find(arg).is_some() && !Path::new(arg).exists() => {
      bail!("no audio file given: wavo run {arg} AUDIO")
    }
    [audio] => (models::DEFAULT, audio),
    [model, audio] => (model, audio),
    _ => bail!(USAGE),
  };
  let path = model_path(model)?;
  let bar = ProgressBar::new_spinner().with_finish(ProgressFinish::AndClear); // on errors too
  bar.set_style(ProgressStyle::with_template("{spinner} {msg} {elapsed}")?);
  bar.enable_steady_tick(Duration::from_millis(100));
  bar.set_message("decoding the audio");
  let pcm = audio::read(audio)?;
  bar.set_message("loading the model");
  let model = load(&path)?;
  let max_ms = segment.or(model.max_audio_ms()).unwrap_or(SEGMENT_MS);
  let ranges = split::segments(&pcm, if max_ms == 0 { usize::MAX } else { max_ms as usize * 16 });
  let n = ranges.len();
  let parts = ranges.into_iter().enumerate().map(|(i, r)| {
    bar.set_message(format!("transcribing {}/{n}", i + 1));
    Ok((r.start, model.transcribe(&pcm[r])?))
  });
  let (transcript, starts) = split::join(parts.collect::<Result<Vec<_>>>()?);
  bar.finish_and_clear();
  let (tokens, ms) = (&transcript.tokens, (pcm.len() / 16) as u32);
  match format {
    Some("--json") => println!("{}", output::json(&transcript)),
    // Whisper times segments, not tokens.
    Some(_) if models::architecture(&path).as_deref() == Some("whisper") => {
      print!("{}", output::srt_segments(tokens, ms))
    }
    Some(_) => print!("{}", output::srt(tokens, &starts, ms)),
    None => println!("{}", transcript.text),
  }
  Ok(())
}

/// A model by name, for the commands that manage the cache.
fn named(arg: &str) -> Result<&'static models::Model> {
  match models::resolve(arg)? {
    Source::Model(m) => Ok(m),
    Source::Path(_) => bail!("expected a model name, not a path: {arg}"),
  }
}

/// One thread and progress bar per distinct model.
fn pull(args: &[&str]) -> Result<()> {
  let mut models = args.iter().map(|arg| named(arg)).collect::<Result<Vec<_>>>()?;
  let mut seen = std::collections::HashSet::new();
  models.retain(|m| seen.insert(m.name));
  let (cache, bars) = (hfs::Cache::new(), MultiProgress::new());
  let results: Vec<_> = std::thread::scope(|s| {
    let (cache, bars) = (&cache, &bars);
    let pulls: Vec<_> =
      models.iter().map(|m| s.spawn(move || cache.pull(&m.repo(), &m.file(), bars))).collect();
    pulls.into_iter().map(|pull| pull.join().unwrap()).collect()
  });
  for (m, result) in models.iter().zip(&results) {
    match result {
      Ok(path) => println!("{}", path.display()),
      Err(e) => eprintln!("error: pulling {}: {e:#}", m.name),
    }
  }
  let failed = results.iter().filter(|r| r.is_err()).count();
  ensure!(failed == 0, "{failed} of {} pulls failed", models.len());
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
  let freed = freed.with_context(|| format!("removing {}", m.name))?;
  let freed = freed.with_context(|| format!("{} is not downloaded", m.name))?;
  eprintln!("removed {} ({} freed)", m.name, models::gb(freed));
  Ok(())
}

/// Load time, first call and warm median/min over `n` calls, in milliseconds.
fn bench(model: &str, audio: &str, n: usize) -> Result<()> {
  ensure!(n > 0, "-n must be at least 1");
  let (pcm, path) = (audio::read(audio)?, model_path(model)?);
  let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
  let t = Instant::now();
  let model = load(&path)?;
  let load = ms(t);
  let mut calls = (0..=n).map(|_| {
    let t = Instant::now();
    model.transcribe(&pcm).map(|_| ms(t))
  });
  let first = calls.next().unwrap()?;
  let mut warm = calls.collect::<Result<Vec<_>, _>>()?;
  warm.sort_by(f64::total_cmp);
  let median = (warm[(n - 1) / 2] + warm[n / 2]) / 2.0;
  println!(
    "load {load:.1} ms, first {first:.1} ms, warm median {median:.1} ms, min {:.1} ms",
    warm[0]
  );
  Ok(())
}
