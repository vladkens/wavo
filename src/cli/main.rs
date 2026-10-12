// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo`: speech to text with models pulled by name into the Hugging Face cache.

mod audio;
mod models;
mod output;
mod split;

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use models::Source;
use wavo::hub::{self, Hub, MODELS};
use wavo::{Backend, PrepareStep};

const USAGE: &str = "\
Usage: wavo COMMAND ...

Commands:
  run [MODEL] AUDIO [OPTIONS]  transcribe AUDIO with MODEL (parakeet-v3 by default)
  pull MODEL...                download models
  list                         list downloaded models
  rm MODEL                     delete a downloaded model
  bench MODEL AUDIO [-n N]     time the model load, the first call and N more calls (10);
                               takes --backend too

Quick start:
  wavo AUDIO [OPTIONS]         the same as wavo run AUDIO: transcribe with parakeet-v3,
                               offering to download it first if it is missing

Run options:
  --json                       print the text and its tokens with start times as JSON
  --srt                        print subtitles
  --segment SECS               split longer audio at pauses into parts of up to SECS
                               (15 on the Neural Engine, 25 for GigaAM, 30 for Whisper
                               and Moonshine, else 60; 0 for one pass; at most 15
                               on the Neural Engine)
  -y, --yes                    download a missing model without asking
  --backend auto|gpu|ane       where the encoder runs: auto (the default) takes the Neural
                               Engine for Parakeet and GigaAM once `wavo pull` prepared
                               them on a Mac, else the GPU

MODEL is a name below, its full name or a path to a .gguf file. AUDIO is wav, mp3, m4a, flac
or ogg (Vorbis), downmixed to mono and resampled to 16 kHz.
Models live in the Hugging Face cache (HF_HUB_CACHE or HF_HOME), shared with the hf CLI;
only `wavo pull` and `wavo run` (after asking, or with -y) use the network.

Models:";

/// `wavo run` splits audio into segments of up to this for models without a window of their own.
const SEGMENT_MS: u32 = 60_000;

fn usage() -> String {
  let mut usage = USAGE.to_string();
  for m in MODELS {
    let full = if m.name == m.full_name { String::new() } else { format!(" ({})", m.full_name) };
    usage += &format!("\n  {:<18} {}  {}{full}", m.name, models::gb(m.size), models::about(m));
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

/// Win over a file of the same name in `wavo FILE`.
const COMMANDS: &[&str] = &["help", "run", "pull", "list", "ls", "rm", "bench"];

fn cli(args: &[&str]) -> Result<()> {
  let help = matches!(args, [] | ["help"]) || args.iter().any(|a| matches!(*a, "-h" | "--help"));
  match args {
    _ if help => println!("{}", usage()),
    ["-V" | "--version"] => println!("wavo {}", env!("CARGO_PKG_VERSION")),
    ["run", args @ ..] => run(args)?,
    ["pull", names @ ..] if !names.is_empty() => pull(names)?,
    ["list" | "ls"] => list()?,
    ["rm", name] => rm(name)?,
    ["bench", model, audio, opts @ ..] => bench(model, audio, opts)?,
    // wavo AUDIO, also with run options first: wavo -y AUDIO
    [arg, ..] if arg.starts_with('-') || (!COMMANDS.contains(arg) && Path::new(arg).is_file()) => {
      run(args)?
    }
    _ => bail!("unknown command: wavo {}\n\n{}", args.join(" "), usage()),
  }
  Ok(())
}

/// Whether a command may download a model that isn't in the cache.
#[derive(Clone, Copy, PartialEq)]
enum Download {
  Never,
  /// After the person agrees at a prompt, when stdin and stderr are terminals.
  Ask,
  Yes,
}

/// A model by name, downloaded when `download` allows, or a `.gguf` path.
fn model_path(arg: &str, download: Download) -> Result<PathBuf> {
  let m = match models::resolve(arg)? {
    Source::Path(path) if path.is_file() => return Ok(path),
    Source::Path(path) => bail!("{}: no such file", path.display()),
    Source::Model(m) => m,
  };
  let hub = Hub::new();
  if let Some(path) = hub.find(m.name) {
    return Ok(path);
  }
  let (name, size) = (m.name, models::gb(m.size));
  let question = format!("{name} is not downloaded ({size}). Download it now?");
  if download == Download::Yes || download == Download::Ask && ask(&question) {
    return pull_one(&hub, m, &MultiProgress::new()).with_context(|| format!("pulling {name}"));
  }
  bail!("{name} is not in {hub}; run: wavo pull {name} ({size})")
}

/// Asks on stderr, yes by default. No without a terminal on stdin and stderr, so a script never
/// waits on it, and on EOF (Ctrl-D).
fn ask(question: &str) -> bool {
  use std::io::{IsTerminal, stdin};
  if !(stdin().is_terminal() && std::io::stderr().is_terminal()) {
    return false;
  }
  eprint!("{question} [Y/n] ");
  let mut answer = String::new();
  if !matches!(stdin().read_line(&mut answer), Ok(1..)) {
    eprintln!();
    return false;
  }
  matches!(answer.trim().to_lowercase().as_str(), "" | "y" | "yes")
}

fn load(path: &Path, backend: Backend) -> Result<wavo::Model> {
  wavo::Model::load_with(path, backend).with_context(|| format!("loading {}", path.display()))
}

/// The value of `--backend`.
fn backend(arg: Option<&&str>) -> Result<Backend> {
  Ok(match arg.copied() {
    Some("auto") => Backend::Auto,
    Some("gpu") => Backend::Gpu,
    Some("ane") => Backend::NeuralEngine,
    _ => bail!("--backend takes auto, gpu or ane"),
  })
}

fn run(args: &[&str]) -> Result<()> {
  const USAGE: &str =
    "usage: wavo run [MODEL] AUDIO [--json | --srt] [--segment SECS] [-y] [--backend B]";
  let (mut format, mut segment, mut download, mut rest) = (None, None, Download::Ask, Vec::new());
  let mut on = Backend::Auto;
  let mut args = args.iter();
  while let Some(&arg) = args.next() {
    match arg {
      "--backend" => on = backend(args.next())?,
      "--json" | "--srt" if format.is_none() => format = Some(arg),
      "--segment" if segment.is_none() => {
        let secs = args.next().and_then(|s| s.parse::<u32>().ok()).filter(|&s| s == 0 || s >= 5);
        let secs =
          secs.context("--segment takes whole seconds: 0 for one pass, else at least 5")?;
        segment = Some(secs.saturating_mul(1000));
      }
      "-y" | "--yes" => download = Download::Yes,
      _ if arg.starts_with("--") => bail!(USAGE),
      _ => rest.push(arg),
    }
  }
  let (model, audio) = match *rest {
    [arg] if hub::lookup(arg).is_ok() && !Path::new(arg).exists() => {
      bail!("no audio file given: wavo run {arg} AUDIO")
    }
    [audio] => (models::DEFAULT, audio),
    [model, audio] => (model, audio),
    _ => bail!(USAGE),
  };
  // Before anything is asked or downloaded.
  let meta = std::fs::metadata(audio).with_context(|| format!("reading {audio}"))?;
  ensure!(!meta.is_dir(), "{audio}: a folder, not an audio file");
  let path = model_path(model, download)?;
  let bar = ProgressBar::new_spinner().with_finish(ProgressFinish::AndClear); // on errors too
  bar.set_style(ProgressStyle::with_template("{spinner} {msg} {elapsed}")?);
  bar.enable_steady_tick(Duration::from_millis(100));
  bar.set_message("decoding the audio");
  let pcm = audio::read(audio)?;
  bar.set_message("loading the model");
  let model = load(&path, on)?;
  let mut max_ms = segment.or(model.max_audio_ms()).unwrap_or(SEGMENT_MS);
  if model.backend() == Backend::NeuralEngine {
    // Its window caps the pieces, so subtitles see every cut.
    let window = model.max_audio_ms().unwrap_or(u32::MAX);
    max_ms = if max_ms == 0 { window } else { max_ms.min(window) };
  }
  let ranges = wavo::split(&pcm, if max_ms == 0 { u32::MAX } else { max_ms });
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
    // Whisper times segments, not tokens, and Moonshine neither.
    Some(_) if matches!(models::architecture(&path).as_deref(), Some("whisper" | "moonshine")) => {
      print!("{}", output::srt_segments(tokens, ms))
    }
    Some(_) => print!("{}", output::srt(tokens, &starts, ms)),
    None => println!("{}", transcript.text),
  }
  Ok(())
}

/// A model by name, for the commands that manage the cache.
fn named(arg: &str) -> Result<&'static hub::ModelInfo> {
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
  let (hub, bars) = (Hub::new(), MultiProgress::new());
  let results: Vec<_> = std::thread::scope(|s| {
    let (hub, bars) = (&hub, &bars);
    let pulls: Vec<_> = models.iter().map(|m| s.spawn(move || pull_one(hub, m, bars))).collect();
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

/// Progress goes to a bar in `bars` on a terminal, else to a plain line on stderr at the start of
/// each try and one at the end.
fn pull_one(hub: &Hub, m: &hub::ModelInfo, bars: &MultiProgress) -> Result<PathBuf> {
  let (file, plain, mb) = (m.file(), bars.is_hidden(), |bytes: u64| bytes as f64 / 1e6);
  let style = ProgressStyle::with_template(
    "{msg}: {percent:>3}% {decimal_bytes} / {decimal_total_bytes}, {decimal_bytes_per_sec}",
  )?;
  let (mut bar, mut retries): (Option<(ProgressBar, u64)>, u32) = (None, 0);
  let mut spinner: Option<ProgressBar> = None;
  let path = hub.pull(m.name, |p| {
    if let Some(step) = p.preparing {
      let step = match step {
        PrepareStep::Build => "building",
        PrepareStep::Compile => "compiling",
        _ => "loading, 10–30 s the first time",
      };
      let msg = format!("{}: preparing for the Neural Engine ({step})", m.name);
      if plain {
        eprintln!("{msg}");
      } else {
        let style = ProgressStyle::with_template("{spinner} {msg} {elapsed}").unwrap();
        let new = || {
          let s = bars.add(ProgressBar::new_spinner().with_style(style));
          s.enable_steady_tick(Duration::from_millis(100));
          s.with_finish(ProgressFinish::AndClear)
        };
        spinner.get_or_insert_with(new).set_message(msg);
      }
      return ControlFlow::Continue(());
    }
    if let Some(e) = p.retrying {
      retries += 1;
      bar = None; // dropping a bar clears it
      bars.suspend(|| eprintln!("retrying {file} ({retries}/{}) after: {e}", hub::RETRIES));
      return ControlFlow::Continue(());
    }
    let (bar, _) = bar.get_or_insert_with(|| {
      if plain {
        let resume = (p.done > 0).then(|| format!(", resuming at {:.0} MB", mb(p.done)));
        eprintln!("downloading {file} ({:.2} GB{})", mb(p.total) / 1e3, resume.unwrap_or_default());
      }
      let bar = ProgressBar::new(p.total).with_style(style.clone()).with_message(file.clone());
      let bar = bars.add(bar.with_finish(ProgressFinish::AndClear));
      bar.set_position(p.done);
      bar.reset_eta(); // a resumed part doesn't count toward the speed
      (bar, p.done)
    });
    bar.set_position(p.done);
    ControlFlow::Continue(())
  })?;
  if let Some((bar, start)) = bar {
    bar.finish();
    let (done, size) = (bar.position(), bar.length().unwrap_or_default());
    if plain {
      let speed = mb(done - start) / bar.elapsed().as_secs_f64();
      let pct = done * 100 / size;
      eprintln!("{file}: {pct:3}% {:.0} / {:.0} MB, {speed:.1} MB/s", mb(done), mb(size));
    }
  }
  Ok(path)
}

fn list() -> Result<()> {
  let hub = Hub::new();
  let found: Vec<_> =
    MODELS.iter().filter_map(|m| hub.find(m.name).map(|path| (m, path))).collect();
  // Whether each runs on the Neural Engine; a column only where one can.
  let ane: Vec<_> = found.iter().map(|(_, path)| wavo::Model::prepared(path)).collect();
  let column = ane.iter().any(Option::is_some);
  if found.is_empty() {
    eprintln!("no models downloaded; try: wavo pull {}", models::DEFAULT);
  } else {
    println!("{:<18} {:>7}  {}PATH", "NAME", "SIZE", if column { "ANE  " } else { "" });
  }
  for ((m, path), ane) in found.iter().zip(&ane) {
    let ane = match ane {
      _ if !column => "",
      Some(true) => "yes  ",
      Some(false) => "no   ",
      None => "-    ",
    };
    let size = models::gb(std::fs::metadata(path)?.len());
    println!("{:<18} {size}  {ane}{}", m.name, path.display());
  }
  if let Some(((m, _), _)) = found.iter().zip(&ane).find(|(_, a)| **a == Some(false)) {
    eprintln!(
      "ANE no: runs on the GPU until prepared for the Neural Engine; run: wavo pull {}",
      m.name
    );
  }
  Ok(())
}

fn rm(arg: &str) -> Result<()> {
  let m = named(arg)?;
  let freed = Hub::new().remove(m.name);
  let freed = freed.with_context(|| format!("removing {}", m.name))?;
  let freed = freed.with_context(|| format!("{} is not downloaded", m.name))?;
  eprintln!("removed {} ({} freed)", m.name, models::gb(freed));
  Ok(())
}

/// Load time, first call and warm median/min over `-n` calls (10), in milliseconds.
fn bench(model: &str, audio: &str, opts: &[&str]) -> Result<()> {
  let (mut n, mut on, mut opts) = (10, Backend::Auto, opts.iter());
  while let Some(&opt) = opts.next() {
    match opt {
      "-n" => n = opts.next().and_then(|n| n.parse().ok()).context("-n takes a number")?,
      "--backend" => on = backend(opts.next())?,
      _ => bail!("usage: wavo bench MODEL AUDIO [-n N] [--backend B]"),
    }
  }
  ensure!(n > 0, "-n must be at least 1");
  let (pcm, path) = (audio::read(audio)?, model_path(model, Download::Never)?);
  let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
  let t = Instant::now();
  let model = load(&path, on)?;
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
    "load {load:.1} ms, first {first:.1} ms, warm median {median:.1} ms, min {:.1} ms, on the {}",
    warm[0],
    if model.backend() == Backend::NeuralEngine { "Neural Engine" } else { "GPU" }
  );
  Ok(())
}
