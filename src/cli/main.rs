// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo`: speech to text with models pulled by name into the Hugging Face cache.

/// `println!` that returns the error instead of panicking, as on a closed pipe (`wavo ls | head`).
macro_rules! out {
  ($($arg:tt)*) => {
    std::io::Write::write_fmt(&mut std::io::stdout(), format_args!("{}\n", format_args!($($arg)*)))
  };
}

mod audio;
mod debug;
mod http;
mod lid;
mod models;
mod output;
mod serve;
mod split;

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use wavo::hub::{self, Hub, MODELS};
use wavo::{Backend, PrepareStep};

const USAGE: &str = "\
Usage: wavo COMMAND ...

Commands:
  run [MODEL] AUDIO [OPTIONS]  transcribe AUDIO with MODEL (parakeet-ultra by default)
  pull MODEL...                download models
  list                         list downloaded models
  rm MODEL                     delete a downloaded model
  bench MODEL AUDIO [-n N]     time the model load, the first call and N more calls (10);
                               takes --backend and --device too
  serve [MODEL...] [OPTIONS]   serve the models (parakeet-ultra by default; the first
                               answers unknown model names) over OpenAI's transcription API
  devices                      list the GPUs, numbered for --device
  debug                        print the system, its GPUs and the models, for a bug report

Quick start:
  wavo AUDIO [OPTIONS]         the same as wavo run AUDIO: transcribe with parakeet-ultra,
                               offering to download it first if it is missing

Run options:
  --json                       print the text, its language and its tokens with start times
                               as JSON
  --srt                        print subtitles
  --segment SECS               split longer audio at pauses into parts of up to SECS
                               (25 for GigaAM, 30 for Whisper and Moonshine, else 60;
                               0 for one pass, up to 37.5 s for GigaAM and 45 s for
                               Moonshine; at most 15 on the Neural Engine)
  --language CODE              the language of the speech (en, he, ...) for multilingual
                               Whisper models, instead of detecting it
  --diarize                    tell who speaks with the sortformer model (downloaded as
                               MODEL is): Speaker 1: ... lines, a speaker per token and
                               the utterances in --json, a cue per speaker in --srt
  -y, --yes                    download a missing model without asking
  --backend auto|gpu|ane       where the encoder runs: auto (the default) takes the Neural
                               Engine for Parakeet and GigaAM once `wavo pull` prepared
                               them on a Mac, else the GPU
  --device auto|N|NAME         the GPU to run on: its number in wavo devices or part of its
                               name, ignoring case (the first that matches); auto: wavo's
                               default GPU (0 in wavo devices), or the Neural Engine as
                               --backend picks

Serve options:
  --host ADDR                  the address to listen on (127.0.0.1)
  --port PORT                  the port to listen on (8000)
  --api-key KEY                require `Authorization: Bearer KEY` (or set WAVO_API_KEY)
  --backend auto|gpu|ane       where the encoder runs, as in run
  --device auto|N|NAME         the GPU every model runs on, as in run, chosen at start
  --unload-after DURATION      free a model's memory after DURATION without requests (30s,
                               10m, 1h; 0 right after they stop); its next request loads it
                               again. Without it, the models given stay loaded
  -y, --yes                    download missing models without asking
  POST /v1/audio/transcriptions takes OpenAI's fields file, model, language and
  response_format (json, text, srt, vtt, verbose_json); GET /v1/models lists the downloaded
  models, GET /health the GPU and the loaded models. A downloaded model not listed loads on
  first use, one at a time. POST /asr, POST /detect-language and GET /status answer as
  whisper-asr-webservice does, for Bazarr's subtitles, with the first model.

MODEL is a name below, its full name or a path to a .gguf file (for serve, a name); for
sortformer, wavo run prints who speaks when, up to 4 speakers numbered from 1. AUDIO is
wav, mp3, m4a, flac or ogg (Vorbis), downmixed to mono and resampled to 16 kHz.
Models live in the Hugging Face cache (HF_HUB_CACHE or HF_HOME), shared with the hf CLI;
only `wavo pull`, and `wavo run` and `wavo serve` for a missing model (after asking, or with
-y), download.

Models:";

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
    Err(e) if broken_pipe(&e) => ExitCode::SUCCESS,
    Err(e) => {
      eprintln!("error: {e:#}");
      ExitCode::FAILURE
    }
  }
}

/// Whether `e` comes from a closed stdout, as in `wavo list | head`: a quiet exit then.
fn broken_pipe(e: &anyhow::Error) -> bool {
  let pipe = |x: &std::io::Error| x.kind() == std::io::ErrorKind::BrokenPipe;
  e.chain().any(|x| x.downcast_ref::<std::io::Error>().is_some_and(pipe))
}

/// Win over a file of the same name in `wavo FILE`.
const COMMANDS: &[&str] =
  &["help", "run", "pull", "list", "ls", "rm", "bench", "serve", "devices", "debug"];

fn cli(args: &[&str]) -> Result<()> {
  let help = matches!(args, [] | ["help"]) || args.iter().any(|a| matches!(*a, "-h" | "--help"));
  match args {
    _ if help => out!("{}", usage())?,
    ["-V" | "--version"] => out!("wavo {}", env!("CARGO_PKG_VERSION"))?,
    ["run", args @ ..] => run(args)?,
    ["pull", names @ ..] if !names.is_empty() => pull(names)?,
    ["list" | "ls"] => list()?,
    ["rm", name] => rm(name)?,
    ["bench", model, audio, opts @ ..] => bench(model, audio, opts)?,
    ["serve", args @ ..] => serve::serve(args)?,
    ["devices"] => devices()?,
    ["debug"] => debug::print()?,
    // wavo AUDIO, also with run options first (wavo -y AUDIO), and whatever looks like a file,
    // so that a mistyped one fails as missing
    [arg, ..] if arg.starts_with('-') || !COMMANDS.contains(arg) && file_like(arg) => run(args)?,
    ["pull"] => bail!("usage: wavo pull MODEL..."),
    ["rm", ..] => bail!("usage: wavo rm MODEL"),
    ["bench", ..] => bail!(BENCH),
    [command @ ("list" | "ls" | "devices" | "debug"), ..] => bail!("usage: wavo {command}"),
    [arg, ..] if hub::lookup(arg).is_ok() => {
      bail!("{arg} is a model, not a command; to transcribe with it: wavo run {}", args.join(" "))
    }
    _ => bail!("unknown command: wavo {}; see wavo --help", args.join(" ")),
  }
  Ok(())
}

/// Whether `arg` names a file: an existing one, a path or a name with an audio extension.
fn file_like(arg: &str) -> bool {
  Path::new(arg).is_file() || arg.contains(['/', '\\']) || audio_like(arg)
}

/// Whether `arg` ends in an audio extension.
fn audio_like(arg: &str) -> bool {
  const AUDIO: &[&str] = &["wav", "mp3", "m4a", "mp4", "aac", "flac", "ogg", "oga", "opus"];
  let ext = Path::new(arg).extension().and_then(|e| e.to_str()).unwrap_or_default();
  AUDIO.iter().any(|a| a.eq_ignore_ascii_case(ext))
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
fn model_path(arg: &str, download: Download, on: Backend) -> Result<PathBuf> {
  // `--backend gpu` and `--device` skip preparing for the Neural Engine after a download.
  let hub = Hub::new().prepare(!matches!(on, Backend::Gpu | Backend::GpuAt(_)));
  let e = match hub.resolve(arg) {
    Ok(path) => return Ok(path),
    Err(e) => e,
  };

  // A known name that `resolve` didn't find is not downloaded.
  let Ok(m) = hub::lookup(arg) else { return Err(e.into()) };
  let (name, size) = (m.name, models::gb(m.size));
  let question = format!("{name} is not downloaded ({size}). Download it now?");
  if download == Download::Yes || download == Download::Ask && ask(&question) {
    return pull_one(&hub, m, &MultiProgress::new()).with_context(|| format!("pulling {name}"));
  }

  bail!("{e}; run: wavo pull {name} ({size})")
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

/// The backend for `--backend` and `--device`: a device other than `auto` is a GPU of
/// `wavo::devices()`, by its number there, or the first whose name contains it. The GPUs of the
/// first API come first: DX12's are listed only when none of those matches.
fn target(on: Backend, device: Option<&str>) -> Result<Backend> {
  let Some(arg) = device.filter(|d| !d.eq_ignore_ascii_case("auto")) else { return Ok(on) };
  ensure!(
    on != Backend::NeuralEngine,
    "--device picks a GPU, and --backend ane runs on the Neural Engine: drop one of them"
  );
  let name = |d: &wavo::Device| d.name.to_lowercase().contains(&arg.to_lowercase());
  let find = |gpus: &[wavo::Device]| match arg.parse::<usize>() {
    Ok(i) => (i < gpus.len()).then_some(i),
    Err(_) => gpus.iter().position(name),
  };
  if let Some(i) = find(&wavo::first_gpus().context("listing the GPUs")?) {
    return Ok(Backend::GpuAt(i));
  }

  let gpus = gpus()?;
  let Some(i) = find(&gpus) else {
    bail!(
      "no GPU matches --device {arg}; choose auto, a number or part of a name:\n{}",
      numbered(&gpus)
    )
  };

  Ok(Backend::GpuAt(i))
}

/// The value of `--device`.
fn device<'a>(arg: Option<&&'a str>) -> Result<&'a str> {
  let arg = arg.copied().filter(|a| !a.is_empty());
  arg.context("--device takes auto, a number from wavo devices or part of a GPU's name")
}

/// The GPUs of `wavo::devices()`, which lists the Neural Engine last.
fn gpus() -> Result<Vec<wavo::Device>> {
  let mut devices = wavo::devices().context("listing the GPUs")?;
  devices.retain(|d| d.kind != wavo::DeviceKind::NeuralEngine);
  Ok(devices)
}

/// The GPUs, one per line with its number for `--device`; the first is `--device auto`'s.
fn numbered(gpus: &[wavo::Device]) -> String {
  let line = |(i, d)| format!("{i:>3}  {d}{}", if i == 0 { "  (auto)" } else { "" });
  gpus.iter().enumerate().map(line).collect::<Vec<_>>().join("\n")
}

/// `wavo devices`: the GPUs, then the Neural Engine, which `--backend ane` takes.
fn devices() -> Result<()> {
  let devices = wavo::devices().context("listing the GPUs")?;
  let (gpus, ane): (Vec<_>, Vec<_>) =
    devices.into_iter().partition(|d| d.kind != wavo::DeviceKind::NeuralEngine);
  out!("{}", numbered(&gpus))?;
  for d in ane {
    out!("  -  {d}  (--backend ane)")?;
  }

  Ok(())
}

/// The diarization model of `--diarize`.
const DIARIZER: &str = "sortformer";

fn run(args: &[&str]) -> Result<()> {
  const USAGE: &str = "usage: wavo run [MODEL] AUDIO [--json | --srt] [--segment SECS] \
                       [--language CODE] [--diarize] [--backend B] [--device D] [-y]";
  let (mut format, mut segment, mut download, mut rest) = (None, None, Download::Ask, Vec::new());
  let mut diarize = false;
  let mut options = wavo::Options::default();
  let (mut on, mut gpu) = (Backend::Auto, None);
  let mut args = args.iter();
  while let Some(&arg) = args.next() {
    match arg {
      "--backend" => on = backend(args.next())?,
      "--device" => gpu = Some(device(args.next())?),
      "--json" | "--srt" if format.is_none() => format = Some(arg),
      "--json" | "--srt" if format != Some(arg) => bail!("--json and --srt: choose one"),
      "--segment" if segment.is_none() => {
        let secs = args.next().and_then(|s| s.parse::<u32>().ok()).filter(|&s| s == 0 || s >= 5);
        let secs =
          secs.context("--segment takes whole seconds: 0 for one pass, else at least 5")?;
        segment = Some(secs.saturating_mul(1000));
      }
      "--language" if options.language.is_none() => {
        let code = args.next().context("--language takes a code such as en or he")?;
        options.language = Some(code.to_string());
      }
      "-y" | "--yes" => download = Download::Yes,
      "--diarize" if !diarize => diarize = true,
      _ if arg.starts_with('-') => bail!("unknown or repeated option {arg}\n{USAGE}"),
      _ => rest.push(arg),
    }
  }
  let (model, audio) = match *rest {
    [arg] if hub::lookup(arg).is_ok() && !Path::new(arg).exists() => {
      bail!("no audio file given: wavo run {arg} AUDIO")
    }
    [audio] => (models::DEFAULT, audio),
    [model, audio] if !audio_like(model) => (model, audio),
    [audio, model] if !audio_like(model) => {
      bail!("the model comes first: wavo run {model} {audio}")
    }
    _ if rest.iter().filter(|a| audio_like(a)).count() > 1 => {
      bail!("wavo takes one audio file at a time: run it once per file")
    }
    _ => bail!(USAGE),
  };
  // Before anything is asked or downloaded.
  let on = target(on, gpu)?;
  let meta = match std::fs::metadata(audio) {
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!("no such file: {audio}"),
    meta => meta.with_context(|| format!("reading {audio}"))?,
  };
  ensure!(!meta.is_dir(), "{audio}: a folder, not an audio file");
  let path = model_path(model, download, on)?;
  let diarizer = models::architecture(&path).is_some_and(|a| a == "sortformer");
  let speakers = match diarize && !diarizer {
    true => Some(model_path(DIARIZER, download, Backend::Gpu)?),
    false => None,
  };
  let bar = spinner(); // cleared on errors too
  bar.set_message("decoding the audio");
  let pcm = audio::read(audio)?;
  if diarizer {
    let turns = diarize_with(&path, &pcm, &bar)?;
    bar.finish_and_clear();
    if turns.is_empty() {
      eprintln!("no speech found in {audio}");
    }

    let text = output::turns(&turns, format);
    return Ok(std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?);
  }

  bar.set_message("loading the model");
  let model = load(&path, on)?;
  if let Some(warning) = model.device().warning() {
    bar.suspend(|| eprintln!("warning: {warning}"));
  }

  bar.set_message("transcribing");
  let each = |done: usize, n| bar.set_message(format!("transcribing {}/{n}", (done + 1).min(n)));
  let (transcript, starts) = split::transcribe(&model, &pcm, segment, &options, each)?;
  bar.finish_and_clear();
  // Silence or noise: not an error, so that a batch of files goes on.
  if transcript.text.is_empty() {
    eprintln!("no speech recognized in {audio}");
  }

  let utterances = match speakers {
    Some(diarizer) => {
      let bar = spinner();
      Some(transcript.utterances(&diarize_with(&diarizer, &pcm, &bar)?))
    }
    None => None,
  };

  let (tokens, ms, timed) = (&transcript.tokens, (pcm.len() / 16) as u32, models::timed(&path));
  let text = match (format, &utterances) {
    (Some("--json"), _) => output::json(&transcript, utterances.as_deref()) + "\n",
    (Some(_), Some(u)) => output::srt(&output::speaker_cues(tokens, &starts, u, ms, timed)),
    (Some(_), None) => output::srt(&output::cues(tokens, &starts, ms, timed)),
    (None, Some(u)) => output::by_speaker(u),
    (None, None) if transcript.text.is_empty() => String::new(),
    (None, None) => format!("{}\n", transcript.text),
  };
  Ok(std::io::Write::write_all(&mut std::io::stdout(), text.as_bytes())?)
}

/// The turns of `pcm` by the diarization model at `path`, with progress on `bar`.
fn diarize_with(path: &Path, pcm: &[f32], bar: &ProgressBar) -> Result<Vec<wavo::Turn>> {
  bar.set_message("loading the diarization model");
  let diarizer =
    wavo::Diarizer::load(path).with_context(|| format!("loading {}", path.display()))?;
  bar.set_message("telling the speakers apart");
  Ok(diarizer.diarize(pcm)?)
}

/// A model by name, for the commands that manage the cache.
fn named(arg: &str) -> Result<&'static hub::ModelInfo> {
  ensure!(
    !arg.ends_with(".gguf") && !arg.contains(['/', '\\']),
    "expected a model name, not a path: {arg}"
  );
  Ok(hub::lookup(arg)?)
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
      Ok(path) => out!("{}", path.display())?,
      Err(e) => eprintln!("error: pulling {}: {e:#}", m.name),
    }
  }
  let failed = results.iter().filter(|r| r.is_err()).count();
  ensure!(failed == 0, "{failed} of {} pulls failed", models.len());
  Ok(())
}

/// A ticking spinner with its message and elapsed time, cleared when dropped.
fn spinner() -> ProgressBar {
  let style = ProgressStyle::with_template("{spinner} {msg} {elapsed}").unwrap();
  let bar = ProgressBar::new_spinner().with_style(style).with_finish(ProgressFinish::AndClear);
  bar.enable_steady_tick(Duration::from_millis(100));
  bar
}

/// Progress goes to a bar in `bars` on a terminal, else to a plain line on stderr at the start of
/// each try and one at the end.
fn pull_one(hub: &Hub, m: &hub::ModelInfo, bars: &MultiProgress) -> Result<PathBuf> {
  let (file, plain, mb) = (m.file(), bars.is_hidden(), |bytes: u64| bytes as f64 / 1e6);
  let style = ProgressStyle::with_template(
    "{msg}: {percent:>3}% {decimal_bytes} / {decimal_total_bytes}, {decimal_bytes_per_sec}",
  )?;
  let (mut bar, mut retries): (Option<(ProgressBar, u64)>, u32) = (None, 0);
  let mut prepare: Option<ProgressBar> = None;
  let path = hub.pull(m.name, |p| {
    if let (Some(_), Some(e)) = (p.preparing, p.retrying) {
      prepare = None; // dropping a spinner clears it
      bars.suspend(|| eprintln!("warning: {} runs on the GPU, preparing it failed: {e}", m.name));
      return ControlFlow::Continue(());
    }

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
        prepare.get_or_insert_with(|| bars.add(self::spinner())).set_message(msg);
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
  let found = Hub::new().downloaded();
  // Whether each runs on the Neural Engine; a column only where one can.
  let ane: Vec<_> = found.iter().map(|(_, path)| wavo::Model::prepared(path)).collect();
  let column = ane.iter().any(Option::is_some);
  if found.is_empty() {
    eprintln!("no models downloaded; try: wavo pull {}", models::DEFAULT);
  } else {
    out!("{:<18} {:>7}  {}PATH", "NAME", "SIZE", if column { "ANE  " } else { "" })?;
  }

  for ((m, path), ane) in found.iter().zip(&ane) {
    let ane = match ane {
      _ if !column => "",
      Some(true) => "yes  ",
      Some(false) => "no   ",
      None => "-    ",
    };
    let size = models::gb(std::fs::metadata(path)?.len());
    out!("{:<18} {size}  {ane}{}", m.name, path.display())?;
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

const BENCH: &str = "usage: wavo bench MODEL AUDIO [-n N] [--backend B] [--device D]";

/// Load time, first call and warm median/min over `-n` calls (10), in milliseconds.
fn bench(model: &str, audio: &str, opts: &[&str]) -> Result<()> {
  let (mut n, mut on, mut gpu, mut opts) = (10, Backend::Auto, None, opts.iter());
  while let Some(&opt) = opts.next() {
    match opt {
      "-n" => {
        let calls = opts.next().and_then(|n| n.parse().ok()).filter(|&n| n > 0);
        n = calls.context("-n takes a whole number of calls, at least 1")?;
      }
      "--backend" => on = backend(opts.next())?,
      "--device" => gpu = Some(device(opts.next())?),
      _ => bail!(BENCH),
    }
  }

  let on = target(on, gpu)?;
  let (pcm, path) = (audio::read(audio)?, model_path(model, Download::Never, on)?);
  let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
  let t = Instant::now();
  // A diarization model's call is `diarize`.
  let (model, diarizer) = match models::architecture(&path).as_deref() {
    Some("sortformer") => (None, Some(wavo::Diarizer::load(&path)?)),
    _ => (Some(load(&path, on)?), None),
  };
  let load = ms(t);
  let device = model.as_ref().map(|m| m.device()).or(diarizer.as_ref().map(|d| d.device()));
  eprintln!("device: {}", device.unwrap());
  let mut calls = (0..=n).map(|_| {
    let t = Instant::now();
    match (&model, &diarizer) {
      (Some(m), _) => m.transcribe(&pcm).map(|_| ms(t)),
      (_, Some(d)) => d.diarize(&pcm).map(|_| ms(t)),
      _ => unreachable!(),
    }
  });
  let first = calls.next().unwrap()?;
  let mut warm = calls.collect::<Result<Vec<_>, _>>()?;
  warm.sort_by(f64::total_cmp);
  let median = (warm[(n - 1) / 2] + warm[n / 2]) / 2.0;
  out!(
    "load {load:.1} ms, first {first:.1} ms, warm median {median:.1} ms, min {:.1} ms",
    warm[0]
  )?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The error `wavo ARGS` fails with; each fails before any model, download or GPU.
  fn error(args: &str) -> String {
    format!("{:#}", cli(&args.split(' ').collect::<Vec<_>>()).unwrap_err())
  }

  #[test]
  fn errors_name_the_problem() {
    assert_eq!(error("tlak.m4a"), "no such file: tlak.m4a");
    assert_eq!(error("dir/talk"), "no such file: dir/talk");
    assert_eq!(error("run gigaam-v3 TALK.WAV"), "no such file: TALK.WAV");
    let one = "wavo takes one audio file at a time: run it once per file";
    assert_eq!(error("a.mp3 b.mp3"), one);
    assert_eq!(error("run a.mp3 b.wav c.ogg"), one);
    assert_eq!(
      error("run talk.wav gigaam-v3"),
      "the model comes first: wavo run gigaam-v3 talk.wav"
    );
    assert_eq!(
      error("gigaam-v3 talk.wav"),
      "gigaam-v3 is a model, not a command; to transcribe with it: wavo run gigaam-v3 talk.wav"
    );
    assert_eq!(error("frobnicate"), "unknown command: wavo frobnicate; see wavo --help");
    assert_eq!(error("pull"), "usage: wavo pull MODEL...");
    assert_eq!(error("bench gigaam-v3"), BENCH);
    assert_eq!(error("list x"), "usage: wavo list");
    assert_eq!(
      error("bench gigaam-v3 talk.wav -n 0"),
      "-n takes a whole number of calls, at least 1"
    );
    assert!(error("talk.wav --jsn").starts_with("unknown or repeated option --jsn\nusage: "));
    assert!(error("talk.wav -n 5").starts_with("unknown or repeated option -n\n"));
    assert_eq!(error("talk.wav --json --srt"), "--json and --srt: choose one");
    assert_eq!(error("serve --port 99999"), "--port takes a port");
    let unload = "--unload-after takes a duration such as 30s, 10m or 1h";
    assert_eq!(error("serve --unload-after -1"), unload);
    assert!(error("serve --jsn").starts_with("unknown option --jsn\nusage: wavo serve "));
    assert_eq!(error("serve dir/m.gguf"), "expected a model name, not a path: dir/m.gguf");
    let ane = "--device picks a GPU, and --backend ane runs on the Neural Engine: drop one of them";
    assert_eq!(error("bench gigaam-v3 talk.wav --device 1 --backend ane"), ane);
    assert_eq!(error("serve --backend ane --device nvidia"), ane);
    let device = "--device takes auto, a number from wavo devices or part of a GPU's name";
    assert_eq!(error("talk.wav --device"), device);
    assert_eq!(cli(&["talk.wav", "--device", ""]).unwrap_err().to_string(), device);
    assert_eq!(error("devices x"), "usage: wavo devices");
  }

  #[test]
  fn audio_names() {
    for arg in ["a.wav", "b.MP3", "dir/c.m4a", "d.flac", "e.ogg", "f.opus"] {
      assert!(audio_like(arg), "{arg}");
    }
    for arg in ["whisper-tiny.en", "parakeet-tdt-0.6b-v3", "model.gguf", "wav", "talk"] {
      assert!(!audio_like(arg), "{arg}");
    }
  }
}
