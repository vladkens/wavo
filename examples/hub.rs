// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Lists the models, downloads one on a thread while the main thread shows its progress, then
//! transcribes a 16 kHz mono WAV of any length with it, piece by piece:
//!
//! `cargo run --release --example hub --no-default-features --features hub -- MODEL AUDIO.wav`
//!
//! `--stop-at MB` cancels the download there; the next run resumes it.

use std::ops::ControlFlow;
use std::sync::mpsc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let (name, wav, stop) = match &args[..] {
    [name, wav] => (name.clone(), wav, u64::MAX),
    [name, wav, flag, mb] if flag == "--stop-at" => {
      (name.clone(), wav, mb.parse::<u64>()? * 1_000_000)
    }
    _ => return Err("usage: hub MODEL AUDIO.wav [--stop-at MB]".into()),
  };
  for m in wavo::hub::MODELS {
    println!("{:<18} {:>5} MB  {}", m.name, m.size / 1_000_000, m.languages);
  }

  // The download thread sends whole percents; the channel closes when the pull returns.
  let (tx, rx) = mpsc::channel();
  let pull = std::thread::spawn(move || {
    let mut last = None;
    wavo::hub::Hub::new().pull(&name, |p| {
      if let Some(step) = p.preparing {
        match p.retrying {
          Some(e) => eprintln!("\npreparing failed, it runs on the GPU: {e}"),
          None => eprintln!("\npreparing for the Neural Engine: {step:?}"),
        }

        return ControlFlow::Continue(());
      }

      if let Some(e) = p.retrying {
        eprintln!("\nretrying after: {e}");
      }
      let pct = p.done * 100 / p.total;
      if last != Some(pct) {
        last = Some(pct);
        let _ = tx.send(pct);
      }
      if p.done < stop { ControlFlow::Continue(()) } else { ControlFlow::Break(()) }
    })
  });
  for pct in rx {
    eprint!("\rdownloading: {pct:3}%");
  }
  eprintln!();
  let path = match pull.join().expect("the download thread panicked") {
    Err(wavo::Error::Cancelled) => {
      eprintln!("cancelled; run again to resume");
      return Ok(());
    }
    path => path?,
  };
  println!("{}", path.display());

  let mut reader = hound::WavReader::open(wav)?;
  let spec = reader.spec();
  if (spec.channels, spec.sample_rate, spec.bits_per_sample) != (1, 16_000, 16) {
    return Err(format!("{wav}: not 16 kHz mono 16-bit").into());
  }
  let pcm = reader.samples::<i16>().map(|s| Ok(s? as f32 / 32768.0)).collect::<Result<Vec<_>>>()?;

  // Audio of any length, split at pauses, with the pieces' progress.
  let model = wavo::Model::load(&path)?;
  let transcript = model.transcribe_long(&pcm, &wavo::Options::default(), |piece| {
    eprint!("\rtranscribing: {}/{}", piece.index + 1, piece.count);
    ControlFlow::Continue(())
  })?;
  eprintln!();

  println!("{}", transcript.text);
  let first = transcript.tokens.first().map_or(0, |t| t.start_ms);
  println!("{} tokens, the first at {first} ms", transcript.tokens.len());
  Ok(())
}
