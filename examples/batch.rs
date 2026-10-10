// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Many recordings through one model load, and two such runs compared (`make compare`).
//!
//! `batch MODEL.gguf LIST` transcribes each 16 kHz mono WAV listed in LIST (one path per line) and
//! prints JSON lines shaped like `transcribe-cli -m MODEL.gguf --batch LIST --batch-jsonl`: a
//! header with `load_ms`, then per file `file`, `text`, `audio_ms` and `ms`, the wall time of
//! `Model::transcribe`.
//!
//! `batch compare A.jsonl B.jsonl` joins two such files by `file` and prints speed, then exact-text
//! agreement and the word error rate of A against B (lowercase, punctuation dropped) by duration,
//! then the files whose text differs. A transcribe.cpp file's time per call is its `mel_ms +
//! encode_ms + decode_ms`, a few ms below the call's wall time. `/usr/bin/time -l` output saved
//! next to a JSONL as `NAME.time` adds the process's wall time and peak footprint.

use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().skip(1).collect();
  match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
    ["compare", a, b] => compare(a, b),
    [model, list] => run(model, list),
    _ => Err("usage: batch MODEL.gguf LIST | batch compare A.jsonl B.jsonl".into()),
  }
}

fn ms(t: Instant) -> f64 {
  t.elapsed().as_secs_f64() * 1e3
}

fn run(model: &str, list: &str) -> Result<()> {
  let t = Instant::now();
  let model = wavo::Model::load(model)?;
  println!("{{\"type\":\"batch_header\",\"load_ms\":{:.1}}}", ms(t));
  let mut failed = 0;
  for file in std::fs::read_to_string(list)?.lines().map(str::trim).filter(|l| !l.is_empty()) {
    let mut wav = hound::WavReader::open(file).map_err(|e| format!("{file}: {e}"))?;
    if (wav.spec().channels, wav.spec().sample_rate) != (1, 16000) {
      return Err(format!("{file}: not 16 kHz mono").into());
    }
    let pcm = wav.samples::<i16>().map(|s| Ok(s? as f32 / 32768.0)).collect::<Result<Vec<_>>>()?;
    let t = Instant::now();
    let result = model.transcribe(&pcm);
    let elapsed = ms(t);
    let (text, error) = match result {
      Ok(transcript) => (transcript.text, String::new()),
      Err(e) => {
        failed += 1;
        (String::new(), format!(",\"error\":{}", quote(&e.to_string())))
      }
    };
    let (file, text, audio_ms) = (quote(file), quote(&text), pcm.len() / 16);
    println!(
      "{{\"file\":{file},\"text\":{text},\"audio_ms\":{audio_ms},\"ms\":{elapsed:.1}{error}}}"
    );
  }
  if failed > 0 {
    return Err(format!("{failed} files failed (\"error\" in their rows)").into());
  }
  Ok(())
}

fn quote(s: &str) -> String {
  let mut out = String::from("\"");
  for c in s.chars() {
    match c {
      '"' | '\\' => out.extend(['\\', c]),
      '\n' => out += "\\n",
      c if c < ' ' => out += &format!("\\u{:04x}", c as u32),
      c => out.push(c),
    }
  }
  out + "\""
}

/// The raw value of the first `"key":` in a flat JSON line (the top-level `text` comes before
/// transcribe.cpp's `segments`).
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
  Some(&line[line.find(&format!("\"{key}\":"))? + key.len() + 3..])
}

fn string(line: &str, key: &str) -> Option<String> {
  let mut chars = field(line, key)?.strip_prefix('"')?.chars();
  let mut out = String::new();
  loop {
    match chars.next()? {
      '"' => return Some(out),
      '\\' => match chars.next()? {
        'n' => out.push('\n'),
        'r' => out.push('\r'),
        't' => out.push('\t'),
        'u' => {
          let hex: String = chars.by_ref().take(4).collect();
          out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
        }
        c => out.push(c),
      },
      c => out.push(c),
    }
  }
}

fn number(line: &str, key: &str) -> Option<f64> {
  let value = field(line, key)?;
  value[..value.find([',', '}']).unwrap_or(value.len())].parse().ok()
}

struct Run {
  name: String,
  load_ms: Option<f64>,
  files: Vec<String>,
  rows: HashMap<String, (String, f64, bool)>, // text, ms, error
  time: Option<(f64, f64)>,                   // process wall s, peak footprint MiB
}

fn read_run(path: &str) -> Result<Run> {
  let stem = path.strip_suffix(".jsonl").unwrap_or(path);
  let name = stem.rsplit('/').next().unwrap_or(stem).to_string();
  let mut run = Run { name, load_ms: None, files: Vec::new(), rows: HashMap::new(), time: None };
  for line in std::fs::read_to_string(path)?.lines().filter(|l| l.starts_with('{')) {
    if line.contains("\"batch_header\"") {
      run.load_ms = number(line, "load_ms");
      continue;
    }
    let file = string(line, "file").ok_or_else(|| format!("{path}: no file in {line}"))?;
    let ms = number(line, "ms").unwrap_or_else(|| {
      ["mel_ms", "encode_ms", "decode_ms"].iter().filter_map(|k| number(line, k)).sum()
    });
    let text = string(line, "text").unwrap_or_default();
    run.rows.insert(file.clone(), (text, ms, line.contains("\"error\":")));
    run.files.push(file);
  }
  // `/usr/bin/time -l`: "  12.34 real ..." and "  123456789  peak memory footprint".
  if let Ok(time) = std::fs::read_to_string(format!("{stem}.time")) {
    let first = |end: &str| {
      let line = time.lines().find(|l| l.contains(end))?;
      line.split_whitespace().next()?.parse::<f64>().ok()
    };
    if let (Some(real), Some(peak)) = (first(" real "), first("peak memory footprint")) {
      run.time = Some((real, peak / 1048576.0));
    }
  }
  Ok(run)
}

fn words(text: &str) -> Vec<String> {
  let text: String = text.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
  text.to_lowercase().split_whitespace().map(String::from).collect()
}

/// Word-level edit distance (substitutions, insertions and deletions).
fn edits(a: &[String], b: &[String]) -> usize {
  let mut row: Vec<usize> = (0..=b.len()).collect();
  for (i, wa) in a.iter().enumerate() {
    let mut diag = row[0];
    row[0] = i + 1;
    for (j, wb) in b.iter().enumerate() {
      let next = (diag + usize::from(wa != wb)).min(row[j] + 1).min(row[j + 1] + 1);
      (diag, row[j + 1]) = (row[j + 1], next);
    }
  }
  row[b.len()]
}

/// The `p` quantile of sorted values, nearest rank.
fn quantile(sorted: &[f64], p: f64) -> f64 {
  sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Upper ends of the duration buckets, in seconds.
const BUCKETS: [(f64, &str); 6] = [
  (2.0, "< 2 s"),
  (5.0, "2-5 s"),
  (10.0, "5-10 s"),
  (25.0, "10-25 s"),
  (60.0, "25-60 s"),
  (f64::INFINITY, ">= 60 s"),
];

#[derive(Default)]
struct Tally {
  files: usize,
  audio_s: f64,
  exact: usize,
  edits: usize,
  words: usize,
  ms: [f64; 2],
  rtf: [Vec<f64>; 2],
}

fn compare(a: &str, b: &str) -> Result<()> {
  let runs = [read_run(a)?, read_run(b)?];
  let mut buckets: Vec<Tally> = BUCKETS.iter().map(|_| Tally::default()).collect();
  let (mut all, mut differ, mut empty, mut errors) = (Tally::default(), Vec::new(), [0; 2], [0; 2]);
  for file in runs[0].files.iter().filter(|f| runs[1].rows.contains_key(*f)) {
    let wav = hound::WavReader::open(file).map_err(|e| format!("{file}: {e}"))?;
    let audio_s = wav.duration() as f64 / wav.spec().sample_rate as f64;
    let rows = [&runs[0].rows[file], &runs[1].rows[file]];
    let (wa, wb) = (words(&rows[0].0), words(&rows[1].0));
    let n = edits(&wa, &wb);
    if rows[0].0 != rows[1].0 {
      differ.push((file, audio_s, n, wb.len()));
    }
    let bucket = BUCKETS.iter().position(|&(end, _)| audio_s < end).unwrap();
    for tally in [&mut buckets[bucket], &mut all] {
      tally.files += 1;
      tally.audio_s += audio_s;
      tally.exact += usize::from(rows[0].0 == rows[1].0);
      (tally.edits, tally.words) = (tally.edits + n, tally.words + wb.len());
      for (k, row) in rows.iter().enumerate() {
        tally.ms[k] += row.1;
        tally.rtf[k].push(row.1 / 1e3 / audio_s);
      }
    }
    for (k, row) in rows.iter().enumerate() {
      empty[k] += usize::from(row.0.trim().is_empty());
      errors[k] += usize::from(row.2);
    }
  }
  for tally in buckets.iter_mut().chain([&mut all]) {
    tally.rtf.iter_mut().for_each(|v| v.sort_by(f64::total_cmp));
  }
  if all.files == 0 {
    return Err("no file in both runs".into());
  }
  let missing = runs.iter().map(|r| r.files.len() - all.files).collect::<Vec<_>>();
  println!(
    "{} files in both ({:.2} h of audio), {missing:?} only in one",
    all.files,
    all.audio_s / 3600.0
  );
  println!("\n| | {} | {} |\n|---|---|---|", runs[0].name, runs[1].name);
  let row =
    |label: &str, f: &dyn Fn(usize) -> String| println!("| {label} | {} | {} |", f(0), f(1));
  let opt = |v: Option<f64>, digits: usize| v.map_or("-".into(), |v| format!("{v:.digits$}"));
  row("load, ms", &|k| opt(runs[k].load_ms, 0));
  row("compute, s", &|k| format!("{:.1}", all.ms[k] / 1e3));
  row("RTF median", &|k| format!("{:.4}", quantile(&all.rtf[k], 0.5)));
  row("RTF p90", &|k| format!("{:.4}", quantile(&all.rtf[k], 0.9)));
  row("RTF max", &|k| format!("{:.4}", quantile(&all.rtf[k], 1.0)));
  row("process wall, s", &|k| opt(runs[k].time.map(|t| t.0), 1));
  row("peak footprint, MiB", &|k| opt(runs[k].time.map(|t| t.1), 0));
  row("empty texts", &|k| empty[k].to_string());
  row("errors", &|k| errors[k].to_string());
  let (na, nb) = (&runs[0].name, &runs[1].name);
  println!(
    "\n| Duration | Files | Audio, h | Same text | WER | Compute {na} / {nb}, s | RTF median {na} / {nb} |"
  );
  println!("|---|---|---|---|---|---|---|");
  for (t, label) in buckets.iter().zip(BUCKETS.iter().map(|b| b.1)).chain([(&all, "all")]) {
    if t.files == 0 {
      continue;
    }
    println!(
      "| {label} | {} | {:.2} | {:.2}% | {:.3}% | {:.1} / {:.1} | {:.4} / {:.4} |",
      t.files,
      t.audio_s / 3600.0,
      100.0 * t.exact as f64 / t.files as f64,
      100.0 * t.edits as f64 / t.words.max(1) as f64,
      t.ms[0] / 1e3,
      t.ms[1] / 1e3,
      quantile(&t.rtf[0], 0.5),
      quantile(&t.rtf[1], 0.5),
    );
  }
  println!("\n{} files differ (file, audio s, word edits / words):", differ.len());
  for (file, audio_s, n, words) in differ {
    println!("{file}\t{audio_s:.1}\t{n}/{words}");
  }
  Ok(())
}
