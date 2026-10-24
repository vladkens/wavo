// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The language ID evaluation of the langid prototype, on the clips `scripts/lid-data.sh` puts in
//! `3rd/lid/` (FLEURS test clips, 12 per language). Prints Markdown tables.
//!
//! - `langid-eval audio MODEL`: `Model::detect_language` on 2, 3, 5 and 10 s cuts and whole clips,
//!   without and with an allowlist; accuracy, the commonest confusions and the time per call.
//! - `langid-eval text MODEL`: text language ID (whatlang, and the rules in `examples/langid.rs`)
//!   on the transcripts of a Parakeet model over the same cuts.
//! - `langid-eval episodes`: synthetic episodes of same-language blocks; spans, boundaries and
//!   false switches for whisper-tiny per piece (policies simulated on the pieces' probabilities,
//!   and the library's `sticky_language`) and for Parakeet per piece with text language ID.
//! - `langid-eval switch WAV...`: each downloaded model's transcript of each file, for
//!   code-switched speech.

#[path = "langid/text.rs"]
mod text;

use std::collections::HashMap;
use std::error::Error;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use wavo::{Model, Options};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const LANGS: &[&str] =
  &["en", "ru", "uk", "de", "fr", "es", "it", "pl", "cs", "sk", "he", "ar", "zh"];
/// Those of `LANGS` that Parakeet v3 and ultra transcribe.
const PARAKEET: &[&str] = &["en", "ru", "uk", "de", "fr", "es", "it", "pl", "cs", "sk"];
const CUTS: &[u32] = &[2, 3, 5, 10, 0];
const SR: usize = 16_000;

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().skip(1).collect();
  match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
    ["audio", model] => audio(model),
    ["text", model] => text_lid(model),
    ["episodes"] => episodes(),
    ["switch", ref wavs @ ..] => switch(wavs),
    _ => Err("usage: langid-eval audio MODEL | text MODEL | episodes | switch WAV...".into()),
  }
}

fn root() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join("3rd/lid")
}

fn load(name: &str) -> Result<Model> {
  let path = wavo::hub::Hub::new().find(name).ok_or(format!("{name} is not downloaded"))?;
  Ok(Model::load_with(path, wavo::Backend::Gpu)?)
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
  let mut wav = hound::WavReader::open(path)?;
  assert_eq!((wav.spec().channels, wav.spec().sample_rate), (1, 16000), "{path:?}");
  Ok(wav.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect())
}

/// The clips of a language: audio and transcription.
fn clips(lang: &str) -> Result<Vec<(Vec<f32>, String)>> {
  let dir = root().join(lang);
  let list = std::fs::read_to_string(dir.join("list.tsv"))?;
  let clip = |line: &str| -> Result<_> {
    let (file, text) = line.split_once('\t').ok_or("list.tsv")?;
    Ok((read_wav(&dir.join(file))?, text.to_string()))
  };
  list.lines().map(clip).collect()
}

/// The cuts of `secs` seconds of a clip, from 0.5 s on (FLEURS clips start with a pause), one
/// after another; the whole clip for 0.
fn cuts(pcm: &[f32], secs: u32) -> Vec<&[f32]> {
  if secs == 0 {
    return vec![pcm];
  }

  let n = secs as usize * SR;
  let mut out = Vec::new();
  let mut at = SR / 2;
  while at + n <= pcm.len() && out.len() < 3 {
    out.push(&pcm[at..at + n]);
    at += n;
  }

  out
}

fn cut_name(secs: u32) -> String {
  if secs == 0 { "clip".into() } else { format!("{secs} s") }
}

fn median(v: &mut [f64]) -> f64 {
  v.sort_by(f64::total_cmp);
  if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// Accuracy as `87% (52/60)`.
fn pct(ok: usize, n: usize) -> String {
  if n == 0 { "-".into() } else { format!("{:.0}% ({ok}/{n})", ok as f64 * 100.0 / n as f64) }
}

fn confusions(c: &HashMap<(String, String), usize>, top: usize) -> String {
  let mut v: Vec<_> = c.iter().collect();
  v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
  let v: Vec<String> = v.iter().take(top).map(|((t, p), n)| format!("{t}→{p} {n}")).collect();
  v.join(", ")
}

fn options(languages: &[&str]) -> Options {
  let mut o = Options::default();
  o.languages = languages.iter().map(|l| l.to_string()).collect();
  o
}

/// The allowlists: none, the 13 test languages, and the clip's own with English (a person who
/// names the two languages they listen to).
fn allowlists(lang: &str) -> [(&'static str, Vec<&str>); 3] {
  let pair = if lang == "en" { vec!["en", "ru"] } else { vec![lang, "en"] };
  [("all", vec![]), ("13", LANGS.to_vec()), ("own+en", pair)]
}

fn audio(name: &str) -> Result<()> {
  let model = load(name)?;
  // (allowlist, cut) -> (right, all); confusions per cut without an allowlist; times per cut.
  let mut acc: HashMap<(&str, u32), (usize, usize)> = HashMap::new();
  let mut conf: HashMap<u32, HashMap<(String, String), usize>> = HashMap::new();
  let mut times: HashMap<u32, Vec<f64>> = HashMap::new();
  let mut per_lang: HashMap<(&str, u32), (usize, usize)> = HashMap::new();
  model.detect_language(&[0.0; SR], &Options::default())?; // warm up
  for &lang in LANGS {
    for (pcm, _) in clips(lang)? {
      for &secs in CUTS {
        for cut in cuts(&pcm, secs) {
          for (list, langs) in allowlists(lang) {
            let t = Instant::now();
            let ranked = model.detect_language(cut, &options(&langs))?;
            if list == "all" {
              times.entry(secs).or_default().push(t.elapsed().as_secs_f64() * 1e3);
            }

            let got = ranked.first().map_or("-", |l| l.code.as_str()).to_string();
            let e = acc.entry((list, secs)).or_default();
            e.0 += usize::from(got == lang);
            e.1 += 1;
            if list == "all" {
              let e = per_lang.entry((lang, secs)).or_default();
              e.0 += usize::from(got == lang);
              e.1 += 1;
              if got != lang {
                *conf.entry(secs).or_default().entry((lang.to_string(), got)).or_default() += 1;
              }
            }
          }
        }
      }
    }
  }

  println!("### {name}: `detect_language` accuracy\n");
  let head: Vec<String> = CUTS.iter().map(|&s| cut_name(s)).collect();
  println!("| allowlist | {} |\n|---|{}", head.join(" | "), "---|".repeat(CUTS.len()));
  for list in ["all", "13", "own+en"] {
    let row: Vec<String> =
      CUTS.iter().map(|&s| acc.get(&(list, s)).map_or("-".into(), |a| pct(a.0, a.1))).collect();
    println!("| {list} | {} |", row.join(" | "));
  }

  let row: Vec<String> =
    CUTS.iter().map(|&s| format!("{:.1}", median(times.get_mut(&s).unwrap()))).collect();
  println!("| ms per call (median) | {} |", row.join(" | "));
  println!("\nPer language, no allowlist:\n");
  println!("| lang | {} |\n|---|{}", head.join(" | "), "---|".repeat(CUTS.len()));
  for &lang in LANGS {
    let row: Vec<String> = CUTS
      .iter()
      .map(|&s| per_lang.get(&(lang, s)).map_or("-".into(), |a| format!("{}/{}", a.0, a.1)))
      .collect();
    println!("| {lang} | {} |", row.join(" | "));
  }

  println!("\nCommonest confusions, no allowlist:\n");
  for &secs in CUTS {
    println!("- {}: {}", cut_name(secs), confusions(&conf.remove(&secs).unwrap_or_default(), 6));
  }

  Ok(())
}

fn text_lid(name: &str) -> Result<()> {
  let model = load(name)?;
  let parakeet: Vec<&str> = model.languages().iter().map(String::as_str).collect();
  // Detectors: whatlang without and with an allowlist, the rules with the test languages.
  let methods: [(&str, Vec<&str>, bool); 4] = [
    ("whatlang, all 69", vec![], true),
    ("whatlang, Parakeet's 25", parakeet.clone(), true),
    ("whatlang, the 10", PARAKEET.to_vec(), true),
    ("rules, the 10", PARAKEET.to_vec(), false),
  ];
  let mut acc: HashMap<(usize, u32), (usize, usize)> = HashMap::new();
  let mut by_len: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
  let mut conf: HashMap<usize, HashMap<(String, String), usize>> = HashMap::new();
  let mut times: Vec<(usize, f64)> = Vec::new();
  let mut asr: Vec<f64> = Vec::new();
  let buckets = [20, 50, 100, usize::MAX];
  for &lang in PARAKEET {
    for (pcm, _) in clips(lang)? {
      for &secs in CUTS {
        for cut in cuts(&pcm, secs) {
          let t = Instant::now();
          let transcript = model.transcribe(cut)?;
          asr.push(t.elapsed().as_secs_f64() * 1e3);
          let chars = transcript.text.chars().count();
          let bucket = buckets.iter().position(|&b| chars <= b).unwrap();
          for (m, (_, langs, whatlang)) in methods.iter().enumerate() {
            let t = Instant::now();
            let got = match whatlang {
              true => text::whatlang(&transcript.text, langs),
              false => text::rules(&transcript.text, langs),
            };
            times.push((m, t.elapsed().as_secs_f64() * 1e6));
            let got = got.unwrap_or("-").to_string();
            let ok = usize::from(got == lang);
            let e = acc.entry((m, secs)).or_default();
            (e.0, e.1) = (e.0 + ok, e.1 + 1);
            let e = by_len.entry((m, bucket)).or_default();
            (e.0, e.1) = (e.0 + ok, e.1 + 1);
            if ok == 0 {
              *conf.entry(m).or_default().entry((lang.to_string(), got)).or_default() += 1;
            }
          }
        }
      }
    }
  }

  println!("### Text language ID on {name} transcripts (the 10 test languages it covers)\n");
  let head: Vec<String> = CUTS.iter().map(|&s| cut_name(s)).collect();
  println!(
    "| method | {} | µs per call |\n|---|{}---|",
    head.join(" | "),
    "---|".repeat(CUTS.len())
  );
  for (m, (label, ..)) in methods.iter().enumerate() {
    let row: Vec<String> =
      CUTS.iter().map(|&s| acc.get(&(m, s)).map_or("-".into(), |a| pct(a.0, a.1))).collect();
    let mut t: Vec<f64> = times.iter().filter(|x| x.0 == m).map(|x| x.1).collect();
    println!("| {label} | {} | {:.0} |", row.join(" | "), median(&mut t));
  }

  println!("\nBy transcript length:\n");
  println!("| method | ≤20 chars | 21–50 | 51–100 | >100 |\n|---|---|---|---|---|");
  for (m, (label, ..)) in methods.iter().enumerate() {
    let row: Vec<String> =
      (0..4).map(|b| by_len.get(&(m, b)).map_or("-".into(), |a| pct(a.0, a.1))).collect();
    println!("| {label} | {} |", row.join(" | "));
  }

  println!("\nCommonest confusions:\n");
  for (m, (label, ..)) in methods.iter().enumerate() {
    println!("- {label}: {}", confusions(&conf.remove(&m).unwrap_or_default(), 6));
  }

  println!("\n{name} transcription per cut: median {:.0} ms", median(&mut asr));
  Ok(())
}

/// A synthetic episode: blocks of a language's clips back to back (`-` for silence), each up to
/// the given seconds.
struct Episode {
  name: &'static str,
  blocks: &'static [(&'static str, u32)],
}

const EPISODES: &[Episode] = &[
  Episode { name: "silence→en→es", blocks: &[("-", 8), ("en", 90), ("es", 90)] },
  Episode { name: "ru→uk", blocks: &[("ru", 120), ("uk", 120)] },
  Episode { name: "de→en→de", blocks: &[("de", 60), ("en", 60), ("de", 60)] },
  Episode { name: "en→ru→en→ru", blocks: &[("en", 60), ("ru", 60), ("en", 40), ("ru", 60)] },
];

/// The audio of an episode and its truth: (language, start, end) in samples. A block takes its
/// language's clips from `used` on, whole, until the next one would pass its length.
fn build(e: &Episode, used: &mut HashMap<&str, usize>) -> Result<(Vec<f32>, Truth)> {
  let (mut pcm, mut truth) = (Vec::new(), Vec::new());
  for &(lang, secs) in e.blocks {
    let start = pcm.len();
    let want = secs as usize * SR;
    if lang == "-" {
      pcm.resize(start + want, 0.0);
    } else {
      let all = clips(lang)?;
      let next = used.entry(lang).or_default();
      while pcm.len() - start < want {
        let (clip, _) = &all[*next % all.len()];
        *next += 1;
        if pcm.len() - start + clip.len() > want && pcm.len() > start {
          break;
        }

        pcm.extend_from_slice(clip);
      }
    }

    truth.push((lang.to_string(), start, pcm.len()));
  }

  Ok((pcm, truth))
}

/// Per block: language, start and end in samples.
type Truth = Vec<(String, usize, usize)>;

/// A piece: start and end in samples, the language decided, whether it has text.
#[derive(Clone)]
struct Span {
  start: usize,
  end: usize,
  lang: Option<String>,
}

/// Span accuracy over the speech (the share of 100 ms steps in a language block whose span has
/// that language), boundary errors (s) of the true changes found within 30 s, missed changes and
/// false switches.
fn score(spans: &[Span], truth: &[(String, usize, usize)]) -> (f64, Vec<f64>, usize, usize) {
  let step = SR / 10;
  let (mut ok, mut all) = (0, 0);
  for (lang, start, end) in truth.iter().filter(|t| t.0 != "-") {
    for at in (*start..*end).step_by(step) {
      all += 1;
      let span = spans.iter().find(|s| s.start <= at && at < s.end);
      ok += usize::from(span.and_then(|s| s.lang.as_deref()) == Some(lang));
    }
  }

  // Switches: where the language changes between spans with one.
  let mut switches: Vec<(usize, String)> = Vec::new();
  let mut last: Option<&str> = None;
  for s in spans {
    if let Some(l) = s.lang.as_deref() {
      if last.is_some_and(|x| x != l) {
        switches.push((s.start, l.to_string()));
      }

      last = Some(l);
    }
  }

  let speech: Vec<_> = truth.iter().filter(|t| t.0 != "-").collect();
  let (mut errors, mut missed, mut matched) = (Vec::new(), 0, vec![false; switches.len()]);
  for w in speech.windows(2) {
    let (b, lang) = (w[1].1, &w[1].0);
    let near = (0..switches.len())
      .filter(|&i| !matched[i] && &switches[i].1 == lang && switches[i].0.abs_diff(b) <= 30 * SR)
      .min_by_key(|&i| switches[i].0.abs_diff(b));
    match near {
      Some(i) => {
        matched[i] = true;
        errors.push(switches[i].0.abs_diff(b) as f64 / SR as f64);
      }
      None => missed += 1,
    }
  }

  (ok as f64 / all.max(1) as f64, errors, missed, matched.iter().filter(|m| !**m).count())
}

/// A language choice per piece from its probabilities (allowlist order) and whether it has
/// speech: `p` and `k` as `SWITCH_P` and `SWITCH_K`, `twice` for two such pieces in a row.
fn policy(
  pieces: &[(Vec<(String, f32)>, bool)],
  p: f32,
  k: f32,
  twice: bool,
) -> Vec<Option<String>> {
  let mut current: Option<String> = None;
  let mut pending: Option<String> = None;
  let mut out = Vec::new();
  for (probs, speech) in pieces {
    let best = &probs[0];
    let prob = |c: &str| probs.iter().find(|x| x.0 == c).map_or(0.0, |x| x.1);
    let pick = match &current {
      None => {
        if *speech {
          current = Some(best.0.clone());
        }

        best.0.clone()
      }
      Some(c) if *speech && best.0 != *c && best.1 >= p && best.1 >= k * prob(c) => {
        if !twice || pending.as_deref() == Some(&best.0) {
          current = Some(best.0.clone());
          pending = None;
          best.0.clone()
        } else {
          pending = Some(best.0.clone());
          c.clone()
        }
      }
      Some(c) => {
        if *speech {
          pending = None;
        }

        c.clone()
      }
    };
    out.push(Some(pick));
  }

  out
}

fn episodes() -> Result<()> {
  let mut used = HashMap::new();
  let built: Vec<_> = EPISODES.iter().map(|e| build(e, &mut used)).collect::<Result<_>>()?;
  for (e, (pcm, truth)) in EPISODES.iter().zip(&built) {
    let blocks: Vec<String> =
      truth.iter().map(|t| format!("{} {:.0} s", t.0, (t.2 - t.1) as f64 / SR as f64)).collect();
    println!("- {}: {}", e.name, blocks.join(", "));
    let out = root().join(format!("episode-{}.wav", e.name.replace('→', "-")));
    let spec = hound::WavSpec {
      channels: 1,
      sample_rate: 16000,
      bits_per_sample: 16,
      sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&out, spec)?;
    for &s in pcm {
      w.write_sample((s * 32767.0).round().clamp(-32768.0, 32767.0) as i16)?;
    }

    w.finalize()?;
  }

  type Row = (String, f64, Vec<f64>, usize, usize);
  let mut rows: Vec<(String, Vec<Row>)> = Vec::new();
  let mut add = |method: &str, episode: &str, spans: &[Span], truth: &[(String, usize, usize)]| {
    let (acc, errors, missed, false_switches) = score(spans, truth);
    let row = (episode.to_string(), acc, errors, missed, false_switches);
    match rows.iter_mut().find(|r| r.0 == method) {
      Some(r) => r.1.push(row),
      None => rows.push((method.to_string(), vec![row])),
    }
  };

  // Whisper-tiny: pieces of up to 30 s; their probabilities and whether they have text.
  let tiny = load("whisper-tiny")?;
  for (e, (pcm, truth)) in EPISODES.iter().zip(&built) {
    for (label, langs) in [("all", vec![]), ("13", LANGS.to_vec())] {
      let ranges = wavo::split(pcm, 30_000);
      let mut pieces = Vec::new();
      for r in &ranges {
        let ranked = tiny.detect_language(&pcm[r.clone()], &options(&langs))?;
        let text = tiny.transcribe_with(&pcm[r.clone()], &options(&langs))?;
        let probs: Vec<(String, f32)> =
          ranked.into_iter().map(|l| (l.code, l.probability)).collect();
        pieces.push((probs, !text.text.is_empty()));
      }

      let spans = |choice: Vec<Option<String>>| -> Vec<Span> {
        (ranges.iter().zip(choice))
          .map(|(r, lang)| Span { start: r.start, end: r.end, lang })
          .collect()
      };
      for (name, p, k, twice) in [
        ("independent", 0.0, 0.0, false),
        ("sticky 0.5/2×", 0.5, 2.0, false),
        ("sticky 0.7/4×", 0.7, 4.0, false),
        ("sticky 0.9/10×", 0.9, 10.0, false),
        ("sticky 0.7/4×, twice", 0.7, 4.0, true),
      ] {
        let choice = match name {
          "independent" => pieces.iter().map(|p| Some(p.0[0].0.clone())).collect(),
          _ => policy(&pieces, p, k, twice),
        };
        add(&format!("whisper-tiny {name}, allowlist {label}"), e.name, &spans(choice), truth);
      }

      // The library's own: transcribe_long with sticky_language, which must match 0.7/4×.
      let mut o = options(&langs);
      o.sticky_language = true;
      let mut got = Vec::new();
      tiny.transcribe_long(pcm, &o, |piece| {
        got.push(piece.transcript.language.clone());
        ControlFlow::Continue(())
      })?;
      let sim = policy(&pieces, 0.7, 4.0, false);
      if got != sim {
        eprintln!("{} {label}: sticky_language {got:?}\n  simulated {sim:?}", e.name);
      }
    }
  }

  drop(tiny);
  for name in ["parakeet-ultra", "parakeet-v3"] {
    let model = load(name)?;
    let parakeet: Vec<&str> = model.languages().iter().map(String::as_str).collect();
    for (e, (pcm, truth)) in EPISODES.iter().zip(&built) {
      for window in [30_000, 60_000] {
        let mut o = Options::default();
        o.window_ms = Some(window);
        let mut pieces: Vec<(usize, String)> = Vec::new();
        model.transcribe_long(pcm, &o, |p| {
          pieces.push((p.start_ms as usize * 16, p.transcript.text.clone()));
          ControlFlow::Continue(())
        })?;
        for (label, langs) in [("25", parakeet.clone()), ("10", PARAKEET.to_vec())] {
          let mut last = None;
          let spans: Vec<Span> = (0..pieces.len())
            .map(|i| {
              let end = pieces.get(i + 1).map_or(pcm.len(), |p| p.0);
              // A piece without text keeps the language before it.
              let lang = text::whatlang(&pieces[i].1, &langs).map(str::to_string).or(last.clone());
              last.clone_from(&lang);
              Span { start: pieces[i].0, end, lang }
            })
            .collect();
          let method = format!("{name} {} s + whatlang {label}", window / 1000);
          add(&method, e.name, &spans, truth);
        }
      }
    }
  }

  println!(
    "\n| method | per episode | span accuracy | boundary error (s) | missed | false switches |"
  );
  println!("|---|---|---|---|---|---|");
  for (method, per) in &rows {
    let acc: Vec<String> = per.iter().map(|r| format!("{:.0}", r.1 * 100.0)).collect();
    let errors: Vec<f64> = per.iter().flat_map(|r| r.2.clone()).collect();
    let mean = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    let max = errors.iter().copied().fold(0.0, f64::max);
    let missed: usize = per.iter().map(|r| r.3).sum();
    let false_switches: usize = per.iter().map(|r| r.4).sum();
    let all = per.iter().map(|r| r.1).sum::<f64>() / per.len() as f64 * 100.0;
    println!(
      "| {method} | {} | {all:.0}% | mean {mean:.1}, max {max:.1} | {missed} | {false_switches} |",
      acc.join(" / ")
    );
  }

  println!(
    "\n(per-episode span accuracy in the order: {})",
    EPISODES.iter().map(|e| e.name).collect::<Vec<_>>().join(", ")
  );
  Ok(())
}

fn switch(wavs: &[&str]) -> Result<()> {
  let downloaded = wavo::hub::Hub::new().downloaded();
  for (info, path) in downloaded {
    if !["gigaam-v3", "parakeet-v3", "parakeet-ultra", "whisper-tiny"].contains(&info.name) {
      continue;
    }

    let model = Model::load_with(path, wavo::Backend::Gpu)?;
    for wav in wavs {
      let pcm = read_wav(Path::new(wav))?;
      let mut o = Options::default();
      if info.name == "whisper-tiny" {
        o.language = Some("ru".into());
      }

      let got = model.transcribe_with(&pcm, &o)?;
      let name = Path::new(wav).file_stem().unwrap().to_string_lossy();
      println!("{}\t{name}\t{}", info.name, got.text);
    }
  }

  Ok(())
}
