// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The model for the language spoken, among the downloaded ones: `wavo run` without a model and
//! `wavo serve` without one in the request. Detection hears all of the audio up to 15 s, and
//! pieces spread over longer audio, so that its start alone doesn't decide. Whisper's language
//! token (the smallest multilingual Whisper downloaded), with a prior on the languages of the
//! downloaded models, tells Parakeet's 25 languages from the rest; among those 25 the language of
//! Parakeet v3's or ultra's text (`lid`) weighs most, as whatlang names it better than whisper-tiny
//! does past 2 s. Either alone does when the other isn't downloaded.

use std::ops::Range;

use wavo::hub::{self, ModelInfo};
use wavo::{Backend, Model};

use crate::lid;

/// The longest piece of audio Parakeet's text is told from, in milliseconds: 15 s, the Neural
/// Engine's window.
const PIECE_MS: u32 = 15_000;
/// How many pieces spread over longer audio Parakeet's text is told from.
const PIECES: usize = 3;
/// How many times likelier to Whisper a language of the downloaded models is than another.
const PRIOR: f32 = 10.0;
/// How many times likelier to Whisper the language of Parakeet's text is.
const TEXT: f32 = 100.0;
/// Whisper's share of the languages Parakeet doesn't know, after the prior, from which Parakeet's
/// text doesn't count.
const FOREIGN: f32 = 0.3;
/// A model made for a language is taken over another unless its word error rate is past the
/// other's over this.
const OWN: f32 = 0.9;
/// The same for one made for the language or Parakeet v3 and ultra over the rest: they run several
/// times faster than a Whisper as accurate.
const FAST: f32 = 0.9;
/// The model whose word error rate stands in for another's on a language that one wasn't measured
/// on: measured on 82 languages.
const MEASURED: &str = "whisper-turbo";

/// Whether `m` tells many languages apart by Whisper's language tokens.
fn whisper(m: &ModelInfo) -> bool {
  m.family == "Whisper" && m.langs.len() > 2
}

/// Whether `m` is Parakeet v3 or ultra, which tell their 25 languages apart without saying which.
fn parakeet(m: &ModelInfo) -> bool {
  m.family == "Parakeet" && m.langs.len() > 2
}

/// The order of `m` for the language `code`, lower first: its FLEURS word error rate, cut by `OWN`
/// for a model made for the language (GigaAM for Russian, the Hebrew and Ukrainian fine-tunes,
/// Parakeet v2 for English) and by `FAST` for those and Parakeet v3 and ultra. `None` when `m`
/// doesn't transcribe the language; infinite when no rate says how it does.
fn order(m: &ModelInfo, code: &str) -> Option<f32> {
  if m.diarization() || !m.langs.contains(&code) {
    return None;
  }

  let own = m.langs.len() <= 2 && m.langs[0] == code;
  let wer = wer(m, code).unwrap_or(f32::INFINITY);
  let wer = if own { wer * OWN } else { wer };
  Some(if own || parakeet(m) { wer * FAST } else { wer })
}

/// `m`'s word error rate on `code`; when that wasn't measured, `MEASURED`'s times `m`'s mean ratio
/// to it on the languages both were measured on (whisper-large-v3 isn't on Hebrew, whisper-large-v2
/// only on English).
fn wer(m: &ModelInfo, code: &str) -> Option<f32> {
  if let Some(wer) = m.wer(code) {
    return Some(wer);
  }

  let by = hub::lookup(MEASURED).ok()?;
  let ratios: Vec<f32> = m.langs.iter().filter_map(|c| Some(m.wer(c)? / by.wer(c)?)).collect();
  if ratios.is_empty() {
    return None;
  }

  Some(by.wer(code)? * ratios.iter().sum::<f32>() / ratios.len() as f32)
}

/// The model of `models` for the language `code`, the first of them on a tie; `None` when none
/// transcribes it.
fn best(models: &[&'static ModelInfo], code: &str) -> Option<&'static ModelInfo> {
  let ranked = models.iter().filter_map(|&m| Some((order(m, code)?, m)));
  ranked.min_by(|a, b| a.0.total_cmp(&b.0)).map(|x| x.1)
}

/// The downloaded speech-to-text models that `fits` takes, in `hub::MODELS` order.
fn downloaded(fits: impl Fn(&ModelInfo) -> bool) -> Vec<&'static ModelInfo> {
  let found = hub::Hub::new().downloaded().into_iter().map(|(m, _)| m);
  found.filter(|m| !m.diarization() && fits(m)).collect()
}

/// The model of `models` when one serves every language they transcribe, so that there is
/// nothing to detect.
fn only(models: &[&'static ModelInfo]) -> Option<&'static ModelInfo> {
  let mut codes = models.iter().flat_map(|m| m.langs.iter());
  let first = best(models, codes.next()?)?;
  codes.all(|c| best(models, c) == Some(first)).then_some(first)
}

/// The downloaded model for the language spoken in `pcm` (16 kHz mono), or for `language` when
/// given, among those that run on `on` and, to `diarize`, give the time of each word; the first of
/// them when the language isn't told, `None` when none is downloaded. `load` loads a detector and
/// `model` reaches its model.
pub fn choose<T, E: From<wavo::Error>>(
  pcm: &[f32],
  language: Option<&str>,
  on: Backend,
  diarize: bool,
  load: impl FnMut(&'static ModelInfo) -> Result<T, E>,
  model: impl Fn(&T) -> &Model,
) -> Result<Option<&'static ModelInfo>, E> {
  let found = downloaded(|m| {
    (on != Backend::NeuralEngine || m.neural_engine()) && !(diarize && m.family == "Moonshine")
  });
  let Some(&first) = found.first() else { return Ok(None) };
  if let Some(m) = only(&found) {
    return Ok(Some(m));
  }

  let code = match language {
    Some(code) => Some(code.to_string()),
    None => detect(pcm, &found, load, model)?,
  };
  Ok(Some(code.and_then(|c| best(&found, &c)).unwrap_or(first)))
}

/// The language of `pcm`, 16 kHz mono, among what `models` transcribe, told by the detectors
/// among them, which `load` loads and `model` reaches; `None` when nothing told it: no detector
/// downloaded, or no speech.
fn detect<T, E: From<wavo::Error>>(
  pcm: &[f32],
  models: &[&'static ModelInfo],
  mut load: impl FnMut(&'static ModelInfo) -> Result<T, E>,
  model: impl Fn(&T) -> &Model,
) -> Result<Option<String>, E> {
  let (text, token) = (text(models), token(models));
  if pcm.is_empty() {
    return Ok(None);
  }

  let mut weighed = Vec::new();
  if let Some(m) = token {
    let loaded = load(m)?;
    // Up to 5 windows of 30 s spread over the audio, those with speech averaged.
    let ranked = model(&loaded).detect_language(pcm)?.into_iter().map(|l| (l.code, l.probability));
    weighed = weigh(ranked, &languages(models));
    let foreign = |p: &ModelInfo| weighed.iter().filter(|l| !p.langs.contains(&l.0.as_str()));
    if text.is_none_or(|p| foreign(p).map(|l| l.1).sum::<f32>() >= FOREIGN) {
      return Ok(weighed.first().map(|l| l.0.clone()));
    }
  }

  let Some(m) = text else { return Ok(None) };
  let loaded = load(m)?;
  let pieces = spread(pcm, PIECE_MS, PIECES);
  let transcripts = pieces.iter().map(|r| model(&loaded).transcribe(&pcm[r.clone()]));
  let transcripts = transcripts.collect::<wavo::Result<Vec<_>>>()?;
  let said: Vec<&str> = transcripts.iter().map(|t| t.text.as_str()).collect();
  let codes: Vec<String> = m.langs.iter().map(|c| c.to_string()).collect();
  Ok(pick(&weighed, lid::language(&said.join(" "), &codes)))
}

/// Up to `k` (2 or more) of the pieces of `pcm` of up to `max_ms`, cut at pauses, spread from its
/// start to its end.
fn spread(pcm: &[f32], max_ms: u32, k: usize) -> Vec<Range<usize>> {
  let pieces = wavo::split(pcm, max_ms);
  let n = pieces.len();
  if n <= k {
    return pieces;
  }

  (0..k).map(|i| pieces[i * (n - 1) / (k - 1)].clone()).collect()
}

/// The model of `models` whose text tells the language: Parakeet ultra, else v3.
fn text(models: &[&'static ModelInfo]) -> Option<&'static ModelInfo> {
  models.iter().copied().filter(|m| parakeet(m)).min_by_key(|m| m.name != "parakeet-ultra")
}

/// The model of `models` whose language token tells the language: the smallest multilingual
/// Whisper.
fn token(models: &[&'static ModelInfo]) -> Option<&'static ModelInfo> {
  models.iter().copied().filter(|m| whisper(m)).min_by_key(|m| m.size)
}

/// The languages `models` transcribe, but multilingual Whisper's, which take every language.
fn languages(models: &[&'static ModelInfo]) -> Vec<&'static str> {
  let mut codes: Vec<&str> =
    models.iter().filter(|m| !whisper(m)).flat_map(|m| m.langs).copied().collect();
  codes.sort_unstable();
  codes.dedup();
  codes
}

/// Whisper's languages and their probabilities with the prior on `codes`, most likely first,
/// summing to 1.
fn weigh(ranked: impl IntoIterator<Item = (String, f32)>, codes: &[&str]) -> Vec<(String, f32)> {
  let prior = |code: &str| if codes.contains(&code) { PRIOR } else { 1.0 };
  let weighed = ranked.into_iter().map(|(code, p)| (prior(&code) * p, code));
  let mut out: Vec<(String, f32)> = weighed.map(|(p, code)| (code, p)).collect();
  let sum: f32 = out.iter().map(|l| l.1).sum();
  out.iter_mut().for_each(|l| l.1 /= sum.max(f32::MIN_POSITIVE));
  out.sort_by(|a, b| b.1.total_cmp(&a.1));
  out
}

/// The language of Whisper's `weighed` ones with the text's `said` weighing `TEXT` times more,
/// the text's when `lid` is sure of it; the text's alone without Whisper's, Whisper's first
/// without the text's.
fn pick(weighed: &[(String, f32)], said: Option<(&str, f64)>) -> Option<String> {
  let Some((said, confidence)) = said else { return weighed.first().map(|l| l.0.clone()) };
  let p = |code: &str| weighed.iter().find(|l| l.0 == code).map_or(0.0, |l| l.1);
  let top = weighed.first().filter(|l| confidence < 1.0 && l.1 > p(said) * TEXT);
  Some(top.map_or(said, |l| &l.0).to_string())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn models(names: &str) -> Vec<&'static ModelInfo> {
    names.split(' ').map(|n| hub::lookup(n).unwrap()).collect()
  }

  #[test]
  fn the_model_for_a_language() {
    let all = models(
      "parakeet-v3 parakeet-v2 parakeet-ultra parakeet-he parakeet-uk gigaam-v3 gigaam-v3-ctc \
       whisper-turbo whisper-turbo-he whisper-tiny whisper-large-v3 moonshine-tiny",
    );
    let best = |code| best(&all, code).map(|m| m.name);
    assert_eq!(best("ru"), Some("gigaam-v3"), "not whisper-large-v3, though more accurate");
    assert_eq!(best("uk"), Some("parakeet-uk"), "made for it, though ultra measured a bit better");
    assert_eq!(best("he"), Some("whisper-turbo-he"));
    assert_eq!(best("en"), Some("parakeet-v2"));
    assert_eq!(best("de"), Some("parakeet-ultra"));
    assert_eq!(best("ar"), Some("whisper-large-v3"));
    assert_eq!(best("xx"), None);
    let few = models("parakeet-v3 gigaam-v3-ctc moonshine-tiny whisper-tiny");
    assert_eq!(super::best(&few, "ru").unwrap().name, "parakeet-v3", "lowercase gigaam is worse");
    assert_eq!(super::best(&few, "en").unwrap().name, "parakeet-v3");
    assert_eq!(super::best(&few, "he").unwrap().name, "whisper-tiny");
    let best = |names, code| super::best(&models(names), code).unwrap().name;
    assert_eq!(best("whisper-tiny whisper-large-v3", "he"), "whisper-large-v3", "not measured");
    assert_eq!(best("whisper-tiny whisper-large-v2", "de"), "whisper-large-v2", "only on en");
    assert_eq!(best("moonshine-tiny parakeet-uk", "en"), "parakeet-uk", "made for it, but worse");
    assert_eq!(best("whisper-tiny.en whisper-turbo", "en"), "whisper-turbo");
    assert_eq!(only(&models("parakeet-ultra parakeet-v3")).unwrap().name, "parakeet-ultra");
    assert_eq!(only(&models("gigaam-v3")).unwrap().name, "gigaam-v3");
    assert_eq!(only(&models("parakeet-ultra gigaam-v3")), None);
  }

  #[test]
  fn whisper_and_text_together() {
    let lang = |code: &str, probability: f32| (code.to_string(), probability);
    // whisper-tiny on Hebrew: Arabic first; the prior on the downloaded Hebrew model wins.
    let ranked = [lang("ar", 0.6), lang("he", 0.1), lang("ru", 0.01)];
    assert_eq!(weigh(ranked.clone(), &["he", "ru"])[0].0, "he");
    assert_eq!(weigh(ranked, &["ru"])[0].0, "ar");
    let weighed = weigh([lang("ru", 0.8), lang("uk", 0.15), lang("en", 0.05)], &[]);
    assert_eq!(pick(&weighed, Some(("uk", 0.5))).as_deref(), Some("uk"), "the text weighs more");
    assert_eq!(pick(&weighed, Some(("de", 0.5))).as_deref(), Some("ru"), "past Whisper's doubt");
    assert_eq!(pick(&weighed, Some(("de", 1.0))).as_deref(), Some("de"), "sure");
    assert_eq!(pick(&weighed, None).as_deref(), Some("ru"));
    assert_eq!(pick(&[], Some(("de", 0.1))).as_deref(), Some("de"));
    assert_eq!(pick(&[], None), None);
  }

  #[test]
  fn pieces_spread() {
    let s = 16_000;
    let pcm = vec![0.0; s * 100];
    let pieces = spread(&pcm, 15_000, 3);
    assert_eq!(pieces.len(), 3, "{pieces:?}");
    assert_eq!((pieces[0].start, pieces[2].end), (0, pcm.len()), "from start to end: {pieces:?}");
    assert!(pieces[1].start > s * 30 && pieces[1].end < s * 70, "the middle: {pieces:?}");
    assert!(pieces.iter().all(|r| r.len() <= s * 15), "{pieces:?}");
    assert_eq!(spread(&pcm[..s * 15], 15_000, 3).len(), 1, "short audio whole");
  }
}
