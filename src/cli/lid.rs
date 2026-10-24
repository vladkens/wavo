// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The language of a model's text, for models that don't say which of their languages they heard
//! (Parakeet v3 and ultra): whatlang's trigrams among the model's languages.

/// ISO 639-1 codes, as the models name languages, and whatlang's ISO 639-3 ones: Parakeet's 25
/// but Maltese, which whatlang lacks, and a few more of Whisper's.
const CODES: &str = "en:eng ru:rus uk:ukr be:bel bg:bul de:deu fr:fra es:spa it:ita pt:por pl:pol \
  cs:ces sk:slk sl:slv hr:hrv nl:nld da:dan sv:swe fi:fin et:est lv:lav lt:lit hu:hun ro:ron el:ell \
  he:heb ar:ara zh:cmn ja:jpn ko:kor";

/// Pairs of `CODES`.
fn codes() -> impl Iterator<Item = (&'static str, &'static str)> {
  CODES.split_whitespace().filter_map(|pair| pair.split_once(':'))
}

/// whatlang's language of an ISO 639-1 code.
fn lang(code: &str) -> Option<whatlang::Lang> {
  codes().find(|x| x.0 == code).and_then(|x| whatlang::Lang::from_code(x.1))
}

/// The language of `text` among `codes` (ISO 639-1; whatlang's 69 when none of them is known to
/// it), with whatlang's confidence in [0, 1]; `None` without letters to tell by.
pub fn language(text: &str, codes: &[String]) -> Option<(&'static str, f64)> {
  let list: Vec<whatlang::Lang> = codes.iter().filter_map(|c| lang(c)).collect();
  let detector = match list.is_empty() {
    true => whatlang::Detector::new(),
    false => whatlang::Detector::with_allowlist(list),
  };
  let info = detector.detect(text)?;
  let code = info.lang().code();
  Some((self::codes().find(|x| x.1 == code).map_or(code, |x| x.0), info.confidence()))
}

/// The English name of a language, lowercase, as Whisper's `LANGUAGES` has it (`english`): for
/// whisper-asr-webservice's `detected_language`. The code itself for one whatlang doesn't know.
pub fn name(code: &str) -> String {
  lang(code).map_or(code.to_string(), |l| l.eng_name().to_lowercase())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn languages_of_text() {
    let codes = |c: &[&str]| c.iter().map(|c| c.to_string()).collect::<Vec<_>>();
    let parakeet = codes(&["en", "de", "uk", "ru", "pl", "cs", "sk"]);
    let said = |text| language(text, &parakeet).map(|l| l.0);
    assert_eq!(said("The quick brown fox jumps over the lazy dog."), Some("en"));
    assert_eq!(said("Привіт, як справи? Що ти робиш сьогодні ввечері?"), Some("uk"));
    assert_eq!(said("Привет, как дела? Что ты делаешь сегодня вечером?"), Some("ru"));
    assert_eq!(said("Ich habe heute keine Zeit, aber morgen gerne."), Some("de"));
    assert_eq!(said(""), None);
    // Outside the list: the closest within it.
    assert_eq!(language("Привет, как дела?", &codes(&["uk", "en"])).map(|l| l.0), Some("uk"));
    assert_eq!(name("en"), "english");
    assert_eq!(name("uk"), "ukrainian");
    assert_eq!(name("haw"), "haw");
  }
}
