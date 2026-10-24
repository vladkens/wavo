// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The language of a transcript by a few rules (script, letters and common words) for a handful
//! of languages: the hand-rolled alternative to whatlang (`src/cli/lid.rs`) the evaluation
//! measures.

/// Letters, then common words, of the languages the rules know.
const RULES: &[(&str, &str, &str)] = &[
  ("en", "", "the and of to is in that it was for are with as on be this by have from not"),
  (
    "de",
    "äöüß",
    "der die und das ist nicht ein eine zu den von mit sich des auf für im dem auch es",
  ),
  ("fr", "éèêàçœùûîôë", "le la les et des est un une du que en dans pour qui pas sur au il ce se"),
  ("es", "ñáíóúé¿¡", "el la los las y de que en es un una por con para del se no al lo su"),
  ("it", "àèìòùé", "il la di che e è un una per del della non sono in gli le con si da al"),
  ("pl", "ąęłńśźżó", "i w nie się na że jest z do to jak o ale po przez od są tak czy dla"),
  ("cs", "ěřůčšžýáíéúňťď", "a je se na v že to s z do jsou ve by jako ale pro o od tak který"),
  ("sk", "ľĺŕôäčšžýáíéúňťď", "a je sa na v že to s z do sú vo by ako ale pre o od tak ktorý"),
  ("ru", "ыэъё", "и в не на что с по это как он она из для от а к был то его они"),
  ("uk", "іїєґʼ'", "і в не на що з по це як він вона із для від а до був та його вони у"),
];

/// The language of `text` among `allow` (the languages of `RULES` when empty) by its script, the
/// letters only some of them have (worth 3 when only one allowed language has it, else 1) and
/// common words (1 each); `None` without a letter of a known script.
pub fn rules(text: &str, allow: &[&str]) -> Option<&'static str> {
  let text = text.to_lowercase();
  let cyrillic = text.chars().filter(|c| ('\u{400}'..='\u{4ff}').contains(c)).count();
  let latin = text.chars().filter(|c| c.is_alphabetic() && (*c as u32) < 0x250).count();
  if cyrillic + latin == 0 {
    return None;
  }

  let script_ok = |l: &str| (l == "ru" || l == "uk") == (cyrillic > latin);
  let rules: Vec<_> =
    RULES.iter().filter(|r| (allow.is_empty() || allow.contains(&r.0)) && script_ok(r.0)).collect();
  let words: Vec<&str> = text.split(|c: char| !c.is_alphabetic() && c != '\'').collect();
  let score = |r: &(&str, &str, &str)| {
    let letters: f32 = (text.chars().filter(|c| r.1.contains(*c)))
      .map(|c| if rules.iter().filter(|o| o.1.contains(c)).count() == 1 { 3.0 } else { 1.0 })
      .sum();
    let common = words.iter().filter(|w| r.2.split(' ').any(|x| x == **w)).count();
    letters + common as f32
  };
  // The first of the highest.
  let best = rules.iter().map(|r| (r.0, score(r))).reduce(|b, x| if x.1 > b.1 { x } else { b });
  best.map(|b| b.0)
}
