// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `--model auto`: whisper-tiny names the language, a table names the model for it. The person's
//! own languages, read from the system, are a prior: another language wins only when Whisper is
//! sure of it, since tiny is least reliable on the short clips of dictation.

use std::process::Command;

use wavo::Language;

/// Detects the language with this model.
pub const DETECTOR: &str = "whisper-tiny";

/// The person's own languages and English count this many times their probability, so another
/// language wins only when Whisper is far surer of it. Picked on the reference samples and their
/// 2–3 s cuts (`docs/plans/completed/20261011-cli-ux.md`).
const OWN: f32 = 30.0;

/// The model for a language: GigaAM for Russian, Parakeet for the 25 languages of v3, else
/// Whisper.
pub fn model_for(code: &str) -> &'static str {
  const PARAKEET: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
  ];
  match code {
    "ru" => "gigaam-v3",
    _ if PARAKEET.contains(&code) => "parakeet-v3",
    _ => "whisper-turbo",
  }
}

/// The most likely language, favouring `own` and English. `ranked` is most likely first, as
/// `Model::languages` returns it.
pub fn choose<'a>(ranked: &'a [Language], own: &[String]) -> Option<&'a Language> {
  let i = pick(&ranked.iter().map(|l| (l.code.as_str(), l.probability)).collect::<Vec<_>>(), own);
  ranked.get(i?)
}

/// The index `choose` takes in `(code, probability)` pairs: the highest probability, with the
/// person's own languages and English weighted by [`OWN`]; the first of equals.
fn pick(ranked: &[(&str, f32)], own: &[String]) -> Option<usize> {
  let weight = |&(code, p): &(&str, f32)| {
    if code == "en" || own.iter().any(|o| o == code) { p * OWN } else { p }
  };
  let best = |best: usize, (i, l): (usize, &(&str, f32))| {
    if weight(l) > weight(&ranked[best]) { i } else { best }
  };
  (!ranked.is_empty()).then(|| ranked.iter().enumerate().fold(0, best))
}

/// The person's languages as two-letter codes: macOS preferred languages and keyboard layouts,
/// Windows display languages, else `LANGUAGE`, `LC_ALL`, `LC_MESSAGES` and `LANG`.
pub fn system_languages() -> Vec<String> {
  let run = |cmd: &str, args: &[&str]| {
    let out = Command::new(cmd).args(args).output().ok().filter(|o| o.status.success());
    out.map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default()
  };
  let mut codes = if cfg!(target_os = "macos") {
    let mut codes = mac_languages(&run("defaults", &["read", "-g", "AppleLanguages"]));
    let sources = run("defaults", &["read", "com.apple.HIToolbox", "AppleEnabledInputSources"]);
    codes.extend(keyboard_languages(&sources));
    codes
  } else if cfg!(windows) {
    let key = r"HKCU\Control Panel\International\User Profile";
    windows_languages(&run("reg", &["query", key, "/v", "Languages"]))
  } else {
    Vec::new()
  };
  let vars = ["LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"].map(std::env::var);
  codes.extend(locale_languages(&vars.map(Result::unwrap_or_default)));
  codes
}

/// The language of a locale or tag: `ru` from `ru_RU.UTF-8`, `en-US` or `zh-Hans`; `None` for
/// `C` and `POSIX`.
fn code(tag: &str) -> Option<String> {
  let code = tag.trim().trim_matches('"').split(['_', '-', '.', '@']).next()?.to_lowercase();
  (code.len() >= 2 && code.len() <= 3 && code.bytes().all(|b| b.is_ascii_lowercase()))
    .then_some(code)
}

/// `defaults read -g AppleLanguages`: a list of quoted or bare tags, one per line.
fn mac_languages(out: &str) -> Vec<String> {
  out.lines().filter_map(|l| code(l.trim().trim_end_matches(','))).collect()
}

/// Keyboard layouts and input methods named in `AppleEnabledInputSources`, such as
/// `"KeyboardLayout Name" = RussianWin;` or `com.apple.inputmethod.Kotoeri`.
fn keyboard_languages(out: &str) -> Vec<String> {
  #[rustfmt::skip]
  const NAMES: &[(&str, &str)] = &[
    ("Russian", "ru"), ("Ukrainian", "uk"), ("Belarusian", "be"), ("German", "de"),
    ("French", "fr"), ("Spanish", "es"), ("Italian", "it"), ("Polish", "pl"), ("Czech", "cs"),
    ("Portuguese", "pt"), ("Brazilian", "pt"), ("Dutch", "nl"), ("Swedish", "sv"),
    ("Norwegian", "no"), ("Danish", "da"), ("Finnish", "fi"), ("Greek", "el"), ("Turkish", "tr"),
    ("Hebrew", "he"), ("Arabic", "ar"), ("Hungarian", "hu"), ("Romanian", "ro"),
    ("Bulgarian", "bg"), ("Serbian", "sr"), ("Croatian", "hr"), ("Slovak", "sk"),
    ("Slovenian", "sl"), ("Lithuanian", "lt"), ("Latvian", "lv"), ("Estonian", "et"),
    ("Kazakh", "kk"), ("Georgian", "ka"), ("Armenian", "hy"), ("Vietnamese", "vi"),
    ("Thai", "th"), ("Hindi", "hi"), ("SCIM", "zh"), ("TCIM", "zh"), ("Kotoeri", "ja"),
    ("Japanese", "ja"), ("Korean", "ko"),
  ];
  NAMES.iter().filter(|(name, _)| out.contains(name)).map(|(_, code)| code.to_string()).collect()
}

/// `reg query ... /v Languages`: `Languages    REG_MULTI_SZ    en-US\0ru-RU`.
fn windows_languages(out: &str) -> Vec<String> {
  let line = out.lines().find(|l| l.contains("REG_MULTI_SZ")).unwrap_or_default();
  let tags = line.split_whitespace().last().unwrap_or_default();
  tags.split(r"\0").filter_map(code).collect()
}

/// `LANGUAGE` is a list such as `uk:ru:en`; the others are one locale each.
fn locale_languages(vars: &[String]) -> Vec<String> {
  vars.iter().flat_map(|v| v.split(':')).filter_map(code).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn choose_with_the_prior() {
    let own = ["uk".to_string()];
    assert_eq!(pick(&[("ru", 0.6), ("uk", 0.3), ("en", 0.1)], &own), Some(1));
    assert_eq!(pick(&[("ru", 0.98), ("uk", 0.01)], &own), Some(0)); // sure enough
    assert_eq!(pick(&[("nn", 0.7), ("de", 0.2), ("en", 0.1)], &own), Some(2));
    assert_eq!(pick(&[("ru", 0.6), ("de", 0.4), ("en", 0.001)], &own), Some(0)); // not ours
    assert_eq!(pick(&[("de", 0.5), ("fr", 0.5)], &own), Some(0)); // the first of equals
    assert_eq!(pick(&[], &own), None);
  }

  #[test]
  fn models() {
    assert_eq!(model_for("ru"), "gigaam-v3");
    assert_eq!(model_for("uk"), "parakeet-v3");
    assert_eq!(model_for("en"), "parakeet-v3");
    assert_eq!(model_for("zh"), "whisper-turbo");
  }

  #[test]
  fn system_language_formats() {
    let mac = "(\n    \"en-IE\",\n    \"zh-Hans-CN\",\n    uk\n)\n";
    assert_eq!(mac_languages(mac), ["en", "zh", "uk"]);
    let sources = "(\n {\n \"KeyboardLayout Name\" = ABC;\n },\n {\n \"KeyboardLayout Name\" = \
                   RussianWin;\n },\n {\n \"Bundle ID\" = \"com.apple.inputmethod.Kotoeri\";\n }\n)";
    assert_eq!(keyboard_languages(sources), ["ru", "ja"]);
    let reg = "\r\nHKEY_CURRENT_USER\\Control Panel\\International\\User Profile\r\n    \
               Languages    REG_MULTI_SZ    en-US\\0ru-RU\r\n\r\n";
    assert_eq!(windows_languages(reg), ["en", "ru"]);
    let vars = ["uk:ru:en", "", "C", "de_DE.UTF-8"].map(String::from);
    assert_eq!(locale_languages(&vars), ["uk", "ru", "en", "de"]);
  }
}
