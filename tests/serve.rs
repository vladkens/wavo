// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo serve` end to end: started on a free port with a smoke model (`make models`), it must
//! answer curl's uploads with the text `wavo run` prints, for the model it was started with and
//! for one it loads on demand, and Bazarr's `POST /asr` with its subtitles, `POST /detect-language`
//! and `GET /status`.
#![cfg(feature = "cli")]

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};

const WAVO: &str = env!("CARGO_BIN_EXE_wavo");

/// The server process, killed on drop, also when the test fails.
struct Server(Child);

impl Drop for Server {
  fn drop(&mut self) {
    let _ = self.0.kill();
    let _ = self.0.wait();
  }
}

fn output(command: &mut Command) -> String {
  let out = command.output().unwrap();
  let err = String::from_utf8_lossy(&out.stderr);
  assert!(out.status.success(), "{command:?}: {err}");
  String::from_utf8(out.stdout).unwrap()
}

#[test]
fn transcribes_like_run() {
  let jfk = Path::new(env!("CARGO_MANIFEST_DIR")).join("3rd/transcribe.cpp/samples/jfk.wav");
  assert!(jfk.exists(), "{jfk:?} is missing: run make samples");
  let mut child = Command::new(WAVO)
    .args(["serve", "whisper-tiny", "--port", "0", "--api-key", "k"])
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
  let stderr = child.stderr.take().unwrap();
  let _server = Server(child);
  // Ready once the models have loaded: `listening on http://127.0.0.1:PORT`.
  let mut lines = BufReader::new(stderr).lines();
  let url = (lines.by_ref().map(Result::unwrap))
    .find_map(|l| l.strip_prefix("listening on ").map(str::to_string))
    .expect("wavo serve exited before listening");
  std::thread::spawn(move || lines.for_each(drop)); // the request log, so it never blocks

  let send = |path: &str, fields: &[&str]| {
    let mut curl = Command::new("curl");
    curl.args(["-sS", "--fail-with-body", "-H", "Authorization: Bearer k"]);
    curl.args(fields.iter().flat_map(|f| ["-F", f]));
    output(curl.arg(format!("{url}{path}")))
  };
  let post = |fields: &[&str]| send("/v1/audio/transcriptions", fields);
  let file = format!("file=@{}", jfk.display());
  for model in ["whisper-tiny", "moonshine-tiny"] {
    let run = output(Command::new(WAVO).args(["run", model]).arg(&jfk));
    let served = post(&[&file, &format!("model={model}"), "response_format=text"]);
    assert_eq!(served, run, "{model}");
  }

  let json = post(&[&file, "model=whisper-1"]);
  let text = output(Command::new(WAVO).args(["run", "whisper-tiny"]).arg(&jfk));
  assert_eq!(json, format!("{{\"text\": \"{}\"}}\n", text.trim_end()), "the default model");

  // As Bazarr sends it: the WAV's samples without the header.
  let wav = std::fs::read(&jfk).unwrap();
  let data = wav.windows(4).position(|w| w == b"data").unwrap() + 8;
  let pcm = std::env::temp_dir().join(format!("wavo-serve-{}.pcm", std::process::id()));
  std::fs::write(&pcm, &wav[data..]).unwrap();
  let query = "task=transcribe&language=en&output=srt&encode=false&video_file=%2Fa.mkv";
  let srt = send(&format!("/asr?{query}"), &[&format!("audio_file=@{}", pcm.display())]);
  std::fs::remove_file(&pcm).unwrap();
  let run =
    output(Command::new(WAVO).args(["run", "whisper-tiny", "--srt", "--language", "en"]).arg(&jfk));
  assert_eq!(srt, run, "/asr");
  let detected = send("/detect-language", &[&format!("audio_file=@{}", jfk.display())]);
  let english = "{\"detected_language\": \"english\", \"language_code\": \"en\", \"confidence\": ";
  assert!(detected.starts_with(english), "/detect-language: {detected}");
  let mut curl = Command::new("curl");
  let status =
    output(curl.args(["-sS", "-H", "Authorization: Bearer k", &format!("{url}/status")]));
  assert_eq!(status, format!("{{\"version\": \"wavo {}\"}}\n", env!("CARGO_PKG_VERSION")));
}
