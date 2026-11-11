// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo serve` end to end: started on a free port with a smoke model (`make models`), it must
//! answer curl's uploads with the text `wavo run` prints, for the model it was started with and
//! for one it loads on demand, and Bazarr's `POST /asr` with its subtitles, `POST /detect-language`
//! and `GET /status`; with `--unload-after 0` it frees its model and an on-demand one after their
//! requests, and loads its model again for `/asr`, `/detect-language` and transcriptions.
#![cfg(feature = "cli")]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

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

fn jfk() -> PathBuf {
  let jfk = Path::new(env!("CARGO_MANIFEST_DIR")).join("3rd/transcribe.cpp/samples/jfk.wav");
  assert!(jfk.exists(), "{jfk:?} is missing: run make samples");
  jfk
}

/// `wavo serve` with `args` on a free port, and its URL once the models have loaded.
fn serve(args: &[&str]) -> (Server, String) {
  let mut child = Command::new(WAVO)
    .args(["serve", "--port", "0"])
    .args(args)
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
  let stderr = child.stderr.take().unwrap();
  let server = Server(child);
  // Ready once the models have loaded: `listening on http://127.0.0.1:PORT`.
  let mut lines = BufReader::new(stderr).lines();
  let url = (lines.by_ref().map(Result::unwrap))
    .find_map(|l| l.strip_prefix("listening on ").map(str::to_string))
    .expect("wavo serve exited before listening");
  std::thread::spawn(move || lines.for_each(drop)); // the request log, so it never blocks
  (server, url)
}

#[test]
fn transcribes_like_run() {
  let jfk = jfk();
  let (_server, url) = serve(&["whisper-tiny", "--api-key", "k"]);

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
  let audio_file = format!("audio_file=@{}", pcm.display());
  let srt = send(&format!("/asr?{query}"), &[&audio_file]);
  let detected = send("/detect-language?encode=false&video_file=%2Fa.mkv", &[&audio_file]);
  std::fs::remove_file(&pcm).unwrap();
  let english = "{\"detected_language\": \"english\", \"language_code\": \"en\", \"confidence\": ";
  assert!(detected.starts_with(english), "/detect-language: {detected}");
  let run =
    output(Command::new(WAVO).args(["run", "whisper-tiny", "--srt", "--language", "en"]).arg(&jfk));
  assert_eq!(srt, run, "/asr");
  let mut curl = Command::new("curl");
  let status =
    output(curl.args(["-sS", "-H", "Authorization: Bearer k", &format!("{url}/status")]));
  assert_eq!(status, format!("{{\"version\": \"wavo {}\"}}\n", env!("CARGO_PKG_VERSION")));
}

#[test]
fn frees_idle_models() {
  let (_server, url) = serve(&["whisper-tiny", "--unload-after", "0"]);
  let post = |path: &str, fields: &[&str]| {
    let mut curl = Command::new("curl");
    curl.args(["-sS", "--fail-with-body"]).args(fields.iter().flat_map(|f| ["-F", f]));
    output(curl.arg(format!("{url}{path}")))
  };
  let health = || output(Command::new("curl").args(["-sS", &format!("{url}/health")]));
  let freed = || {
    (0..50).any(|_| {
      std::thread::sleep(Duration::from_millis(100));
      health().contains("\"models\": []")
    })
  };
  let (file, audio_file) =
    (format!("file=@{}", jfk().display()), format!("audio_file=@{}", jfk().display()));

  let first = post("/v1/audio/transcriptions", &[&file]);
  assert!(freed(), "still loaded 5 s after the request: {}", health());
  let asr = post("/asr?output=txt", &[&audio_file]);
  assert_eq!(first, format!("{{\"text\": \"{}\"}}\n", asr.trim_end()), "/asr loaded it again");
  assert!(freed(), "still loaded 5 s after /asr: {}", health());
  let detected = post("/detect-language", &[&audio_file]);
  assert!(detected.contains("\"language_code\": \"en\""), "{detected}");
  assert!(freed(), "still loaded 5 s after /detect-language: {}", health());
  let moonshine = post("/v1/audio/transcriptions", &[&file, "model=moonshine-tiny"]);
  assert!(moonshine.contains("country"), "{moonshine}");
  assert!(freed(), "moonshine-tiny still loaded 5 s after its request: {}", health());
  assert_eq!(post("/v1/audio/transcriptions", &[&file]), first, "loaded again");
}
