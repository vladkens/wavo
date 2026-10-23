// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo serve`: OpenAI's transcription API over HTTP, on `std::net` with a thread per
//! connection. The models given on the command line stay loaded, the first one answering model
//! names wavo doesn't know (OpenAI's `whisper-1`); another downloaded model loads on first use,
//! one at a time. Requests share each model, and their calls take turns on the GPU. `POST /asr`
//! and `GET /status` answer as whisper-asr-webservice does, for Bazarr's subtitles.

use std::collections::HashSet;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use wavo::Backend;
use wavo::hub::{self, Hub, ModelInfo};

use crate::http::{self, Head, Part, Res, fail};
use crate::output::{self, quote};
use crate::{Download, audio, debug, models, split};

const USAGE: &str = "usage: wavo serve [MODEL...] [--host ADDR] [--port PORT] [--api-key KEY] [-y]";
/// Connections served at once; as many more get 503, and past that they are dropped.
const MAX_CONNECTIONS: usize = 16;
/// The largest request body: some 4.5 hours of 16 kHz 16-bit WAV.
const MAX_BODY: usize = 512 << 20;
/// The longest audio, in seconds, whatever its compression.
const MAX_AUDIO: usize = 3 * 3600;
/// How long one read from or write to a client may wait, and the request head may take.
const TIMEOUT: Duration = Duration::from_secs(30);
/// How long a whole request may take to arrive: 512 MB at some 300 KB/s.
const MAX_UPLOAD: Duration = Duration::from_secs(1800);
/// How long a response may take to send: the `verbose_json` of 3 hours, some 10 MB, at 35 KB/s.
const MAX_SEND: Duration = Duration::from_secs(300);
/// The longest form field other than the file, such as `model`, in bytes.
const MAX_FIELD: usize = 256;
/// How long the part of a request nobody reads is drained after the response; also the timeout
/// for a request answered with 503.
const LINGER: Duration = Duration::from_secs(1);
const FORMATS: &[&str] = &["json", "text", "srt", "vtt", "verbose_json"];
/// whisper-asr-webservice's `output` formats, and the `FORMATS` that render them: its JSON, the
/// result of Whisper's `transcribe`, has the text, segments and language as `verbose_json` does.
const ASR_FORMATS: &[(&str, &str)] =
  &[("txt", "text"), ("srt", "srt"), ("vtt", "vtt"), ("json", "verbose_json")];
const JSON: &str = "application/json";
const TEXT: &str = "text/plain; charset=utf-8";

struct Loaded {
  info: &'static ModelInfo,
  model: wavo::Model,
  /// Whether it times tokens, not segments only: see `models::timed`.
  timed: bool,
}

struct Server {
  /// The models given on the command line, loaded for good; the first is the default.
  models: Vec<Arc<Loaded>>,
  /// The other models requests asked for; locked while one loads.
  extra: Mutex<Extra<Loaded>>,
  key: Option<String>,
  /// Open connections.
  open: AtomicUsize,
}

/// A successful response, and a note for the log.
struct Reply {
  content_type: &'static str,
  body: String,
  note: String,
}

pub fn serve(args: &[&str]) -> Result<()> {
  let (mut host, mut port, mut key, mut download) = ("127.0.0.1", 8000, None, Download::Ask);
  let mut names = Vec::new();
  let mut args = args.iter().copied();
  while let Some(arg) = args.next() {
    match arg {
      "--host" => host = args.next().context("--host takes an address, such as 0.0.0.0")?,
      "--port" => port = args.next().and_then(|p| p.parse().ok()).context("--port takes a port")?,
      "--api-key" => key = Some(args.next().context("--api-key takes a key")?.to_string()),
      "-y" | "--yes" => download = Download::Yes,
      _ if arg.starts_with('-') => bail!("unknown option {arg}\n{USAGE}"),
      _ => names.push(crate::named(arg)?),
    }
  }

  let key = key.or_else(|| std::env::var("WAVO_API_KEY").ok()).filter(|k| !k.is_empty());
  if names.is_empty() {
    names.push(hub::lookup(models::DEFAULT)?);
  }

  let mut seen = HashSet::new();
  names.retain(|m| seen.insert(m.name));
  // Before the models load: a port in use fails at once.
  let listener =
    TcpListener::bind((host, port)).with_context(|| format!("listening on {host}:{port}"))?;
  let models = names.iter().map(|m| load(m, &crate::model_path(m.name, download, Backend::Auto)?));
  let models = models.collect::<Result<Vec<_>>>()?;
  let device = models[0].model.device();
  if let Some(warning) = debug::software(device) {
    eprintln!("warning: {warning}");
  }

  let addr = listener.local_addr()?;
  if !addr.ip().is_loopback() && key.is_none() {
    eprintln!("warning: no --api-key: anyone who reaches {addr} can use it");
  }

  let names: Vec<&str> = models.iter().map(|m| m.info.name).collect();
  eprintln!("{} on {device}\nlistening on http://{addr}", names.join(", "));
  let extra = Mutex::new(Extra { last: None, used: Vec::new() });
  let server = Server { models, extra, key, open: AtomicUsize::new(0) };
  std::thread::scope(|s| {
    for stream in listener.incoming() {
      let Ok(stream) = stream else {
        std::thread::sleep(Duration::from_millis(100)); // out of file handles: let some close
        continue;
      };
      let (open, slot) = (server.open.fetch_add(1, Ordering::Relaxed), Slot(&server.open));
      if open >= 2 * MAX_CONNECTIONS {
        continue;
      }

      let server = &server;
      let thread = std::thread::Builder::new();
      if let Err(e) = thread.spawn_scoped(s, move || server.connection(stream, open, slot)) {
        eprintln!("starting a thread: {e}");
      }
    }
  });
  Ok(())
}

fn load(info: &'static ModelInfo, path: &Path) -> Result<Arc<Loaded>> {
  Ok(Arc::new(Loaded {
    info,
    model: crate::load(path, Backend::Auto)?,
    timed: models::timed(path),
  }))
}

/// A client's socket whose reads and writes wait up to `TIMEOUT` each and fail past `until`, also
/// when the client sends or reads a byte at a time.
struct Timed<'a> {
  stream: &'a TcpStream,
  until: Instant,
}

impl Timed<'_> {
  /// The time a read or write may wait.
  fn left(&self) -> io::Result<Option<Duration>> {
    let left = self.until.saturating_duration_since(Instant::now()).min(TIMEOUT);
    if left.is_zero() { Err(io::ErrorKind::TimedOut.into()) } else { Ok(Some(left)) }
  }
}

impl Read for Timed<'_> {
  fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
    self.stream.set_read_timeout(self.left()?)?;
    let mut stream = self.stream;
    stream.read(buf)
  }
}

impl Write for Timed<'_> {
  fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
    self.stream.set_write_timeout(self.left()?)?;
    let mut stream = self.stream;
    stream.write(buf)
  }

  fn flush(&mut self) -> io::Result<()> {
    Ok(())
  }
}

/// An open connection, counted until dropped.
struct Slot<'a>(&'a AtomicUsize);

impl Drop for Slot<'_> {
  fn drop(&mut self) {
    self.0.fetch_sub(1, Ordering::Relaxed);
  }
}

impl Server {
  /// Serves one request, with `open` others open (503 from `MAX_CONNECTIONS` on), and logs it:
  /// method, path, status, time and a note or the error. Then drains what the client still sends
  /// for a while, so that closing the socket doesn't reset the connection before the client
  /// reads the response.
  fn connection(&self, stream: TcpStream, open: usize, _slot: Slot) {
    let (t, busy) = (Instant::now(), open >= MAX_CONNECTIONS);
    let limit = if busy { LINGER } else { TIMEOUT };
    // The head must come within `limit`, the body within `MAX_UPLOAD`.
    let mut r = BufReader::with_capacity(64 << 10, Timed { stream: &stream, until: t + limit });
    let (method, path, reply) = match http::head(&mut r) {
      Ok(None) => return,
      Ok(Some(head)) if busy => {
        let busy = format!("busy with {MAX_CONNECTIONS} requests; retry later");
        (head.method, head.path, Err(fail(503, busy)))
      }
      Ok(Some(head)) => {
        r.get_mut().until = t + MAX_UPLOAD;
        let mut w = Timed { stream: &stream, until: t + MAX_UPLOAD }; // for 100 Continue
        let route = AssertUnwindSafe(|| self.route(&head, &mut r, &mut w));
        let reply = catch_unwind(route).unwrap_or_else(|_| Err(fail(500, "wavo panicked")));
        (head.method, head.path, reply)
      }
      Err(e) => ("-".into(), "-".into(), Err(e)),
    };
    let (status, note) = match &reply {
      Ok(reply) => (200, reply.note.clone()),
      Err(e) => (e.status, e.message.clone()),
    };
    let until = Instant::now() + if busy { LINGER } else { MAX_SEND };
    let _ = send(Timed { stream: &stream, until }, reply); // fails when the client is gone
    let sep = if note.is_empty() { "" } else { ": " };
    eprintln!("{method} {path} {status} {} ms{sep}{note}", t.elapsed().as_millis());
    let _ = stream.shutdown(Shutdown::Write);
    r.get_mut().until = Instant::now() + LINGER;
    let _ = io::copy(&mut r, &mut io::sink());
  }

  fn route(&self, head: &Head, r: &mut impl BufRead, w: &mut impl Write) -> Res<Reply> {
    if head.path != "/health" && !authorized(self.key.as_deref(), head.header("authorization")) {
      return Err(fail(401, "a missing or wrong API key: send Authorization: Bearer KEY"));
    }

    match (head.method.as_str(), head.path.as_str()) {
      ("GET", "/health") => Ok(self.health()),
      ("GET", "/status") => Ok(status()),
      ("GET", "/v1/models") => Ok(self.list()),
      ("POST", "/v1/audio/transcriptions") => self.transcribe(head, r, w),
      ("POST", "/asr") => self.asr(head, r, w),
      (method, path) => Err(fail(404, format!("no such endpoint: {method} {path}"))),
    }
  }

  /// The GPU, the loaded models (not one still loading) and any warning about the GPU.
  fn health(&self) -> Reply {
    let device = self.models[0].model.device();
    let extra = self.extra.try_lock().map(|x| x.used.iter().filter_map(Weak::upgrade).collect());
    let extra: Vec<_> = extra.unwrap_or_default();
    let names: Vec<String> = self.models.iter().chain(&extra).map(|m| quote(m.info.name)).collect();
    let warning = debug::software(device).map(|w| quote(&w)).unwrap_or_default();
    let (device, names) = (quote(&device.to_string()), names.join(", "));
    let fields = format!("\"device\": {device}, \"models\": [{names}], \"warnings\": [{warning}]");
    let body = format!("{{\"status\": \"ok\", {fields}}}\n");
    Reply { content_type: JSON, body, note: String::new() }
  }

  /// The downloaded models in OpenAI's list format, the ones given on the command line first.
  fn list(&self) -> Reply {
    let mut found: Vec<_> = crate::downloaded(&Hub::new()).into_iter().map(|(m, _)| m).collect();
    let given = |m: &ModelInfo| self.models.iter().position(|x| x.info.name == m.name);
    found.sort_by_key(|m| given(m).unwrap_or(usize::MAX));
    let model = |m: &ModelInfo| {
      let (id, owner) = (quote(m.name), quote(m.owner));
      format!("{{\"id\": {id}, \"object\": \"model\", \"created\": 0, \"owned_by\": {owner}}}")
    };
    let data: Vec<String> = found.into_iter().map(model).collect();
    let body = format!("{{\"object\": \"list\", \"data\": [{}]}}\n", data.join(", "));
    Reply { content_type: JSON, body, note: String::new() }
  }

  /// `POST /v1/audio/transcriptions` with OpenAI's form: `file`, `model`, `language` and
  /// `response_format`; `prompt`, `temperature` and the rest are ignored.
  fn transcribe(&self, head: &Head, r: &mut impl BufRead, w: &mut impl Write) -> Res<Reply> {
    let body = http::body(r, w, head, MAX_BODY)?;
    let parts = http::multipart(head.header("content-type").unwrap_or_default(), &body)?;
    let field = |name| field(&body, &parts, name);
    if field("stream")?.is_some_and(|s| s == "true") {
      return Err(fail(400, "stream=true is not supported: the response holds the whole text"));
    }

    let format = field("response_format")?.filter(|f| !f.is_empty()).unwrap_or("json".into());
    if !FORMATS.contains(&format.as_str()) {
      let formats = FORMATS.join(", ");
      return Err(fail(400, format!("response_format {format:?} is not one of {formats}")));
    }

    let mut options = wavo::Options::default();
    options.language = field("language")?.filter(|l| !l.is_empty());
    let model = self.pick(&field("model")?.unwrap_or_default())?;
    let pcm = decode(body, &parts, "file", false)?;
    reply(&model, &pcm, &options, &format, false)
  }

  /// `POST /asr` of whisper-asr-webservice, as Bazarr calls it: the form field `audio_file`, and
  /// in the query `language`, `output` (txt, srt, vtt or json) and `encode` (`false` for raw
  /// 16 kHz mono 16-bit PCM). The first model answers; `task=translate` fails, the rest is ignored.
  fn asr(&self, head: &Head, r: &mut impl BufRead, w: &mut impl Write) -> Res<Reply> {
    let query = |key| http::query(&head.query, key).filter(|v| !v.is_empty());
    if query("task").is_some_and(|t| t != "transcribe") {
      return Err(fail(400, "only task=transcribe is supported: wavo doesn't translate"));
    }

    let output = query("output").unwrap_or("txt");
    let Some(&(_, format)) = ASR_FORMATS.iter().find(|f| f.0 == output) else {
      return Err(fail(400, format!("output {output:?} is not one of txt, srt, vtt, json")));
    };
    let model = &self.models[0];
    if let Some(code) = query("language").filter(|c| !model.info.langs.contains(c)) {
      let name = model.info.name;
      return Err(fail(
        400,
        format!("{name} doesn't transcribe {code:?}: serve a model that does"),
      ));
    }

    let mut options = wavo::Options::default();
    options.language = query("language").map(str::to_string);
    let body = http::body(r, w, head, MAX_BODY)?;
    let parts = http::multipart(head.header("content-type").unwrap_or_default(), &body)?;
    let pcm = decode(body, &parts, "audio_file", query("encode") == Some("false"))?;
    reply(model, &pcm, &options, format, true)
  }

  /// The model a request's `model` names: one given on the command line, else another loaded
  /// one, else a downloaded one, loaded. 404 for a model not downloaded.
  fn pick(&self, name: &str) -> Res<Arc<Loaded>> {
    let Some(m) = wanted(name) else { return Ok(self.models[0].clone()) };
    if let Some(loaded) = self.models.iter().find(|x| x.info.name == m.name) {
      return Ok(loaded.clone());
    }

    let not_found = || fail(404, format!("{0} is not downloaded; run: wavo pull {0}", m.name));
    let fetch = || {
      let path = Hub::new().find(m.name).ok_or_else(not_found)?;
      load(m, &path).map_err(|e| fail(500, format!("{e:#}")))
    };
    let mut extra = self.extra.lock().unwrap_or_else(PoisonError::into_inner);
    extra.get(|x| x.info.name == m.name, fetch)
  }
}

/// The models loaded on request: the last one asked for stays loaded, the others only while
/// requests use them.
struct Extra<T> {
  last: Option<Arc<T>>,
  used: Vec<Weak<T>>,
}

impl<T> Extra<T> {
  /// The model `is` picks among the loaded ones, else the one `load` loads, which becomes the
  /// last; the previous last one's memory goes once its requests finish.
  fn get(&mut self, is: impl Fn(&T) -> bool, load: impl FnOnce() -> Res<Arc<T>>) -> Res<Arc<T>> {
    self.used.retain(|x| x.strong_count() > 0);
    let loaded = match self.used.iter().filter_map(Weak::upgrade).find(|x| is(x)) {
      Some(loaded) => loaded,
      None => {
        self.last = None;
        let loaded = load()?;
        self.used.push(Arc::downgrade(&loaded));
        loaded
      }
    };
    self.last = Some(loaded.clone());
    Ok(loaded)
  }
}

/// The wavo model a request's `model` names exactly, by its short or full name; `None` for the
/// default: an empty or unknown name, such as OpenAI's `whisper-1`.
fn wanted(name: &str) -> Option<&'static ModelInfo> {
  hub::MODELS.iter().find(|m| m.name == name || m.full_name == name)
}

/// The audio of the form field `name`: a file, or with `raw` 16 kHz mono 16-bit PCM. Takes the
/// body to decode it without a copy.
fn decode(mut body: Vec<u8>, parts: &[Part], name: &str, raw: bool) -> Res<Vec<f32>> {
  let file = parts.iter().find(|p| p.name == name);
  let file = file.ok_or_else(|| fail(400, format!("send the audio in the form field {name}")))?;
  if raw {
    let (pcm, _) = body[file.data.clone()].as_chunks::<2>();
    if pcm.len() > MAX_AUDIO * 16_000 {
      return Err(fail(400, format!("the audio is over {} hours", MAX_AUDIO / 3600)));
    }

    return Ok(pcm.iter().map(|&s| i16::from_le_bytes(s) as f32 / 32768.0).collect());
  }

  let name = file.filename.as_deref().map(Path::new);
  let ext = name.and_then(Path::extension).and_then(|e| e.to_str()).map(str::to_string);
  body.truncate(file.data.end);
  body.drain(..file.data.start);
  audio::decode_bytes(body, ext.as_deref(), MAX_AUDIO).map_err(|e| {
    match e.is::<audio::Unsupported>() {
      true => fail(415, "the audio is not wav, mp3, m4a, flac or ogg (Vorbis): convert it first"),
      false => fail(400, format!("decoding the audio: {e:#}")),
    }
  })
}

/// Transcribes `pcm` with `m` into `format`, one of `FORMATS`. With `lenient`, a language the
/// model lists but can't be told (Parakeet v3 detects its own) is left to it: Bazarr names every
/// audio track's language.
fn reply(m: &Loaded, pcm: &[f32], o: &wavo::Options, format: &str, lenient: bool) -> Res<Reply> {
  let run = |o: &wavo::Options| split::transcribe(&m.model, pcm, None, o, |_, _| ());
  let transcript = match run(o) {
    Err(wavo::Error::Language(_)) if lenient => run(&wavo::Options::default()),
    transcript => transcript,
  };
  let (t, starts) = transcript.map_err(|e| match e {
    wavo::Error::Language(_) => fail(400, e.to_string()),
    e => fail(500, format!("transcribing: {e}")),
  })?;

  let ms = (pcm.len() / 16) as u32;
  let cues = || output::cues(&t.tokens, &starts, ms, m.timed);
  let (content_type, body) = match format {
    "json" => (JSON, format!("{{\"text\": {}}}\n", quote(&t.text))),
    "text" => (TEXT, format!("{}\n", t.text)),
    "srt" => (TEXT, output::srt(&cues())),
    "vtt" => ("text/vtt; charset=utf-8", output::vtt(&cues())),
    _ => {
      // The language asked for, else the model's only one.
      let langs = m.info.langs;
      let language = o.language.as_deref().or((langs.len() == 1).then(|| langs[0]));
      (JSON, output::verbose_json(&t, &cues(), language.unwrap_or_default(), ms) + "\n")
    }
  };
  let note = format!("{}, {:.1} s of audio", m.info.name, ms as f64 / 1000.0);
  Ok(Reply { content_type, body, note })
}

/// wavo's version, for Bazarr's Test Connection (whisper-asr-webservice's `GET /status`).
fn status() -> Reply {
  let body = format!("{{\"version\": \"wavo {}\"}}\n", env!("CARGO_PKG_VERSION"));
  Reply { content_type: JSON, body, note: String::new() }
}

/// The text of the form field `name`, trimmed; 400 when it is over `MAX_FIELD` bytes.
fn field(body: &[u8], parts: &[Part], name: &str) -> Res<Option<String>> {
  let Some(part) = parts.iter().find(|p| p.name == name) else { return Ok(None) };
  if part.data.len() > MAX_FIELD {
    return Err(fail(400, format!("the form field {name} is over {MAX_FIELD} bytes")));
  }

  Ok(Some(String::from_utf8_lossy(&body[part.data.clone()]).trim().to_string()))
}

/// Whether the `Authorization` header holds `key` as a Bearer token; without a key, any request
/// is. The comparison takes as long wherever the first difference is.
fn authorized(key: Option<&str>, header: Option<&str>) -> bool {
  let Some(key) = key else { return true };
  let token = header.and_then(|h| h.trim().split_once(' '));
  let token = token.filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"));
  let token = token.map_or("", |(_, token)| token.trim());
  let diff = token.bytes().zip(key.bytes()).fold(0, |acc, (a, b)| acc | (a ^ b));
  token.len() == key.len() && diff == 0
}

/// Writes `reply`, or the error in OpenAI's format: `{"error": {"message", "type"}}`.
fn send(mut w: impl Write, reply: Res<Reply>) -> io::Result<()> {
  let e = match reply {
    Ok(reply) => return http::respond(&mut w, 200, reply.content_type, reply.body.as_bytes()),
    Err(e) => e,
  };
  let kind = if e.status >= 500 { "server_error" } else { "invalid_request_error" };
  let message = quote(&e.message);
  let body = format!("{{\"error\": {{\"message\": {message}, \"type\": \"{kind}\"}}}}\n");
  http::respond(&mut w, e.status, JSON, body.as_bytes())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn model_names() {
    for name in ["", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "Whisper-Tiny"] {
      assert_eq!(wanted(name), None, "{name}");
    }
    assert_eq!(wanted("gigaam-v3").unwrap().name, "gigaam-v3");
    assert_eq!(wanted("gigaam-v3-e2e-rnnt").unwrap().name, "gigaam-v3", "a full name");
    assert_eq!(wanted("whisper-tiny").unwrap().name, "whisper-tiny");
    assert_eq!(wanted("whisper-tin"), None, "no typo suggestions");
  }

  #[test]
  fn form_fields() {
    let body = format!("whisper-1{}", "x".repeat(257));
    let parts = |len: usize| [Part { name: "model".into(), filename: None, data: 0..len }];
    assert_eq!(field(body.as_bytes(), &parts(9), "model"), Ok(Some("whisper-1".into())));
    assert_eq!(field(body.as_bytes(), &parts(9), "language"), Ok(None));
    assert!(field(body.as_bytes(), &parts(256), "model").is_ok());
    let e = fail(400, "the form field model is over 256 bytes");
    assert_eq!(field(body.as_bytes(), &parts(257), "model"), Err(e));
  }

  /// Bazarr's `encode=false` audio: 16 kHz mono 16-bit little-endian PCM, without a header.
  #[test]
  fn raw_pcm() {
    let body = b"xx\x00\x80\xff\x7f\x01\x00\x07--".to_vec();
    let parts = [Part { name: "audio_file".into(), filename: None, data: 2..9 }];
    let pcm = vec![-1.0, 32767.0 / 32768.0, 1.0 / 32768.0];
    assert_eq!(decode(body.clone(), &parts, "audio_file", true), Ok(pcm), "an odd byte dropped");
    let e = fail(400, "send the audio in the form field file");
    assert_eq!(decode(body, &parts, "file", true), Err(e));
    assert!(ASR_FORMATS.iter().all(|f| FORMATS.contains(&f.1)));
  }

  /// A, B and A again while the first A is still in use share that A.
  #[test]
  fn models_in_use_are_shared() {
    let mut extra = Extra { last: None, used: Vec::new() };
    let mut loads = 0;
    let mut get = |extra: &mut Extra<&'static str>, name: &'static str| {
      let load = || {
        loads += 1;
        Ok(Arc::new(name))
      };
      extra.get(|&x| x == name, load).unwrap()
    };
    let a = get(&mut extra, "a");
    let b = get(&mut extra, "b");
    assert!(Arc::ptr_eq(&a, &get(&mut extra, "a")), "a still in use");
    assert!(Arc::ptr_eq(&a, &get(&mut extra, "a")), "the last one");
    drop((a, b));
    assert_eq!(Arc::strong_count(&get(&mut extra, "a")), 2, "kept as the last");
    get(&mut extra, "b");
    assert_eq!(extra.used.len(), 2);
    get(&mut extra, "c");
    assert_eq!((extra.used.len(), loads), (2, 4), "a was freed; a, b, b and c loaded");
  }

  /// A client that reads a little at a time can't keep a response going past its deadline.
  #[test]
  fn writes_stop_at_the_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reading = done.clone();
    let reader = std::thread::spawn(move || {
      let mut buf = [0; 4096];
      while !reading.load(Ordering::Relaxed) && client.read(&mut buf).is_ok_and(|n| n > 0) {
        std::thread::sleep(Duration::from_millis(10));
      }
    });
    let (stream, t) = (listener.accept().unwrap().0, Instant::now());
    let mut w = Timed { stream: &stream, until: t + Duration::from_millis(300) };
    let e = w.write_all(&vec![0; 64 << 20]).unwrap_err();
    assert!(matches!(e.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock), "{e}");
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    done.store(true, Ordering::Relaxed);
    reader.join().unwrap();
  }

  #[test]
  fn api_keys() {
    assert!(authorized(None, None));
    assert!(authorized(None, Some("Bearer anything")));
    assert!(authorized(Some("k3y"), Some("Bearer k3y")));
    assert!(authorized(Some("k3y"), Some("bearer  k3y ")));
    for header in [None, Some(""), Some("Bearer"), Some("Bearer k3"), Some("Bearer k3yy")] {
      assert!(!authorized(Some("k3y"), header), "{header:?}");
    }
    assert!(!authorized(Some("k3y"), Some("Basic k3y")));
    assert!(!authorized(Some("k3y"), Some("k3y")));
  }
}
