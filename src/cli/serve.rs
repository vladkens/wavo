// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo serve`: OpenAI's transcription API over HTTP, on `std::net` with a thread per
//! connection. The models given on the command line stay loaded, the first one answering model
//! names wavo doesn't know (OpenAI's `whisper-1`); another downloaded model loads on first use,
//! one at a time. Requests share each model, and their calls take turns on the GPU.

use std::collections::HashSet;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use wavo::hub::{self, Hub, ModelInfo};

use crate::http::{self, Head, Res, fail};
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
/// How long the part of a request nobody reads is drained after the response; also the timeout
/// for a request answered with 503.
const LINGER: Duration = Duration::from_secs(1);
const FORMATS: &[&str] = &["json", "text", "srt", "vtt", "verbose_json"];
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
  /// The other model a request asked for last, locked while it loads.
  extra: Mutex<Option<Arc<Loaded>>>,
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
  let models = names.iter().map(|m| load(m, &crate::model_path(m.name, download)?));
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
  let server = Server { models, extra: Mutex::new(None), key, open: AtomicUsize::new(0) };
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
  Ok(Arc::new(Loaded { info, model: crate::load(path)?, timed: models::timed(path) }))
}

/// A client's socket whose reads wait up to `TIMEOUT` each and fail past `until`, also when the
/// client sends a byte at a time.
struct Timed<'a> {
  stream: &'a TcpStream,
  until: Instant,
}

impl Read for Timed<'_> {
  fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
    let left = self.until.saturating_duration_since(Instant::now()).min(TIMEOUT);
    if left.is_zero() {
      return Err(io::ErrorKind::TimedOut.into());
    }

    self.stream.set_read_timeout(Some(left))?;
    let mut stream = self.stream;
    stream.read(buf)
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
    let _ = stream.set_write_timeout(Some(limit));
    // The head must come within `limit`, the body only without a pause of `TIMEOUT`.
    let mut r = BufReader::with_capacity(64 << 10, Timed { stream: &stream, until: t + limit });
    let (method, path, reply) = match http::head(&mut r) {
      Ok(None) => return,
      Ok(Some(head)) if busy => {
        let busy = format!("busy with {MAX_CONNECTIONS} requests; retry later");
        (head.method, head.path, Err(fail(503, busy)))
      }
      Ok(Some(head)) => {
        r.get_mut().until = t + MAX_UPLOAD;
        let route = AssertUnwindSafe(|| self.route(&head, &mut r, &mut &stream));
        let reply = catch_unwind(route).unwrap_or_else(|_| Err(fail(500, "wavo panicked")));
        (head.method, head.path, reply)
      }
      Err(e) => ("-".into(), "-".into(), Err(e)),
    };
    let (status, note) = match &reply {
      Ok(reply) => (200, reply.note.clone()),
      Err(e) => (e.status, e.message.clone()),
    };
    let _ = send(&stream, reply); // fails when the client is gone
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
      ("GET", "/v1/models") => Ok(self.list()),
      ("POST", "/v1/audio/transcriptions") => self.transcribe(head, r, w),
      (method, path) => Err(fail(404, format!("no such endpoint: {method} {path}"))),
    }
  }

  /// The GPU, the loaded models (not one still loading) and any warning about the GPU.
  fn health(&self) -> Reply {
    let device = self.models[0].model.device();
    let extra = self.extra.try_lock().ok().and_then(|x| x.clone());
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
    let mut body = http::body(r, w, head, MAX_BODY)?;
    let parts = http::multipart(head.header("content-type").unwrap_or_default(), &body)?;
    let part = |name| parts.iter().find(|p| p.name == name);
    let field =
      |name| part(name).map(|p| String::from_utf8_lossy(&body[p.data.clone()]).trim().to_string());
    if field("stream").is_some_and(|s| s == "true") {
      return Err(fail(400, "stream=true is not supported: the response holds the whole text"));
    }

    let format = field("response_format").filter(|f| !f.is_empty()).unwrap_or("json".into());
    if !FORMATS.contains(&format.as_str()) {
      let formats = FORMATS.join(", ");
      return Err(fail(400, format!("response_format {format:?} is not one of {formats}")));
    }

    let mut options = wavo::Options::default();
    options.language = field("language").filter(|l| !l.is_empty());
    let model = self.pick(&field("model").unwrap_or_default())?;
    let file = part("file").ok_or_else(|| fail(400, "send the audio in the form field file"))?;
    let name = file.filename.as_deref().map(Path::new);
    let ext = name.and_then(Path::extension).and_then(|e| e.to_str()).map(str::to_string);
    // The file's bytes alone, without a copy.
    let data = file.data.clone();
    body.truncate(data.end);
    body.drain(..data.start);
    let pcm =
      audio::decode_bytes(body, ext.as_deref(), MAX_AUDIO).map_err(|e| match e
        .is::<audio::Unsupported>()
      {
        true => fail(415, "the audio is not wav, mp3, m4a, flac or ogg (Vorbis): convert it first"),
        false => fail(400, format!("decoding the audio: {e:#}")),
      })?;
    let transcript = split::transcribe(&model.model, &pcm, None, &options, |_, _| ());
    let (t, starts) = transcript.map_err(|e| match e {
      wavo::Error::Language(_) => fail(400, e.to_string()),
      e => fail(500, format!("transcribing: {e}")),
    })?;

    let ms = (pcm.len() / 16) as u32;
    let cues = || output::cues(&t.tokens, &starts, ms, model.timed);
    let (content_type, body) = match format.as_str() {
      "json" => (JSON, format!("{{\"text\": {}}}\n", quote(&t.text))),
      "text" => (TEXT, format!("{}\n", t.text)),
      "srt" => (TEXT, output::srt(&cues())),
      "vtt" => ("text/vtt; charset=utf-8", output::vtt(&cues())),
      _ => {
        // The language asked for, else the model's only one.
        let langs = model.info.langs;
        let language = options.language.as_deref().or((langs.len() == 1).then(|| langs[0]));
        (JSON, output::verbose_json(&t, &cues(), language.unwrap_or_default(), ms) + "\n")
      }
    };
    let note = format!("{}, {:.1} s of audio", model.info.name, ms as f64 / 1000.0);
    Ok(Reply { content_type, body, note })
  }

  /// The model a request's `model` names: one given on the command line, else the other one
  /// loaded last, else a downloaded one, loaded in its place. 404 for a model not downloaded.
  fn pick(&self, name: &str) -> Res<Arc<Loaded>> {
    let Some(m) = wanted(name) else { return Ok(self.models[0].clone()) };
    if let Some(loaded) = self.models.iter().find(|x| x.info.name == m.name) {
      return Ok(loaded.clone());
    }

    let mut extra = self.extra.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(loaded) = extra.as_ref().filter(|x| x.info.name == m.name) {
      return Ok(loaded.clone());
    }

    let not_found = || fail(404, format!("{0} is not downloaded; run: wavo pull {0}", m.name));
    let path = Hub::new().find(m.name).ok_or_else(not_found)?;
    *extra = None; // its memory goes once its requests finish
    let loaded = load(m, &path).map_err(|e| fail(500, format!("{e:#}")))?;
    *extra = Some(loaded.clone());
    Ok(loaded)
  }
}

/// The wavo model a request's `model` names; `None` for the default: an empty or unknown name,
/// such as OpenAI's `whisper-1`.
fn wanted(name: &str) -> Option<&'static ModelInfo> {
  hub::lookup(name).ok()
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
fn send(mut w: &TcpStream, reply: Res<Reply>) -> io::Result<()> {
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
