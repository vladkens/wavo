// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The HTTP/1.1 that `wavo serve` speaks: a request head parsed by `httparse`, a body by
//! Content-Length or chunked, a `multipart/form-data` form, and a response that closes the
//! connection.

use std::io::{self, BufRead, Read, Write};
use std::ops::Range;

/// The longest request head (request line and headers), and line of a chunked body, in bytes.
const MAX_HEAD: usize = 64 << 10;

/// Why a request fails: its status code and a message for the client and the log.
#[derive(Debug, PartialEq)]
pub struct Fail {
  pub status: u16,
  pub message: String,
}

pub type Res<T> = Result<T, Fail>;

pub fn fail(status: u16, message: impl Into<String>) -> Fail {
  Fail { status, message: message.into() }
}

impl From<io::Error> for Fail {
  fn from(e: io::Error) -> Self {
    let timeout = matches!(e.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock);
    fail(if timeout { 408 } else { 400 }, format!("reading the request: {e}"))
  }
}

/// A request line and its headers.
pub struct Head {
  pub method: String,
  /// The path without the query.
  pub path: String,
  /// Names in lowercase, values as sent.
  pub headers: Vec<(String, String)>,
}

impl Head {
  /// The first header named `name`, given in lowercase.
  pub fn header(&self, name: &str) -> Option<&str> {
    self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
  }
}

/// Reads a request head: `None` when the connection closes before a request starts.
pub fn head(r: &mut impl BufRead) -> Res<Option<Head>> {
  let mut buf = Vec::new();
  while !buf.ends_with(b"\r\n\r\n") && !buf.ends_with(b"\n\n") {
    let room = (MAX_HEAD + 1 - buf.len()) as u64;
    if r.by_ref().take(room).read_until(b'\n', &mut buf)? == 0 {
      return if buf.is_empty() { Ok(None) } else { Err(fail(400, "the request ended early")) };
    }

    if buf.len() > MAX_HEAD {
      return Err(fail(431, "the request head is over 64 KiB"));
    }
  }

  let mut headers = [httparse::EMPTY_HEADER; 64];
  let mut req = httparse::Request::new(&mut headers);
  match req.parse(&buf) {
    Ok(httparse::Status::Complete(_)) => {}
    Err(httparse::Error::TooManyHeaders) => return Err(fail(431, "over 64 headers")),
    _ => return Err(fail(400, "not an HTTP request")),
  }

  let (method, path) = (req.method.unwrap_or_default(), req.path.unwrap_or_default());
  let path = path.split('?').next().unwrap_or_default();
  let header = |h: &httparse::Header| {
    (h.name.to_ascii_lowercase(), String::from_utf8_lossy(h.value).into_owned())
  };
  let headers = req.headers.iter().map(header).collect();
  Ok(Some(Head { method: method.into(), path: path.into(), headers }))
}

/// Reads the body of the request with `head`, of up to `limit` bytes, by its Content-Length or
/// chunked. Once the length is known to fit, answers `Expect: 100-continue` on `w`.
pub fn body(r: &mut impl BufRead, w: &mut impl Write, head: &Head, limit: usize) -> Res<Vec<u8>> {
  let encoding = head.header("transfer-encoding").map(str::to_ascii_lowercase);
  let chunked = encoding.is_some_and(|e| e.ends_with("chunked"));
  let len = match head.header("content-length") {
    _ if chunked => None,
    Some(len) => Some(len.trim().parse::<usize>().map_err(|_| fail(400, "bad Content-Length"))?),
    None => return Err(fail(411, "send the body with a Content-Length or chunked")),
  };
  if len.is_some_and(|len| len > limit) {
    return Err(too_large(limit));
  }

  if head.header("expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue")) {
    w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    w.flush()?;
  }

  let Some(len) = len else { return chunks(r, limit) };
  let mut body = Vec::new();
  r.by_ref().take(len as u64).read_to_end(&mut body)?;
  if body.len() < len {
    return Err(fail(400, "the body ended early"));
  }

  Ok(body)
}

fn too_large(limit: usize) -> Fail {
  fail(413, format!("the body is over {} MB", limit >> 20))
}

/// A chunked body: chunks of a hex size line (maybe with `;extensions`), the data and a line
/// end, up to a chunk of size 0, then trailer lines up to an empty one.
fn chunks(r: &mut impl BufRead, limit: usize) -> Res<Vec<u8>> {
  let mut body = Vec::new();
  loop {
    let size = match httparse::parse_chunk_size(&line(r)?) {
      Ok(httparse::Status::Complete((_, size))) => size,
      _ => return Err(fail(400, "bad chunk size")),
    };
    if size == 0 {
      break;
    }

    if size > (limit - body.len()) as u64 {
      return Err(too_large(limit));
    }

    let n = r.by_ref().take(size).read_to_end(&mut body)?;
    if n as u64 != size || !matches!(&line(r)?[..], b"\r\n" | b"\n") {
      return Err(fail(400, "bad chunk"));
    }
  }

  while !matches!(&line(r)?[..], b"\r\n" | b"\n") {}
  Ok(body)
}

/// A line with its line end.
fn line(r: &mut impl BufRead) -> Res<Vec<u8>> {
  let mut line = Vec::new();
  r.by_ref().take(MAX_HEAD as u64).read_until(b'\n', &mut line)?;
  if !line.ends_with(b"\n") {
    return Err(fail(400, "a chunked body ended early"));
  }

  Ok(line)
}

/// One field of a `multipart/form-data` form.
#[derive(Debug, PartialEq)]
pub struct Part {
  pub name: String,
  pub filename: Option<String>,
  /// Where its value lies in the body.
  pub data: Range<usize>,
}

/// The fields of a form `body` sent with the Content-Type `content_type`.
pub fn multipart(content_type: &str, body: &[u8]) -> Res<Vec<Part>> {
  let kind = content_type.split(';').next().unwrap_or_default().trim();
  let boundary = param(content_type, "boundary").filter(|_| kind == "multipart/form-data");
  let boundary = boundary.ok_or_else(|| fail(415, "send the request as multipart/form-data"))?;
  let bad = || fail(400, "a broken multipart/form-data body");
  // Each part ends at a line end and a delimiter; the first delimiter needs no line end.
  let delimiter = format!("\r\n--{boundary}").into_bytes();
  let mut pos = find(body, &delimiter[2..], 0).ok_or_else(bad)? + delimiter.len() - 2;
  let mut parts = Vec::new();
  while !body[pos..].starts_with(b"--") {
    if !body[pos..].starts_with(b"\r\n") {
      return Err(bad());
    }

    let mut headers = [httparse::EMPTY_HEADER; 16];
    let Ok(httparse::Status::Complete((n, headers))) =
      httparse::parse_headers(&body[pos + 2..], &mut headers)
    else {
      return Err(bad());
    };
    let disposition = headers.iter().find(|h| h.name.eq_ignore_ascii_case("content-disposition"));
    let disposition = disposition.map(|h| String::from_utf8_lossy(h.value)).unwrap_or_default();
    let start = pos + 2 + n;
    let end = find(body, &delimiter, start).ok_or_else(bad)?;
    let (name, filename) = (param(&disposition, "name"), param(&disposition, "filename"));
    parts.push(Part { name: name.unwrap_or_default(), filename, data: start..end });
    pos = end + delimiter.len();
  }

  Ok(parts)
}

/// The parameter `key` of a header value such as `form-data; name="file"`, unquoted.
fn param(value: &str, key: &str) -> Option<String> {
  value.split(';').skip(1).find_map(|p| {
    let (k, v) = p.split_once('=')?;
    k.trim().eq_ignore_ascii_case(key).then(|| v.trim().trim_matches('"').to_string())
  })
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
  let i = haystack.get(from..)?.windows(needle.len()).position(|w| w == needle)?;
  Some(from + i)
}

/// Writes a whole response, which closes the connection.
pub fn respond(w: &mut impl Write, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
  let reason = match status {
    200 => "OK",
    401 => "Unauthorized",
    404 => "Not Found",
    408 => "Request Timeout",
    411 => "Length Required",
    413 => "Content Too Large",
    415 => "Unsupported Media Type",
    431 => "Request Header Fields Too Large",
    503 => "Service Unavailable",
    500.. => "Internal Server Error",
    _ => "Bad Request",
  };
  let auth = if status == 401 { "WWW-Authenticate: Bearer\r\n" } else { "" };
  let len = body.len();
  let head = format!(
    "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\n\
     Connection: close\r\n{auth}\r\n"
  );
  w.write_all(head.as_bytes())?;
  w.write_all(body)?;
  w.flush()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The body of a raw request, and what was written back before it.
  fn request(raw: &[u8]) -> (Res<Vec<u8>>, String) {
    let (mut r, mut w) = (raw, Vec::new());
    let head = head(&mut r).unwrap().unwrap();
    (body(&mut r, &mut w, &head, 100), String::from_utf8(w).unwrap())
  }

  fn status<T>(x: &Res<T>) -> u16 {
    x.as_ref().map_or_else(|e| e.status, |_| 200)
  }

  #[test]
  fn heads() {
    let raw = b"GET /v1/models?x=1 HTTP/1.1\r\nHost: a\r\nAuthorization: Bearer k\r\n\r\n";
    let head = head(&mut &raw[..]).unwrap().unwrap();
    assert_eq!((head.method.as_str(), head.path.as_str()), ("GET", "/v1/models"));
    assert_eq!(head.header("authorization"), Some("Bearer k"));
    let parse = |raw: &[u8]| super::head(&mut &raw[..]);
    assert!(matches!(parse(b""), Ok(None)), "closed before a request");
    assert_eq!(status(&parse(b"GET / HTTP/1.1\r\nHost")), 400);
    assert_eq!(status(&parse(b"\x16\x03\x01 TLS\r\n\r\n")), 400);
    let long = [b"GET / HTTP/1.1\r\nX: ".as_slice(), &[b'a'; 70_000], b"\r\n\r\n"].concat();
    assert_eq!(status(&parse(&long)), 431);
  }

  #[test]
  fn bodies_by_length() {
    let (body, written) = request(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello, and more");
    assert_eq!(body.unwrap(), b"hello");
    assert_eq!(written, "");
    let body = |raw: &[u8]| status(&request(raw).0);
    assert_eq!(body(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhel"), 400);
    assert_eq!(body(b"POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n"), 400);
    assert_eq!(body(b"POST / HTTP/1.1\r\n\r\n"), 411);
  }

  #[test]
  fn expect_continue_only_for_a_body_that_fits() {
    let raw = b"POST / HTTP/1.1\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\nok";
    let (body, written) = request(raw);
    assert_eq!(body.unwrap(), b"ok");
    assert_eq!(written, "HTTP/1.1 100 Continue\r\n\r\n");
    let raw = b"POST / HTTP/1.1\r\nContent-Length: 101\r\nExpect: 100-continue\r\n\r\n";
    let (body, written) = request(raw);
    assert_eq!(status(&body), 413);
    assert_eq!(written, "", "no 100 Continue before a 413");
  }

  #[test]
  fn chunked_bodies() {
    let head = "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
    let body = |chunks: &str| request(format!("{head}{chunks}").as_bytes()).0;
    let chunks = "5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\nTrailer: x\r\n\r\n";
    assert_eq!(body(chunks).unwrap(), b"hello, world");
    assert_eq!(body("0\r\n\r\n").unwrap(), b"");
    assert_eq!(status(&body("5\r\nhel")), 400);
    assert_eq!(status(&body("5\r\nhelloXX0\r\n\r\n")), 400);
    assert_eq!(status(&body("zz\r\n")), 400);
    assert_eq!(status(&body("64\r\n")), 400, "100 bytes fit, but none come");
    assert_eq!(status(&body("65\r\n")), 413, "101 bytes don't");
  }

  #[test]
  fn forms() {
    let body = b"preamble\r\n--XyZ\r\ncontent-disposition: form-data; name=\"file\"; filename=\"a b.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n--X\r\n\r\n--XyZ\r\nContent-Disposition: form-data; name=model\r\n\r\nwhisper-1\r\n--XyZ--\r\n";
    let parts = multipart("multipart/form-data; boundary=\"XyZ\"", body).unwrap();
    let value = |i: usize| std::str::from_utf8(&body[parts[i].data.clone()]).unwrap();
    assert_eq!((parts[0].name.as_str(), parts[0].filename.as_deref()), ("file", Some("a b.wav")));
    assert_eq!(value(0), "RIFF\r\n--X\r\n", "a delimiter needs the whole boundary");
    assert_eq!((parts[1].name.as_str(), parts[1].filename.as_deref()), ("model", None));
    assert_eq!(value(1), "whisper-1");
    let empty = multipart("multipart/form-data; boundary=XyZ", b"--XyZ--\r\n").unwrap();
    assert_eq!(empty, []);
    assert_eq!(multipart("application/json", b"{}").unwrap_err().status, 415);
    assert_eq!(multipart("multipart/form-data", b"").unwrap_err().status, 415);
    let unclosed = b"--XyZ\r\nContent-Disposition: form-data; name=a\r\n\r\nvalue";
    assert_eq!(multipart("multipart/form-data; boundary=XyZ", unclosed).unwrap_err().status, 400);
  }

  #[test]
  fn responses() {
    let mut w = Vec::new();
    respond(&mut w, 401, "application/json", b"{}").unwrap();
    assert_eq!(
      String::from_utf8(w).unwrap(),
      "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\
       Connection: close\r\nWWW-Authenticate: Bearer\r\n\r\n{}"
    );
  }
}
