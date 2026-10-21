// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Files in the Hugging Face cache, laid out as the Hub local cache spec says
//! (<https://huggingface.co/docs/hub/local-cache>), so `hf` and `wavo` find each other's downloads:
//!
//! ```text
//! <root>/models--<org>--<repo>/blobs/<etag>                 file content; LFS etag = SHA-256
//!                             /snapshots/<commit>/<file>     -> ../../blobs/<etag>
//!                             /refs/main                     <commit>
//! <root>/.locks/models--<org>--<repo>/<etag>.lock            held while downloading
//!                                    /wavo.lock              wavo's: held by pull and remove
//! <root>/blobs/<xx>/<hash>                                   newer hf: content shared by repos
//!             /<xx>/<hash>.refs                              repo blobs linked to it, one a line
//!             /<xx>/<hash>.lock                              hf's: held to link, publish, delete
//! ```
//!
//! Newer `hf` versions make `blobs/<etag>` a symlink into that shared store (marked by
//! `<root>/blobs/.huggingface-shared-blobs`): reading just follows it, and `remove` deletes the
//! shared file only once no repo links to it any more, as `hf cache rm` does.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::{Hub, Progress, RETRIES};
use crate::{Error, Result};

/// How long connecting, TLS handshake included, may take.
const CONNECT: Duration = Duration::from_secs(15);
/// How long a response may go without a byte, headers or body, before it counts as broken off.
const IDLE: Duration = Duration::from_secs(30);

/// What a HEAD of `resolve/main/<file>` says about the latest revision.
struct Remote {
  commit: String,
  etag: String,
  size: u64,
  url: String,
}

impl Hub {
  /// The root as huggingface_hub resolves it (`constants.py`).
  pub(super) fn from_env(var: impl Fn(&str) -> Option<OsString>) -> Self {
    let set =
      |k: &'static str, hub: fn(PathBuf) -> PathBuf| var(k).map(|v| (hub(v.into()), Some(k)));
    let (root, from) = (set("HF_HUB_CACHE", |p| p))
      .or_else(|| set("HUGGINGFACE_HUB_CACHE", |p| p))
      .or_else(|| set("HF_HOME", |p| p.join("hub")))
      .unwrap_or_else(|| {
        let user = || PathBuf::from(var("HOME").or_else(|| var("USERPROFILE")).unwrap_or_default());
        let xdg = var("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| user().join(".cache"));
        (xdg.join("huggingface").join("hub"), None)
      });
    Self { root, from, prepare: true }
  }

  fn folder(&self, repo: &str) -> PathBuf {
    self.root.join(format!("models--{}", repo.replace('/', "--")))
  }

  fn snapshots(&self, repo: &str) -> impl Iterator<Item = PathBuf> {
    let dirs = fs::read_dir(self.folder(repo).join("snapshots")).into_iter().flatten().flatten();
    dirs.map(|e| e.path()).filter(|p| p.is_dir())
  }

  /// The cached file: in the snapshot `refs/main` names, else in any snapshot that has it.
  pub(super) fn find_file(&self, repo: &str, file: &str) -> Option<PathBuf> {
    let dir = self.folder(repo);
    let main = fs::read_to_string(dir.join("refs/main"));
    let main = main.ok().map(|commit| dir.join("snapshots").join(commit.trim()));
    main.into_iter().chain(self.snapshots(repo)).map(|s| s.join(file)).find(|p| p.exists())
  }

  /// Downloads the file at `main` unless that revision is cached, as huggingface_hub does. A
  /// dropped or stalled download resumes up to `RETRIES` more times, after 1, 2 and 4 s.
  pub(super) fn pull_file(
    &self,
    repo: &str,
    file: &str,
    mut progress: impl FnMut(Progress<'_>) -> ControlFlow<()>,
  ) -> Result<PathBuf> {
    let agent = agent(IDLE);
    let remote = head(&agent, repo, file)?;
    self.install(repo, file, &remote, |part| {
      let mut retry = 0;
      loop {
        match download(&agent, &remote.url, part, remote.size, &mut progress) {
          Err(e) if retry < RETRIES && transient(&e) => {
            let done = fs::metadata(part).map_or(0, |m| m.len());
            if progress(Progress { done, total: remote.size, retrying: Some(&e), preparing: None })
              .is_break()
            {
              return Err(Error::Cancelled);
            }
            std::thread::sleep(Duration::from_secs(1 << retry));
            retry += 1;
          }
          result => return result,
        }
      }
    })
  }

  /// Holds `<root>/.locks/<repo folder>/<name>.lock` until the file is dropped.
  pub(super) fn lock(&self, repo: &str, name: &str) -> Result<File> {
    let dir = self.root.join(".locks").join(self.folder(repo).file_name().unwrap());
    fs::create_dir_all(&dir)?;
    hold(&dir.join(format!("{name}.lock")))
  }

  /// Links the file into its snapshot, `fetch`ing the blob into `<etag>.incomplete` first when it
  /// is missing, then points `refs/main` at the snapshot.
  fn install(
    &self,
    repo: &str,
    file: &str,
    remote: &Remote,
    fetch: impl FnOnce(&Path) -> Result<()>,
  ) -> Result<PathBuf> {
    if !hex(&remote.commit, &[40]) || !hex(&remote.etag, &[40, 64]) {
      let (commit, etag) = (&remote.commit, &remote.etag);
      return Err(Error::Hub(format!(
        "unexpected commit {commit:?} or etag {etag:?} from the Hub"
      )));
    }
    let _repo = self.lock(repo, "wavo")?;
    let dir = self.folder(repo);
    let pointer = dir.join("snapshots").join(&remote.commit).join(file);
    if !pointer.exists() {
      let blob = dir.join("blobs").join(&remote.etag);
      fs::create_dir_all(blob.parent().unwrap())?;
      fs::create_dir_all(pointer.parent().unwrap())?;
      let _blob = self.lock(repo, &remote.etag)?; // huggingface_hub's own per-blob lock
      if !blob.exists() {
        let part = blob.with_file_name(format!("{}.incomplete", remote.etag));
        fetch(&part)?;
        let got = fs::metadata(&part)?.len();
        if got != remote.size {
          let size = remote.size;
          return Err(Error::Hub(format!("{file}: downloaded {got} bytes, expected {size}")));
        }
        fs::rename(&part, &blob)?;
      }
      link(&blob, &pointer, &remote.etag)?;
    }
    let refs = dir.join("refs/main");
    if fs::read_to_string(&refs).ok().as_deref() != Some(remote.commit.as_str()) {
      fs::create_dir_all(refs.parent().unwrap())?;
      let tmp = refs.with_file_name(format!("main.{}.tmp", std::process::id()));
      fs::write(&tmp, &remote.commit)?;
      fs::rename(&tmp, &refs)?;
    }
    Ok(pointer)
  }

  /// Deletes the file from every snapshot along with its blobs, then the snapshots and refs left
  /// empty, and the repo folder once no snapshot is left; other files stay. A blob in the shared
  /// store goes only once no repo links to it. Returns the bytes freed, or `None` when nothing was
  /// cached.
  pub(super) fn remove_file(&self, repo: &str, file: &str) -> Result<Option<u64>> {
    let dir = self.folder(repo);
    if !dir.exists() {
      return Ok(None);
    }
    let _repo = self.lock(repo, "wavo")?;
    let pointers: Vec<PathBuf> =
      self.snapshots(repo).map(|s| s.join(file)).filter(|p| p.symlink_metadata().is_ok()).collect();
    if pointers.is_empty() {
      return Ok(None);
    }
    let blobs: Vec<PathBuf> = (pointers.iter().filter_map(|p| fs::read_link(p).ok()))
      .filter_map(|target| target.file_name().map(|etag| dir.join("blobs").join(etag)))
      .collect();
    let (mut freed, mut emptied) = (0, Vec::new());
    for pointer in &pointers {
      let meta = pointer.symlink_metadata()?;
      freed += if meta.is_file() { meta.len() } else { 0 };
      fs::remove_file(pointer)?;
      let snapshot = pointer.parent().unwrap();
      if fs::remove_dir(snapshot).is_ok() {
        emptied.push(snapshot.file_name().unwrap().to_string_lossy().into_owned());
      }
    }
    // Blobs are content-addressed: another file name may still point at the same one. A blob two
    // snapshots share is listed twice, and gone on its second turn.
    let (used, mut shared) = (linked(&dir.join("snapshots")), Vec::new());
    for blob in &blobs {
      let Ok(meta) = blob.symlink_metadata() else { continue };
      if used.iter().any(|u| Some(u.as_os_str()) == blob.file_name()) {
        continue;
      }
      if meta.is_symlink() {
        shared.extend(self.shared(blob));
      } else {
        freed += meta.len();
      }
      fs::remove_file(blob)?;
    }
    for r in fs::read_dir(dir.join("refs")).into_iter().flatten().flatten() {
      if fs::read_to_string(r.path()).is_ok_and(|c| emptied.iter().any(|e| e == c.trim())) {
        fs::remove_file(r.path())?;
      }
    }
    if self.snapshots(repo).next().is_none() {
      let rest = fs::read_dir(dir.join("blobs")).into_iter().flatten().flatten();
      shared.extend(rest.filter_map(|b| self.shared(&b.path())));
      fs::remove_dir_all(&dir)?;
    }
    for store in &shared {
      freed += self.sweep(store)?;
    }
    Ok(Some(freed))
  }

  /// The file in the shared store, `<root>/blobs/<xx>/<hash>`, that a repo blob links to.
  fn shared(&self, blob: &Path) -> Option<PathBuf> {
    let store = self.root.join("blobs");
    let hash = fs::read_link(blob).ok()?.file_name()?.to_str()?.to_owned();
    let target = store.join(hash.get(..2)?).join(&hash);
    let marked = store.join(".huggingface-shared-blobs").is_file();
    let same = fs::canonicalize(blob).ok()? == fs::canonicalize(&target).ok()?;
    (hex(&hash, &[64]) && marked && same).then_some(target)
  }

  /// Deletes a shared-store file unless a repo still links to it, and returns the bytes freed.
  /// As in `hf cache rm`, the links listed in its `<hash>.refs` are checked under `hf`'s lock for
  /// the file, so a download that links it meanwhile waits; without the list the file stays. Links
  /// in every `<root>/*/blobs/` count as well, listed or not. A link that can't be checked (no
  /// permission) counts as one, so the file stays.
  fn sweep(&self, store: &Path) -> Result<u64> {
    let name = store.file_name().unwrap().to_string_lossy();
    let _lock = hold(&store.with_file_name(format!("{name}.lock")))?;
    let refs = store.with_file_name(format!("{name}.refs"));
    let Ok(listed) = fs::read_to_string(&refs) else { return Ok(0) };
    let target = fs::canonicalize(store)?;
    let absent =
      |e: &io::Error| matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory);
    let links = |b: &Path| match b.symlink_metadata() {
      Ok(m) if m.is_symlink() => match fs::canonicalize(b) {
        Err(e) if absent(&e) => Ok(false),
        t => Ok(t? == target),
      },
      Err(e) if !absent(&e) => Err(e),
      _ => Ok(false),
    };
    let used = || -> io::Result<bool> {
      for line in listed.lines() {
        if links(&self.root.join(line))? {
          return Ok(true);
        }
      }
      for repo in fs::read_dir(&self.root)? {
        match fs::read_dir(repo?.path().join("blobs")) {
          Err(e) if absent(&e) => {}
          blobs => {
            for b in blobs? {
              if links(&b?.path())? {
                return Ok(true);
              }
            }
          }
        }
      }
      Ok(false)
    };
    if !matches!(used(), Ok(false)) {
      return Ok(0);
    }
    let size = fs::symlink_metadata(store)?.len();
    fs::remove_file(store)?;
    fs::remove_file(&refs)?;
    Ok(size)
  }
}

/// Holds a lock on the file at `path`, created when missing, until the file is dropped.
fn hold(path: &Path) -> Result<File> {
  let lock = OpenOptions::new().create(true).truncate(false).write(true).open(path)?;
  lock.lock()?;
  Ok(lock)
}

/// Whether `s` is lowercase hex of one of the lengths.
fn hex(s: &str, lens: &[usize]) -> bool {
  lens.contains(&s.len()) && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The blob names the symlinks anywhere under `dir` point at.
fn linked(dir: &Path) -> Vec<OsString> {
  let entries = fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path());
  (entries.flat_map(|p| match fs::read_link(&p) {
    Ok(target) => target.file_name().map(OsString::from).into_iter().collect(),
    Err(_) => linked(&p),
  }))
  .collect()
}

fn http(url: &str, e: ureq::Error) -> Error {
  let status = if let ureq::Error::StatusCode(code) = e { Some(code) } else { None };
  Error::Http { url: url.into(), status, message: e.to_string() }
}

/// An agent whose requests fail instead of hanging: on a connection that takes over [`CONNECT`],
/// and on a response that sends nothing for `idle`. `timeout_per_read` restarts on every read,
/// unlike `timeout_recv_body`, a budget for the whole body that a large file outlasts on a slow
/// line, so a download as a whole has no time limit.
fn agent(idle: Duration) -> ureq::Agent {
  ureq::Agent::config_builder()
    .timeout_connect(Some(CONNECT))
    .timeout_recv_response(Some(idle))
    .timeout_per_read(Some(idle))
    .build()
    .into()
}

fn head(agent: &ureq::Agent, repo: &str, file: &str) -> Result<Remote> {
  let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
  let url = format!("{}/{repo}/resolve/main/{file}", endpoint.trim_end_matches('/'));
  let req = agent.head(&url).config().max_redirects(0).build();
  let resp = req.header("Accept-Encoding", "identity").call();
  let resp = resp.map_err(|e| http(&url, e))?;
  let header = |k: &str| resp.headers().get(k).and_then(|v| v.to_str().ok());
  let missing = |what: &str| Error::Hub(format!("{url}: no {what}"));
  // A redirect's Content-Length is its own body's; X-Linked-Size is the file's.
  let length = header("content-length").filter(|_| resp.status() == 200);
  let size = header("x-linked-size").or(length);
  let etag = header("x-linked-etag").or_else(|| header("etag")).ok_or_else(|| missing("ETag"))?;
  Ok(Remote {
    commit: header("x-repo-commit").ok_or_else(|| missing("X-Repo-Commit"))?.into(),
    etag: etag.trim_start_matches("W/").trim_matches('"').into(),
    size: size.and_then(|s| s.parse().ok()).ok_or_else(|| missing("file size"))?,
    url: header("location").filter(|l| l.starts_with("http")).unwrap_or(&url).into(),
  })
}

/// Continues `part` from where an interrupted pull stopped, calling `progress` before every chunk.
fn download(
  agent: &ureq::Agent,
  url: &str,
  part: &Path,
  size: u64,
  progress: &mut impl FnMut(Progress<'_>) -> ControlFlow<()>,
) -> Result<()> {
  // Write access, not append: on Windows an append-only handle can't be truncated.
  let mut out = OpenOptions::new().create(true).truncate(false).write(true).open(part)?;
  let have = out.metadata()?.len();
  let Some(mut done) = resume_at(have, size) else { return Ok(()) };
  let resp = agent.get(url).header("Range", format!("bytes={done}-")).call();
  let resp = resp.map_err(|e| http(url, e))?;
  // A server that ignores Range answers 200 with the whole file.
  if resp.status() != 206 || done != have {
    out.set_len(0)?;
    done = 0;
  }
  out.seek(SeekFrom::Start(done))?;
  copy(resp.into_body().into_reader(), &mut out, done, size, url, progress)
}

/// Where a download continues a part of `have` bytes: `None` when it is complete (cancelled after
/// its last chunk), from 0 when it is longer than the file.
fn resume_at(have: u64, size: u64) -> Option<u64> {
  (have != size).then_some(if have < size { have } else { 0 })
}

/// Copies `body` to `out` behind `done` bytes already there, until `size`. A body that ends early
/// is an `Http` error without a status, so the retry resumes it.
fn copy(
  mut body: impl Read,
  out: &mut impl Write,
  mut done: u64,
  size: u64,
  url: &str,
  progress: &mut impl FnMut(Progress<'_>) -> ControlFlow<()>,
) -> Result<()> {
  let mut buf = vec![0; 1 << 16];
  let broke = |message| Error::Http { url: url.into(), status: None, message };
  loop {
    if progress(Progress { done, total: size, retrying: None, preparing: None }).is_break() {
      return Err(Error::Cancelled);
    }
    let n = match body.read(&mut buf) {
      Ok(0) if done < size => {
        return Err(broke(format!("download ended at {done} of {size} bytes")));
      }
      Ok(0) => return Ok(()),
      Ok(n) => n,
      Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
      Err(e) => return Err(broke(format!("download interrupted: {e}"))),
    };
    out.write_all(&buf[..n])?;
    done += n as u64;
  }
}

/// Whether a failed download may get further on another try: not after an HTTP 4xx or a disk
/// error.
fn transient(e: &Error) -> bool {
  use io::ErrorKind::{PermissionDenied, ReadOnlyFilesystem, StorageFull};
  match e {
    Error::Http { status, .. } => status.is_none_or(|code| code >= 500),
    Error::Io(e) => !matches!(e.kind(), PermissionDenied | ReadOnlyFilesystem | StorageFull),
    _ => false,
  }
}

/// The relative symlink the spec asks for. Where symlinks are unavailable (Windows without
/// developer mode) the blob itself moves into the snapshot, as huggingface_hub does.
fn link(blob: &Path, pointer: &Path, etag: &str) -> io::Result<()> {
  let _ = fs::remove_file(pointer);
  let target = Path::new("..").join("..").join("blobs").join(etag);
  #[cfg(unix)]
  let linked = std::os::unix::fs::symlink(&target, pointer);
  #[cfg(windows)]
  let linked = std::os::windows::fs::symlink_file(&target, pointer);
  linked.or_else(|_| fs::rename(blob, pointer))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn temp(name: &str) -> Hub {
    let root = std::env::temp_dir().join(format!("wavo-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    Hub::at(root)
  }

  fn remote(commit: char, etag: char, size: u64) -> Remote {
    let (commit, etag) = (commit.to_string().repeat(40), etag.to_string().repeat(64));
    Remote { commit, etag, size, url: String::new() }
  }

  fn write(text: &'static str) -> impl FnOnce(&Path) -> Result<()> {
    move |part| Ok(fs::write(part, text)?)
  }

  #[test]
  fn root_follows_huggingface_hub() {
    // Paths compare as `Path`s, so the test holds with Windows' separators too.
    let root = |vars: &[(&str, &str)]| {
      let hub = Hub::from_env(|k| vars.iter().rev().find(|v| v.0 == k).map(|v| v.1.into()));
      (hub.root, hub.from)
    };
    let mut vars = vec![("USERPROFILE", "/u")];
    assert_eq!(root(&vars), (PathBuf::from("/u/.cache/huggingface/hub"), None));
    for (var, value, expected, from) in [
      ("HOME", "/h", "/h/.cache/huggingface/hub", None),
      ("XDG_CACHE_HOME", "/x", "/x/huggingface/hub", None),
      ("HF_HOME", "/hf", "/hf/hub", Some("HF_HOME")),
      ("HUGGINGFACE_HUB_CACHE", "/old", "/old", Some("HUGGINGFACE_HUB_CACHE")),
      ("HF_HUB_CACHE", "/c", "/c", Some("HF_HUB_CACHE")),
    ] {
      vars.push((var, value));
      assert_eq!(root(&vars), (PathBuf::from(expected), from), "{var}");
    }
    let hub = Hub { root: PathBuf::from("/c"), from: Some("HF_HUB_CACHE"), prepare: true };
    assert_eq!(hub.to_string(), "/c (set by HF_HUB_CACHE)");
    assert_eq!(Hub::at("/c").to_string(), "/c");
  }

  #[test]
  fn retries_only_what_may_pass() {
    let status = |code| http("u", ureq::Error::StatusCode(code));
    assert!(transient(&status(503)) && !transient(&status(404)));
    assert!(transient(&http("u", ureq::Error::Io(io::ErrorKind::ConnectionReset.into()))));
    assert!(transient(&http("u", ureq::Error::Timeout(ureq::Timeout::Connect))));
    let broke = Error::Http { url: "u".into(), status: None, message: "interrupted".into() };
    assert!(transient(&broke));
    assert!(transient(&io::Error::from(io::ErrorKind::UnexpectedEof).into()));
    assert!(!transient(&io::Error::from(io::ErrorKind::StorageFull).into()));
    assert!(!transient(&Error::Cancelled) && !transient(&Error::Hub("size".into())));
  }

  #[test]
  fn resumes_what_may_continue() {
    assert_eq!(resume_at(0, 10), Some(0));
    assert_eq!(resume_at(4, 10), Some(4));
    assert_eq!(resume_at(10, 10), None, "complete: no new request");
    assert_eq!(resume_at(12, 10), Some(0), "too long: start over");

    let (mut out, mut seen) = (Vec::new(), Vec::new());
    let mut progress = |p: Progress<'_>| {
      seen.push(p.done);
      ControlFlow::Continue(())
    };
    assert!(copy(&b"6789"[..], &mut out, 6, 10, "u", &mut progress).is_ok());
    let early = copy(&b"67"[..], &mut out, 6, 10, "u", &mut progress).unwrap_err();
    assert!(transient(&early), "an early end is retried: {early}");
    assert_eq!((out, seen), (b"678967".to_vec(), vec![6, 10, 6, 8]));
    let mut stop = |_: Progress<'_>| ControlFlow::Break(());
    let cancelled = copy(&b"6789"[..], &mut Vec::new(), 6, 10, "u", &mut stop);
    assert!(matches!(cancelled, Err(Error::Cancelled)));
  }

  #[cfg(unix)]
  #[test]
  fn install_writes_the_spec_layout() {
    let hub = temp("install");
    let dir = hub.root.join("models--org--m-gguf");
    let a = remote('a', '1', 5);
    let path = hub.install("org/m-gguf", "m.gguf", &a, write("hello")).unwrap();
    assert_eq!(path, dir.join("snapshots").join(&a.commit).join("m.gguf"));
    assert_eq!(fs::read_link(&path).unwrap(), Path::new("../../blobs").join(&a.etag));
    assert_eq!(fs::read_to_string(&path).unwrap(), "hello");
    assert_eq!(fs::read_to_string(dir.join("refs/main")).unwrap(), a.commit);
    let locks = hub.root.join(".locks/models--org--m-gguf");
    assert!(locks.join(format!("{}.lock", a.etag)).is_file() && locks.join("wavo.lock").is_file());
    assert_eq!(hub.find_file("org/m-gguf", "m.gguf").as_ref(), Some(&path));

    // A cached revision and a new revision of the same content download nothing.
    let fail = |_: &Path| -> Result<()> { panic!("downloaded again") };
    assert_eq!(hub.install("org/m-gguf", "m.gguf", &a, fail).unwrap(), path);
    let b = remote('b', '1', 5);
    let path = hub.install("org/m-gguf", "m.gguf", &b, fail).unwrap();
    assert_eq!(fs::read_to_string(dir.join("refs/main")).unwrap(), b.commit);
    assert_eq!(hub.find_file("org/m-gguf", "m.gguf"), Some(path));

    // A short or cancelled download fails and stays for the next pull to resume.
    let c = remote('c', '3', 10);
    assert!(hub.install("org/m-gguf", "m.gguf", &c, write("short")).is_err());
    let part = dir.join("blobs").join(format!("{}.incomplete", c.etag));
    assert_eq!(fs::read_to_string(&part).unwrap(), "short");
    let cancel = |_: &Path| -> Result<()> { Err(Error::Cancelled) };
    assert!(matches!(hub.install("org/m-gguf", "m.gguf", &c, cancel), Err(Error::Cancelled)));
    assert_eq!(fs::read_to_string(&part).unwrap(), "short");
    assert_eq!(fs::read_to_string(dir.join("refs/main")).unwrap(), b.commit);
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[test]
  fn install_takes_only_hex_ids() {
    let hub = temp("ids");
    let ok = remote('a', '1', 5);
    for (commit, etag) in [
      ("../../x", ok.etag.as_str()),
      (&ok.commit[..39], &ok.etag),
      (&"A".repeat(40), &ok.etag),
      (&ok.commit, "../blobs/x"),
      (&ok.commit, &ok.etag[..63]),
      (&ok.commit, "\"0123\""),
    ] {
      let r = Remote { commit: commit.into(), etag: etag.into(), size: 5, url: String::new() };
      assert!(hub.install("org/m", "m.gguf", &r, write("hello")).is_err(), "{commit} {etag}");
    }
    assert!(!hub.root.exists(), "nothing written");
    let sha1 = Remote { etag: "f".repeat(40), ..ok };
    assert!(hub.install("org/m", "m.gguf", &sha1, write("hello")).is_ok());
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn remove_waits_for_the_repo_lock() {
    let hub = temp("lock");
    hub.install("org/m", "m.gguf", &remote('a', '1', 5), write("hello")).unwrap();
    let lock = hub.lock("org/m", "wavo").unwrap();
    std::thread::scope(|s| {
      let (tx, rx) = std::sync::mpsc::channel();
      let h = &hub;
      s.spawn(move || tx.send(h.remove_file("org/m", "m.gguf").unwrap()).unwrap());
      assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "rm ran under the lock");
      drop(lock);
      assert_eq!(rx.recv().unwrap(), Some(5));
    });
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn find_prefers_refs_main() {
    let hub = temp("find");
    let snapshots = hub.folder("org/m").join("snapshots");
    assert_eq!(hub.find_file("org/m", "m.gguf"), None);
    for s in ["x", "y"] {
      fs::create_dir_all(snapshots.join(s)).unwrap();
      fs::write(snapshots.join(s).join("m.gguf"), s).unwrap();
    }
    std::os::unix::fs::symlink("../../blobs/gone", snapshots.join("x/n.gguf")).unwrap();
    assert_eq!(hub.find_file("org/m", "n.gguf"), None);
    fs::create_dir_all(hub.folder("org/m").join("refs")).unwrap();
    fs::write(hub.folder("org/m").join("refs/main"), "y").unwrap();
    assert_eq!(hub.find_file("org/m", "m.gguf"), Some(snapshots.join("y/m.gguf")));
    fs::write(hub.folder("org/m").join("refs/main"), "z").unwrap();
    assert!(hub.find_file("org/m", "m.gguf").is_some());
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn find_and_remove_by_name() {
    let hub = temp("names");
    let m = super::super::lookup("gigaam-v3").unwrap();
    hub.install(&m.repo(), &m.file(), &remote('a', '1', 5), write("hello")).unwrap();
    let path = hub.find("gigaam-v3-e2e-rnnt").unwrap();
    assert!(path.starts_with(hub.root.join("models--handy-computer--gigaam-v3-e2e-rnnt-gguf")));
    assert_eq!(hub.remove("gigaam-v3").unwrap(), Some(5));
    assert_eq!((hub.find("gigaam-v3"), hub.remove("gigaam-v3").unwrap()), (None, None));
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn remove_keeps_other_files() {
    let hub = temp("remove");
    let dir = hub.folder("org/m");
    hub.install("org/m", "m.gguf", &remote('a', '1', 5), write("hello")).unwrap();
    hub.install("org/m", "other.gguf", &remote('a', '2', 3), write("abc")).unwrap();
    hub.install("org/m", "m.gguf", &remote('b', '1', 5), write("hello")).unwrap();
    assert_eq!(hub.remove_file("org/m", "m.gguf").unwrap(), Some(5));
    assert_eq!(hub.find_file("org/m", "m.gguf"), None);
    assert!(!dir.join("blobs").join("1".repeat(64)).exists());
    assert!(!dir.join("snapshots").join("b".repeat(40)).exists());
    assert!(!dir.join("refs/main").exists(), "refs/main named the removed snapshot");
    assert!(hub.find_file("org/m", "other.gguf").is_some());
    assert_eq!(hub.remove_file("org/m", "m.gguf").unwrap(), None);

    // Two names with the same content share a blob: it goes with the last of them.
    hub.install("org/m", "x.gguf", &remote('c', '4', 4), write("four")).unwrap();
    hub.install("org/m", "y.gguf", &remote('c', '4', 4), write("four")).unwrap();
    assert_eq!(hub.remove_file("org/m", "x.gguf").unwrap(), Some(0));
    assert_eq!(fs::read_to_string(hub.find_file("org/m", "y.gguf").unwrap()).unwrap(), "four");
    assert_eq!(hub.remove_file("org/m", "y.gguf").unwrap(), Some(4));
    assert!(!dir.join("blobs").join("4".repeat(64)).exists());
    assert_eq!(hub.remove_file("org/m", "other.gguf").unwrap(), Some(3));
    assert!(!dir.exists());
    fs::remove_dir_all(&hub.root).unwrap();
  }

  /// Puts `text` into the shared store, as a newer `hf` does, and links `file` of `repo` to it
  /// through the repo blob for `etag`; `listed` adds the link to the store's `.refs`. Returns the
  /// shared file.
  #[cfg(unix)]
  fn share(
    hub: &Hub,
    repo: &str,
    file: &str,
    etag: char,
    text: &'static str,
    listed: bool,
  ) -> PathBuf {
    let r = remote('a', etag, text.len() as u64);
    let hash = format!("ab{}", etag.to_string().repeat(62));
    let store = hub.root.join("blobs/ab").join(&hash);
    fs::create_dir_all(store.parent().unwrap()).unwrap();
    fs::write(hub.root.join("blobs/.huggingface-shared-blobs"), "1\n").unwrap();
    fs::write(&store, text).unwrap();
    let blobs = hub.folder(repo).join("blobs");
    fs::create_dir_all(&blobs).unwrap();
    std::os::unix::fs::symlink(format!("../../blobs/ab/{hash}"), blobs.join(&r.etag)).unwrap();
    if listed {
      let refs = store.with_file_name(format!("{hash}.refs"));
      let mut refs = OpenOptions::new().create(true).append(true).open(refs).unwrap();
      let folder = hub.folder(repo).file_name().unwrap().to_str().unwrap().to_owned();
      writeln!(refs, "{folder}/blobs/{}", r.etag).unwrap();
    }
    let path = hub.install(repo, file, &r, write("")).unwrap();
    assert_eq!(fs::read_to_string(path).unwrap(), text);
    store
  }

  #[cfg(unix)]
  #[test]
  fn remove_deletes_a_shared_blob_only_it_links() {
    let hub = temp("shared-sole");
    let store = share(&hub, "org/m", "m.gguf", '1', "hello", true);
    assert_eq!(hub.remove_file("org/m", "m.gguf").unwrap(), Some(5));
    assert!(!store.exists() && !store.with_extension("refs").exists());
    assert!(!hub.folder("org/m").exists());

    // Without a `.refs` list hf keeps the file, and so does remove.
    let store = share(&hub, "org/m", "m.gguf", '1', "hello", false);
    assert_eq!(hub.remove_file("org/m", "m.gguf").unwrap(), Some(0));
    assert!(store.exists() && !hub.folder("org/m").exists());
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn remove_keeps_a_shared_blob_another_repo_links() {
    let hub = temp("shared-other");
    let store = share(&hub, "org/m", "m.gguf", '1', "hello", true);
    share(&hub, "org/n", "n.gguf", '1', "hello", true);
    assert_eq!(hub.remove_file("org/m", "m.gguf").unwrap(), Some(0));
    assert!(store.exists() && !hub.folder("org/m").exists());
    assert_eq!(fs::read_to_string(hub.find_file("org/n", "n.gguf").unwrap()).unwrap(), "hello");
    assert_eq!(hub.remove_file("org/n", "n.gguf").unwrap(), Some(5));
    assert!(!store.exists());

    // A link the list misses counts too.
    let store = share(&hub, "org/m", "m.gguf", '1', "hello", true);
    share(&hub, "org/n", "n.gguf", '1', "hello", false);
    assert_eq!(hub.remove_file("org/m", "m.gguf").unwrap(), Some(0));
    assert_eq!(fs::read_to_string(hub.find_file("org/n", "n.gguf").unwrap()).unwrap(), "hello");
    assert!(store.exists());
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn remove_keeps_a_shared_blob_it_cannot_check() {
    use std::os::unix::fs::PermissionsExt;
    let hub = temp("shared-denied");
    for listed in [true, false] {
      let store = share(&hub, "org/m", "m.gguf", '1', "hello", true);
      share(&hub, "org/n", "n.gguf", '1', "hello", listed);
      let blobs = hub.folder("org/n").join("blobs");
      fs::set_permissions(&blobs, fs::Permissions::from_mode(0o000)).unwrap();
      if fs::read_dir(&blobs).is_ok() {
        fs::set_permissions(&blobs, fs::Permissions::from_mode(0o755)).unwrap();
        break; // root reads it anyway
      }
      let freed = hub.remove_file("org/m", "m.gguf").unwrap();
      fs::set_permissions(&blobs, fs::Permissions::from_mode(0o755)).unwrap();
      assert_eq!((freed, store.exists()), (Some(0), true), "listed: {listed}");
      fs::remove_dir_all(hub.folder("org/n")).unwrap();
      fs::remove_file(store.with_extension("refs")).unwrap();
    }
    fs::remove_dir_all(&hub.root).unwrap();
  }

  #[cfg(unix)]
  #[test]
  fn remove_mixes_shared_and_own_blobs() {
    let hub = temp("shared-mixed");
    let kept = share(&hub, "org/m", "x.gguf", '1', "hello", true);
    share(&hub, "org/n", "x.gguf", '1', "hello", true);
    let gone = share(&hub, "org/m", "y.gguf", '2', "four", true);
    hub.install("org/m", "z.gguf", &remote('a', '3', 3), write("abc")).unwrap();
    assert_eq!(hub.remove_file("org/m", "x.gguf").unwrap(), Some(0));
    assert_eq!(hub.remove_file("org/m", "y.gguf").unwrap(), Some(4));
    assert!(kept.exists() && !gone.exists());
    assert_eq!(fs::read_to_string(hub.find_file("org/m", "z.gguf").unwrap()).unwrap(), "abc");
    assert_eq!(hub.remove_file("org/m", "z.gguf").unwrap(), Some(3));
    assert!(!hub.folder("org/m").exists());
    assert_eq!(fs::read_to_string(hub.find_file("org/n", "x.gguf").unwrap()).unwrap(), "hello");
    fs::remove_dir_all(&hub.root).unwrap();
  }

  /// Serves `body` to one request: honours `Range: bytes=N-` with 206, or always 200 when `range`
  /// is false, like a mirror that ignores it. With `stall` it sends only the first `stall` bytes
  /// and keeps the connection open until the client hangs up.
  fn serve_once(body: &'static [u8], range: bool, stall: Option<usize>) -> String {
    use std::io::{BufRead, BufReader};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/f", listener.local_addr().unwrap());
    std::thread::spawn(move || {
      let (stream, _) = listener.accept().unwrap();
      let mut reader = BufReader::new(stream.try_clone().unwrap());
      let mut from = 0;
      loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
          break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
          from = v.trim().trim_end_matches('-').parse().unwrap();
        }
      }
      let (status, part) =
        if range { ("206 Partial Content", &body[from..]) } else { ("200 OK", body) };
      let mut out = stream;
      let head =
        format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", part.len());
      out.write_all(head.as_bytes()).unwrap();
      out.write_all(&part[..stall.unwrap_or(part.len())]).unwrap();
      let _ = reader.read_to_end(&mut Vec::new());
    });
    url
  }

  #[test]
  fn download_resumes_or_restarts() {
    let hub = temp("download");
    fs::create_dir_all(&hub.root).unwrap();
    let part = hub.root.join("f.incomplete");
    let body: &[u8] = b"hello, world";
    let agent = agent(Duration::from_millis(300));
    let get = |range, stall| {
      let url = serve_once(body, range, stall);
      download(&agent, &url, &part, body.len() as u64, &mut |_| ControlFlow::Continue(()))
    };
    for range in [true, false] {
      fs::write(&part, b"hello").unwrap();
      get(range, None).unwrap();
      // 206 continues after the 5 bytes already there; 200 ignored Range, so the part starts over.
      assert_eq!(fs::read(&part).unwrap(), body, "range: {range}");
    }

    // A body that stops coming fails after the idle time, and the retry continues from there.
    fs::write(&part, b"hello").unwrap();
    let start = std::time::Instant::now();
    let stalled = get(true, Some(2)).unwrap_err();
    assert!(start.elapsed() < Duration::from_secs(5), "took {:?}", start.elapsed());
    assert!(transient(&stalled), "a stall is retried: {stalled}");
    assert_eq!(fs::read(&part).unwrap(), b"hello, ");
    get(true, None).unwrap();
    assert_eq!(fs::read(&part).unwrap(), body);
    fs::remove_dir_all(&hub.root).unwrap();
  }
}
