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
//! ```
//!
//! Newer `hf` versions make `blobs/<etag>` a symlink into a shared `<root>/blobs/<xx>/<hash>`
//! store: reading just follows it, and `remove` leaves such files to `hf cache rm`.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::{Hub, Progress, RETRIES};
use crate::{Error, Result};

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
    Self { root, from }
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
  /// dropped download resumes up to `RETRIES` more times, after 1, 2 and 4 s.
  pub(super) fn pull_file(
    &self,
    repo: &str,
    file: &str,
    mut progress: impl FnMut(Progress<'_>) -> ControlFlow<()>,
  ) -> Result<PathBuf> {
    let remote = head(repo, file)?;
    self.install(repo, file, &remote, |part| {
      let mut retry = 0;
      loop {
        match download(&remote.url, part, remote.size, &mut progress) {
          Err(e) if retry < RETRIES && transient(&e) => {
            let done = fs::metadata(part).map_or(0, |m| m.len());
            if progress(Progress { done, total: remote.size, retrying: Some(&e) }).is_break() {
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
  fn lock(&self, repo: &str, name: &str) -> Result<File> {
    let dir = self.root.join(".locks").join(self.folder(repo).file_name().unwrap());
    fs::create_dir_all(&dir)?;
    let lock = File::create(dir.join(format!("{name}.lock")))?;
    lock.lock()?;
    Ok(lock)
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
    let hex = |s: &str, lens: &[usize]| {
      lens.contains(&s.len()) && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
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
  /// empty, and the repo folder once no snapshot is left; other files stay. Returns the bytes
  /// freed, or `None` when nothing was cached.
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
    if blobs.iter().any(|b| b.is_symlink()) {
      let run = format!("hf cache rm hf://models/{repo}/{file}");
      return Err(Error::Hub(format!("hf keeps {file} in its shared blob store; run: {run}")));
    }
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
    // snapshots share is listed twice, and no longer a file on its second turn.
    let used = linked(&dir.join("snapshots"));
    for blob in blobs.iter().filter(|b| b.is_file()) {
      if !used.iter().any(|u| Some(u.as_os_str()) == blob.file_name()) {
        freed += fs::metadata(blob)?.len();
        fs::remove_file(blob)?;
      }
    }
    for r in fs::read_dir(dir.join("refs")).into_iter().flatten().flatten() {
      if fs::read_to_string(r.path()).is_ok_and(|c| emptied.iter().any(|e| e == c.trim())) {
        fs::remove_file(r.path())?;
      }
    }
    if self.snapshots(repo).next().is_none() {
      fs::remove_dir_all(&dir)?;
    }
    Ok(Some(freed))
  }
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

fn head(repo: &str, file: &str) -> Result<Remote> {
  let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
  let url = format!("{}/{repo}/resolve/main/{file}", endpoint.trim_end_matches('/'));
  let agent: ureq::Agent = ureq::Agent::config_builder().max_redirects(0).build().into();
  let resp = agent.head(&url).header("Accept-Encoding", "identity").call();
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

/// Appends to `part` from where an interrupted pull stopped, calling `progress` before every chunk.
fn download(
  url: &str,
  part: &Path,
  size: u64,
  progress: &mut impl FnMut(Progress<'_>) -> ControlFlow<()>,
) -> Result<()> {
  let mut out = OpenOptions::new().create(true).append(true).open(part)?;
  let have = out.metadata()?.len();
  let mut done = if have < size { have } else { 0 };
  let resp = ureq::get(url).header("Range", format!("bytes={done}-")).call();
  let resp = resp.map_err(|e| http(url, e))?;
  if resp.status() != 206 || done != have {
    out.set_len(0)?;
    done = 0;
  }
  let (mut body, mut buf) = (resp.into_body().into_reader(), vec![0; 1 << 16]);
  loop {
    if progress(Progress { done, total: size, retrying: None }).is_break() {
      return Err(Error::Cancelled);
    }
    let n = match body.read(&mut buf) {
      Ok(0) => return Ok(()),
      Ok(n) => n,
      Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
      Err(e) => {
        let message = format!("download interrupted: {e}");
        return Err(Error::Http { url: url.into(), status: None, message });
      }
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
    let hub = Hub { root: PathBuf::from("/c"), from: Some("HF_HUB_CACHE") };
    assert_eq!(hub.to_string(), "/c (set by HF_HUB_CACHE)");
    assert_eq!(Hub::at("/c").to_string(), "/c");
  }

  #[test]
  fn retries_only_what_may_pass() {
    let status = |code| http("u", ureq::Error::StatusCode(code));
    assert!(transient(&status(503)) && !transient(&status(404)));
    assert!(transient(&http("u", ureq::Error::Io(io::ErrorKind::ConnectionReset.into()))));
    let broke = Error::Http { url: "u".into(), status: None, message: "interrupted".into() };
    assert!(transient(&broke));
    assert!(transient(&io::Error::from(io::ErrorKind::UnexpectedEof).into()));
    assert!(!transient(&io::Error::from(io::ErrorKind::StorageFull).into()));
    assert!(!transient(&Error::Cancelled) && !transient(&Error::Hub("size".into())));
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

  #[cfg(unix)]
  #[test]
  fn remove_leaves_shared_blobs_to_hf() {
    let hub = temp("shared");
    let (dir, hash) = (hub.folder("org/m"), "ab".repeat(32));
    fs::create_dir_all(hub.root.join("blobs/ab")).unwrap();
    fs::write(hub.root.join("blobs/ab").join(&hash), "hello").unwrap();
    fs::create_dir_all(dir.join("blobs")).unwrap();
    let r = remote('a', '1', 5);
    std::os::unix::fs::symlink(format!("../../blobs/ab/{hash}"), dir.join("blobs").join(&r.etag))
      .unwrap();
    let path = hub.install("org/m", "m.gguf", &r, write("")).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), "hello");
    let e = hub.remove_file("org/m", "m.gguf").unwrap_err().to_string();
    assert!(e.contains("hf cache rm hf://models/org/m/m.gguf"), "{e}");
    assert_eq!(hub.find_file("org/m", "m.gguf"), Some(path));
    fs::remove_dir_all(&hub.root).unwrap();
  }
}
