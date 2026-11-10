// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! GGUF v3 reader: metadata from the header, F32/F16/Q8_0 tensors streamed from the file, or on
//! macOS and Windows from its memory map.

use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::f16;

use crate::error::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
  F32,
  F16,
  /// Blocks of 32 weights: f16 scale + 32 × i8 (34 bytes).
  Q8_0,
}

impl Type {
  fn from_ggml(id: u32) -> Option<Self> {
    match id {
      0 => Some(Self::F32),
      1 => Some(Self::F16),
      8 => Some(Self::Q8_0),
      _ => None,
    }
  }

  fn size(self, n: usize) -> usize {
    match self {
      Self::F32 => n * 4,
      Self::F16 => n * 2,
      Self::Q8_0 => n / 32 * 34,
    }
  }
}

/// A metadata value, decoded from the header when a model reads it.
#[derive(Debug, Clone, Copy)]
pub enum Value<'a> {
  /// Any integer type: no model reads a u64 above `i64::MAX`.
  Int(i64),
  Float(f64),
  Bool(bool),
  /// `None` when not UTF-8.
  Str(Option<&'a str>),
  /// Items of GGUF type `item`, read by `Gguf::array`.
  Array {
    item: u32,
    len: usize,
  },
}

/// A metadata type that models read: `u32`, `i32`, `f32`, `bool` or `&str`.
pub trait Meta<'a>: Sized {
  fn from(v: Value<'a>) -> Option<Self>;
}

impl Meta<'_> for u32 {
  fn from(v: Value) -> Option<Self> {
    if let Value::Int(v) = v { v.try_into().ok() } else { None }
  }
}

impl Meta<'_> for i32 {
  fn from(v: Value) -> Option<Self> {
    if let Value::Int(v) = v { v.try_into().ok() } else { None }
  }
}

impl Meta<'_> for f32 {
  fn from(v: Value) -> Option<Self> {
    if let Value::Float(v) = v { Some(v as f32) } else { None }
  }
}

impl Meta<'_> for bool {
  fn from(v: Value) -> Option<Self> {
    if let Value::Bool(v) = v { Some(v) } else { None }
  }
}

impl<'a> Meta<'a> for &'a str {
  fn from(v: Value<'a>) -> Option<Self> {
    if let Value::Str(s) = v { s } else { None }
  }
}

struct Info {
  dims: Vec<usize>,
  ty: Type,
  offset: usize,
}

pub struct Gguf {
  path: PathBuf,
  /// The start of the file, at least the header; tensors inside it are read from here.
  head: Vec<u8>,
  /// Per key the value's GGUF type and its offset in `head`. Values are decoded when read, so a
  /// vocabulary nobody reads costs nothing.
  meta: HashMap<String, (u32, usize)>,
  tensors: HashMap<String, Info>,
  data: usize,
  /// The file the header was read from, for `map`: the same one even if a new file was renamed
  /// over the path since.
  file: Option<File>,
  /// The whole file, where it maps: tensors are copied from it instead of read.
  map: Option<map::Map>,
}

/// A tensor in the file. `dims` are in ggml order: `dims[0]` is the contiguous axis, so a linear
/// weight `[in, out]` is row-major `[out][in]`.
pub struct Tensor<'a> {
  pub dims: &'a [usize],
  pub ty: Type,
  gguf: &'a Gguf,
  start: usize,
  size: usize,
}

impl Gguf {
  /// Reads the header only; tensor data is read when used.
  pub fn open(path: &Path) -> Result<Self> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    // The header is usually far below 1 MiB; read more of the file if it doesn't fit.
    let mut n = len.min(1 << 20);
    loop {
      let mut head = vec![0; n];
      file.seek(SeekFrom::Start(0))?;
      file.read_exact(&mut head)?;
      match Self::parse(head, len, path.into()) {
        Err(_) if n < len => n = len.min(n * 4),
        r => return r.map(|g| Self { file: Some(file), ..g }),
      }
    }
  }

  fn parse(head: Vec<u8>, len: usize, path: PathBuf) -> Result<Self> {
    let mut r = Reader { bytes: &head, pos: 0 };
    if r.take(4)? != b"GGUF" {
      bail!("not a GGUF file");
    }
    let version = r.u32()?;
    if version != 3 {
      bail!("unsupported GGUF version {version}");
    }
    let n_tensors = r.u64()? as usize;
    let n_meta = r.u64()? as usize;

    let mut meta = HashMap::new();
    for _ in 0..n_meta {
      let key = r.string()?;
      let ty = r.u32()?;
      meta.insert(key, (ty, r.pos));
      r.skip(ty)?;
    }

    let mut tensors = HashMap::new();
    for _ in 0..n_tensors {
      let name = r.string()?;
      let n_dims = r.u32()? as usize;
      let dims = (0..n_dims).map(|_| r.u64().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
      let id = r.u32()?;
      let Some(ty) = Type::from_ggml(id) else {
        bail!("tensor {name}: unsupported ggml type {id}")
      };
      let offset = r.u64()? as usize;
      tensors.insert(name, Info { dims, ty, offset });
    }

    let end = r.pos;
    let mut g = Self { path, head, meta, tensors, data: 0, file: None, map: None };
    let align = match g.value("general.alignment") {
      Ok((_, Value::Int(a))) => a as usize,
      _ => 32,
    };
    g.data = end.next_multiple_of(align);
    for (name, t) in &g.tensors {
      if g.data + t.offset + t.ty.size(t.dims.iter().product()) > len {
        bail!("tensor {name} is out of file bounds");
      }
    }
    Ok(g)
  }

  /// The value of `key`, and a reader past it (at an array's items).
  fn value(&self, key: &str) -> Result<(Reader<'_>, Value<'_>)> {
    let Some(&(ty, pos)) = self.meta.get(key) else { bail!("missing metadata {key}") };
    let mut r = Reader { bytes: &self.head, pos };
    let v = r.value(ty)?;
    Ok((r, v))
  }

  /// The metadata value `key` as a `T`.
  pub fn get<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<T> {
    let (_, v) = self.value(key)?;
    match T::from(v) {
      Some(v) => Ok(v),
      None => bail!("metadata {key}: expected {}, got {v:?}", std::any::type_name::<T>()),
    }
  }

  /// The metadata value `key` as a `T`, or `None` when the file has no `key`.
  pub fn optional<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<Option<T>> {
    if !self.meta.contains_key(key) {
      return Ok(None);
    }
    self.get(key).map(Some)
  }

  /// Fails unless the metadata value `key` is one of `allowed`.
  pub fn check<'a, T: Meta<'a> + PartialEq + Debug>(
    &'a self,
    key: &str,
    allowed: &[T],
  ) -> Result<()> {
    let got = self.get::<T>(key)?;
    if !allowed.contains(&got) {
      bail!("{key} is {got:?}, only {allowed:?} supported");
    }
    Ok(())
  }

  /// The metadata array `key` as `T`s.
  pub fn array<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<Vec<T>> {
    let (mut r, Value::Array { item, len }) = self.value(key)? else {
      bail!("metadata {key}: expected an array")
    };
    let items = (0..len).map(|_| r.value(item).map(T::from)).collect::<Result<Option<_>>>()?;
    let Some(items) = items else {
      bail!("metadata {key}: expected an array of {}", std::any::type_name::<T>())
    };
    Ok(items)
  }

  /// The metadata array `key` as `T`s, empty when the file has no `key`.
  pub fn optional_array<'a, T: Meta<'a>>(&'a self, key: &str) -> Result<Vec<T>> {
    if !self.meta.contains_key(key) {
      return Ok(Vec::new());
    }
    self.array(key)
  }

  /// Maps the file, for a load that copies all the weights (the GPU's): tensors come from the map
  /// then, and the whole file is read into the page cache at once rather than page by page as
  /// they are copied. Where it doesn't map, they are read from the file.
  pub fn map(&mut self) {
    self.map = self.file.as_ref().and_then(map::Map::new);
  }

  /// The tensor `name`, which must have exactly the ggml-order `dims`.
  pub fn tensor(&self, name: &str, dims: &[usize]) -> Result<Tensor<'_>> {
    let Some(t) = self.tensors.get(name) else { bail!("missing tensor {name}") };
    if t.dims != dims {
      bail!("tensor {name}: shape {:?}, expected {dims:?}", t.dims);
    }
    let (start, size) = (self.data + t.offset, t.ty.size(dims.iter().product()));
    Ok(Tensor { dims: &t.dims, ty: t.ty, gguf: self, start, size })
  }
}

impl Tensor<'_> {
  pub fn len(&self) -> usize {
    self.dims.iter().product()
  }

  /// Calls `f(offset, bytes)` on consecutive pieces of the tensor's data of up to `chunk` bytes.
  pub fn read(&self, chunk: usize, mut f: impl FnMut(usize, &[u8])) -> Result<()> {
    let mapped = self.gguf.map.as_ref().map_or(&self.gguf.head[..], map::Map::bytes);
    if let Some(bytes) = mapped.get(self.start..self.start + self.size) {
      bytes.chunks(chunk).enumerate().for_each(|(i, c)| f(i * chunk, c));
      return Ok(());
    }
    let mut file = File::open(&self.gguf.path)?;
    file.seek(SeekFrom::Start(self.start as u64))?;
    let mut buf = vec![0; chunk.min(self.size)];
    for offset in (0..self.size).step_by(chunk) {
      let piece = &mut buf[..chunk.min(self.size - offset)];
      file.read_exact(piece)?;
      f(offset, piece);
    }
    Ok(())
  }

  pub fn to_f32(&self) -> Result<Vec<f32>> {
    let mut b = Vec::with_capacity(self.size);
    self.read(self.size.max(1), |_, piece| b.extend_from_slice(piece))?;
    Ok(match self.ty {
      Type::F32 => b.as_chunks().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
      Type::F16 => b.as_chunks().0.iter().map(|&c| f16::from_le_bytes(c).to_f32()).collect(),
      Type::Q8_0 => (b.as_chunks::<34>().0.iter())
        .flat_map(|block| {
          let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
          block[2..].iter().map(move |&q| q as i8 as f32 * d)
        })
        .collect(),
    })
  }
}

#[cfg(test)]
impl Gguf {
  /// An in-memory file holding `(name, ggml dims, type, data)` tensors, for tests.
  pub(crate) fn with_tensors(tensors: &[(&str, &[usize], Type, &[u8])]) -> Self {
    let mut head = Vec::new();
    let mut infos = HashMap::new();
    for &(name, dims, ty, data) in tensors {
      infos.insert(name.to_string(), Info { dims: dims.to_vec(), ty, offset: head.len() });
      head.extend_from_slice(data);
    }
    let (path, meta, data) = (PathBuf::new(), HashMap::new(), 0);
    Self { path, head, meta, tensors: infos, data, file: None, map: None }
  }
}

/// A read-only memory map of the whole file while a model loads, on macOS and Windows: the weights
/// are copied from the page cache into their buffers, with no read into a buffer between, and a
/// file the page cache lacks is read ahead in large requests (macOS `F_RDADVISE`, Windows
/// `PrefetchVirtualMemory`) instead of page by page as it is copied (see docs/perf.md). On Linux
/// the map gained nothing warm and lost cold, so there, as elsewhere, `Map::new` gives `None` and
/// tensors are read from the file.
mod map {
  #[cfg(target_os = "macos")]
  pub use macos::Map;
  #[cfg(windows)]
  pub use windows::Map;

  #[cfg(not(any(target_os = "macos", windows)))]
  pub enum Map {}

  #[cfg(not(any(target_os = "macos", windows)))]
  impl Map {
    pub fn new(_: &std::fs::File) -> Option<Self> {
      None
    }

    pub fn bytes(&self) -> &[u8] {
      match *self {}
    }
  }

  #[cfg(target_os = "macos")]
  mod macos {
    use std::ffi::c_void;
    use std::fs::File;
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
      fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        off: i64,
      ) -> *mut c_void;
      fn munmap(addr: *mut c_void, len: usize) -> i32;
      fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }

    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    const F_RDADVISE: i32 = 44;

    /// `struct radvisory` of `<sys/fcntl.h>`.
    #[repr(C)]
    struct Radvisory {
      offset: i64,
      count: i32,
    }

    pub struct Map {
      ptr: *mut c_void,
      len: usize,
    }

    // SAFETY: the mapping is read-only and lives until `drop`.
    unsafe impl Send for Map {}
    unsafe impl Sync for Map {}

    impl Map {
      /// The file mapped and read ahead, or `None` when it doesn't map: tensors are read then.
      pub fn new(file: &File) -> Option<Self> {
        let (fd, len) = (file.as_raw_fd(), file.metadata().ok()?.len() as usize);
        // SAFETY: a new private read-only mapping, which aliases nothing of ours.
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
        if ptr as isize == -1 {
          return None;
        }

        // A hint only, in pieces of at most 1 GiB (`count` is an `int`): when it fails, the pages
        // are read as they are first touched.
        for offset in (0..len).step_by(1 << 30) {
          let count = (len - offset).min(1 << 30) as i32;
          let advice = Radvisory { offset: offset as i64, count };
          // SAFETY: `fcntl` reads `advice` during the call only.
          unsafe { fcntl(fd, F_RDADVISE, &advice as *const Radvisory) };
        }
        Some(Self { ptr, len })
      }

      pub fn bytes(&self) -> &[u8] {
        // SAFETY: mapped for `len` bytes until `drop`. Model files don't change while they
        // load: one truncated meanwhile would fault (SIGBUS) where a read would fail.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
      }
    }

    impl Drop for Map {
      fn drop(&mut self) {
        // SAFETY: mapped by `new`; no slice of it outlives `self`.
        unsafe { munmap(self.ptr, self.len) };
      }
    }
  }

  #[cfg(windows)]
  mod windows {
    use std::ffi::c_void;
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;

    #[link(name = "kernel32")]
    unsafe extern "system" {
      fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *const c_void,
        protect: u32,
        size_high: u32,
        size_low: u32,
        name: *const u16,
      ) -> *mut c_void;
      fn MapViewOfFile(
        mapping: *mut c_void,
        access: u32,
        offset_high: u32,
        offset_low: u32,
        len: usize,
      ) -> *mut c_void;
      fn UnmapViewOfFile(addr: *const c_void) -> i32;
      fn CloseHandle(handle: *mut c_void) -> i32;
      fn PrefetchVirtualMemory(
        process: *mut c_void,
        n: usize,
        ranges: *const Range,
        flags: u32,
      ) -> i32;
    }

    const PAGE_READONLY: u32 = 2;
    const FILE_MAP_READ: u32 = 4;
    /// `GetCurrentProcess()`'s pseudo handle.
    const CURRENT_PROCESS: *mut c_void = -1isize as *mut c_void;

    /// `WIN32_MEMORY_RANGE_ENTRY` of `<memoryapi.h>`.
    #[repr(C)]
    struct Range {
      addr: *mut c_void,
      len: usize,
    }

    pub struct Map {
      ptr: *mut c_void,
      len: usize,
    }

    // SAFETY: the view is read-only and lives until `drop`.
    unsafe impl Send for Map {}
    unsafe impl Sync for Map {}

    impl Map {
      /// The file mapped and read ahead, or `None` when it doesn't map (an empty file doesn't):
      /// tensors are read then.
      pub fn new(file: &File) -> Option<Self> {
        let len = file.metadata().ok()?.len() as usize;
        let handle = file.as_raw_handle();
        // SAFETY: a read-only mapping of an open file; the view keeps it alive once the
        // mapping's handle is closed.
        let ptr = unsafe {
          let mapping =
            CreateFileMappingW(handle, std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null());
          if mapping.is_null() {
            return None;
          }

          let ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
          CloseHandle(mapping);
          ptr
        };
        if ptr.is_null() {
          return None;
        }

        // A hint only: the system reads the whole file into memory in large requests.
        let range = Range { addr: ptr, len };
        // SAFETY: `range` is the view's and read during the call only.
        unsafe { PrefetchVirtualMemory(CURRENT_PROCESS, 1, &range, 0) };
        Some(Self { ptr, len })
      }

      pub fn bytes(&self) -> &[u8] {
        // SAFETY: mapped for `len` bytes until `drop`. Model files don't change while they
        // load: Windows refuses to truncate a mapped file, and a read error faults where a read
        // would fail.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
      }
    }

    impl Drop for Map {
      fn drop(&mut self) {
        // SAFETY: mapped by `new`; no slice of it outlives `self`.
        unsafe { UnmapViewOfFile(self.ptr) };
      }
    }
  }
}

struct Reader<'a> {
  bytes: &'a [u8],
  pos: usize,
}

impl<'a> Reader<'a> {
  fn take(&mut self, n: usize) -> Result<&'a [u8]> {
    let Some(s) = self.bytes.get(self.pos..self.pos + n) else { bail!("truncated GGUF header") };
    self.pos += n;
    Ok(s)
  }

  fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
    Ok(self.take(N)?.try_into().unwrap())
  }

  fn u32(&mut self) -> Result<u32> {
    Ok(u32::from_le_bytes(self.array()?))
  }

  fn u64(&mut self) -> Result<u64> {
    Ok(u64::from_le_bytes(self.array()?))
  }

  fn bytes(&mut self) -> Result<&'a [u8]> {
    let n = self.u64()? as usize;
    self.take(n)
  }

  fn string(&mut self) -> Result<String> {
    Ok(String::from_utf8_lossy(self.bytes()?).into_owned())
  }

  /// Moves past a value of type `ty`.
  fn skip(&mut self, ty: u32) -> Result<()> {
    let size = |ty| match ty {
      0 | 1 | 7 => Some(1),
      2 | 3 => Some(2),
      4..=6 => Some(4),
      10..=12 => Some(8),
      _ => None,
    };
    match ty {
      8 => _ = self.bytes()?,
      9 => {
        let item = self.u32()?;
        let n = self.u64()? as usize;
        match size(item) {
          Some(size) => _ = self.take(n * size)?,
          None => (0..n).try_for_each(|_| self.skip(item))?,
        }
      }
      _ => match size(ty) {
        Some(size) => _ = self.take(size)?,
        None => bail!("unknown GGUF value type {ty}"),
      },
    }
    Ok(())
  }

  /// Reads a value of type `ty`; for an array its item type and length, the items follow.
  fn value(&mut self, ty: u32) -> Result<Value<'a>> {
    Ok(match ty {
      0 => Value::Int(self.array::<1>()?[0] as i64),
      1 => Value::Int(self.array::<1>()?[0] as i8 as i64),
      2 => Value::Int(u16::from_le_bytes(self.array()?) as i64),
      3 => Value::Int(i16::from_le_bytes(self.array()?) as i64),
      4 => Value::Int(self.u32()? as i64),
      5 => Value::Int(i32::from_le_bytes(self.array()?) as i64),
      6 => Value::Float(f32::from_le_bytes(self.array()?) as f64),
      7 => Value::Bool(self.array::<1>()?[0] != 0),
      8 => Value::Str(std::str::from_utf8(self.bytes()?).ok()),
      9 => Value::Array { item: self.u32()?, len: self.u64()? as usize },
      10 => Value::Int(self.u64()? as i64),
      11 => Value::Int(i64::from_le_bytes(self.array()?)),
      12 => Value::Float(f64::from_le_bytes(self.array()?)),
      _ => bail!("unknown GGUF value type {ty}"),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
  }

  /// A file with a u32, a string array, a bool, a negative i32, an i32 array and an f32, and an
  /// F32 + a Q8_0 tensor.
  fn sample() -> Vec<u8> {
    let mut f = b"GGUF".to_vec();
    f.extend(3u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    f.extend(6u64.to_le_bytes());
    string(&mut f, "a.n");
    f.extend(4u32.to_le_bytes());
    f.extend(7u32.to_le_bytes());
    string(&mut f, "a.tokens");
    f.extend(9u32.to_le_bytes());
    f.extend(8u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    string(&mut f, "▁x");
    string(&mut f, "y");
    string(&mut f, "a.flag");
    f.extend(7u32.to_le_bytes());
    f.push(1);
    string(&mut f, "a.left");
    f.extend(5u32.to_le_bytes());
    f.extend((-1i32).to_le_bytes());
    string(&mut f, "a.durations");
    f.extend(9u32.to_le_bytes());
    f.extend(5u32.to_le_bytes());
    f.extend(3u64.to_le_bytes());
    [0i32, 2, -4].iter().for_each(|v| f.extend(v.to_le_bytes()));
    string(&mut f, "a.alpha");
    f.extend(6u32.to_le_bytes());
    f.extend(0.97f32.to_le_bytes());

    string(&mut f, "w");
    f.extend(2u32.to_le_bytes());
    f.extend(2u64.to_le_bytes());
    f.extend(1u64.to_le_bytes());
    f.extend(0u32.to_le_bytes());
    f.extend(0u64.to_le_bytes());
    string(&mut f, "q");
    f.extend(1u32.to_le_bytes());
    f.extend(32u64.to_le_bytes());
    f.extend(8u32.to_le_bytes());
    f.extend(32u64.to_le_bytes());

    f.resize(f.len().next_multiple_of(32), 0);
    f.extend(1.5f32.to_le_bytes());
    f.extend((-2.0f32).to_le_bytes());
    f.resize(f.len().next_multiple_of(32), 0);
    f.extend(f16::from_f32(0.5).to_le_bytes());
    f.extend((0..32).map(|i| (i as i8 - 16) as u8));
    f
  }

  fn parse(f: Vec<u8>) -> Result<Gguf> {
    let len = f.len();
    Gguf::parse(f, len, PathBuf::new())
  }

  #[test]
  fn reads_metadata_and_tensors() {
    let g = parse(sample()).unwrap();
    assert_eq!(g.get::<u32>("a.n").unwrap(), 7);
    assert_eq!(g.get::<i32>("a.n").unwrap(), 7);
    assert!(g.get::<u32>("missing").is_err());
    assert!(g.get::<&str>("a.n").is_err());
    assert_eq!(g.array::<&str>("a.tokens").unwrap(), ["▁x", "y"]);
    assert!(g.get::<bool>("a.flag").unwrap());
    assert_eq!(g.optional::<bool>("a.flag").unwrap(), Some(true));
    assert_eq!(g.optional::<bool>("missing").unwrap(), None);
    assert!(g.optional::<&str>("a.flag").is_err());
    assert_eq!(g.get::<i32>("a.left").unwrap(), -1);
    assert!(g.get::<u32>("a.left").is_err());
    assert_eq!(g.array::<i32>("a.durations").unwrap(), [0, 2, -4]);
    assert!(g.array::<u32>("a.durations").is_err());
    assert_eq!(g.optional_array::<i32>("a.durations").unwrap(), [0, 2, -4]);
    assert!(g.optional_array::<i32>("missing").unwrap().is_empty());
    assert!(g.optional_array::<u32>("a.durations").is_err());
    assert_eq!(g.get::<f32>("a.alpha").unwrap(), 0.97);

    assert_eq!(g.tensor("w", &[2, 1]).unwrap().to_f32().unwrap(), [1.5, -2.0]);
    assert!(g.tensor("w", &[1, 2]).is_err());

    let q = g.tensor("q", &[32]).unwrap();
    assert_eq!(q.ty, Type::Q8_0);
    let expected: Vec<f32> = (0..32).map(|i| (i - 16) as f32 * 0.5).collect();
    assert_eq!(q.to_f32().unwrap(), expected);

    let mut pieces = Vec::new();
    q.read(16, |offset, piece| pieces.push((offset, piece.len()))).unwrap();
    assert_eq!(pieces, [(0, 16), (16, 16), (32, 2)]);
  }

  /// From a file, as models load on the GPU: through its memory map on macOS and Windows.
  #[test]
  fn reads_tensors_from_files() {
    let path = std::env::temp_dir().join(format!("wavo-gguf-{}.gguf", std::process::id()));
    std::fs::write(&path, sample()).unwrap();
    let mut g = Gguf::open(&path).unwrap();
    g.map();
    assert_eq!(g.map.is_some(), cfg!(any(target_os = "macos", windows)));
    assert_eq!(g.get::<u32>("a.n").unwrap(), 7);
    assert_eq!(g.tensor("w", &[2, 1]).unwrap().to_f32().unwrap(), [1.5, -2.0]);
    let mut pieces = Vec::new();
    let q = g.tensor("q", &[32]).unwrap();
    q.read(16, |offset, piece| pieces.push((offset, piece.to_vec()))).unwrap();
    assert_eq!(
      pieces.iter().map(|(o, p)| (*o, p.len())).collect::<Vec<_>>(),
      [(0, 16), (16, 16), (32, 2)]
    );
    assert_eq!(pieces[0].1[..2], f16::from_f32(0.5).to_le_bytes());
    drop(g);
    std::fs::remove_file(&path).unwrap();
  }

  #[test]
  fn rejects_truncated_files() {
    let mut f = sample();
    f.truncate(f.len() - 1);
    assert!(parse(f).is_err());
  }
}
