// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Setup problems as codes, for apps to explain in their own words.

use crate::{Device, DeviceKind, Error};

/// A problem with this machine's setup that the person can fix, or the fixable cause behind a
/// failed load. An app matches the code and shows its own message; `Display` gives wavo's default
/// English one with the fix, which `wavo run`, `wavo serve` and `wavo debug` print: it advises a
/// Vulkan driver on Linux only, and names no device.
///
/// The codes carry no data, so they are cheap to match, compare and send between processes: the
/// device's name and driver come from [`Model::device`](crate::Model::device) or
/// [`devices`](crate::devices). It is `#[non_exhaustive]`: new codes come in minor releases, so
/// keep a `_` arm that falls back to `Display`.
///
/// # Examples
///
/// ```no_run
/// let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
/// if let Some(issue) = model.device().issue() {
///   let text = match issue {
///     wavo::Issue::SoftwareGpu => "Transcription will be slow: install your GPU's driver".into(),
///     other => other.to_string(),
///   };
///   println!("{}: {text}", model.device().name);
///   for cause in wavo::issues() {
///     println!("{cause}"); // on Linux, why the GPU may be hidden
///   }
/// }
/// # Ok::<(), wavo::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Issue {
  /// No GPU adapter at all, so no model loads on the GPU (the Neural Engine may still run
  /// Parakeet and GigaAM): on Linux usually no Vulkan driver, on Windows no GPU driver, on macOS
  /// no Metal, as in some virtual machines.
  /// [`devices`](crate::devices) and [`Model::load`](crate::Model::load) fail with
  /// [`Error::Adapter`].
  NoGpu,
  /// The adapter is a GPU emulated on the CPU ([`DeviceKind::SoftwareGpu`]: llvmpipe on Linux,
  /// WARP on Windows). Models load and run, 10–50× slower than on a real GPU.
  SoftwareGpu,
  /// Linux: the GPU's device files (`/dev/dri/renderD*`) exist, but this user may open none of
  /// them, so Vulkan falls back to a software GPU or finds none: the cause behind a
  /// [`SoftwareGpu`](Self::SoftwareGpu) or [`NoGpu`](Self::NoGpu).
  NoGpuAccess,
  /// Linux: no `/dev/dri/renderD*` at all: no kernel GPU driver is loaded, or wavo runs in a
  /// container started without `--device /dev/dri`: the cause behind a
  /// [`SoftwareGpu`](Self::SoftwareGpu) or [`NoGpu`](Self::NoGpu).
  NoGpuDevice,
}

impl std::fmt::Display for Issue {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    // Vulkan on Linux only: on Windows the GPU's driver brings it, and DX12 needs none.
    let (linux, mac) = (cfg!(target_os = "linux"), cfg!(target_os = "macos"));
    let driver = if linux { "install the GPU's Vulkan driver" } else { "install the GPU's driver" };
    let mesa = " (mesa-vulkan-drivers on Debian and Ubuntu) and check it with vulkaninfo";
    let slow = "a software GPU, emulated on the CPU: expect 10-50x slower transcription";
    let usermod = "sudo usermod -aG render,video $USER, then log in again";
    let container = "no GPU driver loaded, or a container without --device /dev/dri";
    match self {
      Self::NoGpu if mac => f.write_str("no Metal GPU (in a virtual machine?)"),
      Self::NoGpu => write!(f, "no GPU found: {driver}{}", if linux { mesa } else { "" }),
      Self::SoftwareGpu => write!(f, "{slow}; {driver}"),
      Self::NoGpuAccess => write!(f, "no permission to open /dev/dri/renderD*: {usermod}"),
      Self::NoGpuDevice => write!(f, "no /dev/dri/renderD*: {container}"),
    }
  }
}

/// The setup problems this machine shows without touching a GPU, the likeliest cause first; empty
/// when there is none, and always empty outside Linux. On Linux it returns
/// [`Issue::NoGpuDevice`] or [`Issue::NoGpuAccess`]: it lists `/dev/dri` and checks the render
/// nodes' permissions without opening them, so no graphics API is called, a sleeping discrete GPU
/// stays asleep, and it takes microseconds. It never fails: what it can't read counts as no
/// problem.
///
/// They explain a [`SoftwareGpu`](Issue::SoftwareGpu) or a failed load; with a working GPU they
/// may be noise (a GPU reached another way, such as WSL's `/dev/dxg`), so show them alongside
/// such a problem: [`Device::issue`] for the adapter [`devices`](crate::devices) or
/// [`Model::device`](crate::Model::device) gives, [`Error::issue`] for a failed load.
pub fn issues() -> Vec<Issue> {
  #[cfg(target_os = "linux")]
  return render_nodes(std::path::Path::new("/dev/dri")).into_iter().collect();
  #[cfg(not(target_os = "linux"))]
  Vec::new()
}

impl Device {
  /// [`Issue::SoftwareGpu`] for a GPU emulated on the CPU, `None` otherwise. Costs nothing: the
  /// device is already known, from a loaded model or from [`devices`](crate::devices), which
  /// enumerates the adapters and may wake a discrete GPU on a laptop with two.
  pub fn issue(&self) -> Option<Issue> {
    (self.kind == DeviceKind::SoftwareGpu).then_some(Issue::SoftwareGpu)
  }
}

impl Error {
  /// The fixable cause behind this error, for an app to explain it in its own words:
  /// [`Error::Adapter`] gives [`Issue::NoGpu`], or on Linux the first of [`issues`] when there is
  /// one. `None` for the rest: errors with no setup cause, such as [`Error::Cancelled`] or
  /// [`Error::AudioTooLong`], and [`Error::Device`], whose cause wgpu doesn't tell (out of memory,
  /// a lost device or a driver fault, more often than limits the GPU lacks).
  ///
  /// # Examples
  ///
  /// ```no_run
  /// match wavo::Model::load("gigaam-v3-e2e-rnnt-Q8_0.gguf") {
  ///   Ok(model) => println!("on {}", model.device()),
  ///   Err(e) => match e.issue() {
  ///     Some(issue) => eprintln!("{issue}"),
  ///     None => eprintln!("can't load the model: {e}"),
  ///   },
  /// }
  /// ```
  pub fn issue(&self) -> Option<Issue> {
    let adapter = matches!(self, Self::Adapter(_));
    adapter.then(|| issues().into_iter().next().unwrap_or(Issue::NoGpu))
  }
}

/// Why Linux may hide the GPU, from the render nodes in `dir`: none at all (or no `dir`), or none
/// this user may read and write; `None` when `dir` can't be listed. `access` checks the permission
/// bits and the ACLs (logind grants the seat's user one) without opening the node.
#[cfg(any(target_os = "linux", all(unix, test)))]
fn render_nodes(dir: &std::path::Path) -> Option<Issue> {
  unsafe extern "C" {
    fn access(path: *const std::ffi::c_char, mode: std::ffi::c_int) -> std::ffi::c_int;
  }

  let nodes = match std::fs::read_dir(dir) {
    Ok(dir) => dir.flatten().filter(|e| e.file_name().to_string_lossy().starts_with("renderD")),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(Issue::NoGpuDevice),
    Err(_) => return None,
  };
  let paths = nodes.map(|e| e.path().into_os_string().into_encoded_bytes());
  let paths: Vec<_> = paths.filter_map(|p| std::ffi::CString::new(p).ok()).collect();
  if paths.is_empty() {
    return Some(Issue::NoGpuDevice);
  }

  // SAFETY: each path is a NUL-terminated string that outlives its call. 6: R_OK | W_OK.
  let open = paths.iter().any(|p| unsafe { access(p.as_ptr(), 6) } == 0);
  (!open).then_some(Issue::NoGpuAccess)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(unix)]
  fn render_nodes() {
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!("wavo-dri-{}", std::process::id()));
    assert_eq!(super::render_nodes(&dir), Some(Issue::NoGpuDevice));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("card0"), "").unwrap();
    assert_eq!(super::render_nodes(&dir), Some(Issue::NoGpuDevice));

    let node = dir.join("renderD128");
    std::fs::write(&node, "").unwrap();
    assert_eq!(super::render_nodes(&dir), None);

    let mode = |path: &std::path::Path, mode| {
      std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap()
    };
    mode(&node, 0o400);
    // Root may write it anyway.
    if std::fs::OpenOptions::new().write(true).open(&node).is_err() {
      assert_eq!(super::render_nodes(&dir), Some(Issue::NoGpuAccess));
      mode(&dir, 0o300); // can't be listed
      assert_eq!(super::render_nodes(&dir), None);
      mode(&dir, 0o700);
    }

    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn messages() {
    use Issue::*;

    for issue in [NoGpu, SoftwareGpu, NoGpuAccess, NoGpuDevice] {
      let driver = matches!(issue, NoGpu | SoftwareGpu);
      let vulkan = issue.to_string().contains("Vulkan");
      assert_eq!(vulkan, driver && cfg!(target_os = "linux"), "{issue:?}");
    }
  }
}
