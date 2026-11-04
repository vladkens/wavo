// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! What keeps a model off the GPU, as codes, for apps to explain in their own words.

/// What keeps the GPU from running models, when the person can fix it: a code for an app to show
/// its own message for, in any language, with [`command`](Self::command) as the fix. `Display`
/// gives one short English line for logs and the CLI.
///
/// [`gpu_problem`] tells it before any GPU is touched,
/// [`Device::gpu_problem`](crate::Device::gpu_problem) for an adapter that is a software GPU, and
/// [`Error::gpu_problem`](crate::Error::gpu_problem) for a load that found no adapter. It is
/// `#[non_exhaustive]`: new codes come in minor releases, so keep a `_` arm that falls back to
/// `Display`.
///
/// # Examples
///
/// ```no_run
/// let model = wavo::Model::load("parakeet-tdt-0.6b-v3-Q8_0.gguf")?;
/// if let Some(problem) = model.device().gpu_problem() {
///   let text = match problem {
///     wavo::GpuProblem::NoDriver => "Transcription will be slow: install your GPU's driver".into(),
///     other => other.to_string(),
///   };
///   println!("{text}");
///   if let Some(command) = problem.command() {
///     println!("run: {command}");
///   }
/// }
/// # Ok::<(), wavo::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuProblem {
  /// The GPU's driver is missing: no adapter at all, a GPU emulated on the CPU
  /// ([`DeviceKind::SoftwareGpu`](crate::DeviceKind::SoftwareGpu): llvmpipe on Linux, WARP on
  /// Windows, 10–50× slower), or on Linux no `/dev/dri` render node (no kernel driver, or a
  /// container started without `--device /dev/dri`) and no NVIDIA proprietary driver
  /// (`/dev/nvidiactl`). On Linux the fix is a Vulkan driver, on Windows the GPU maker's driver.
  NoDriver,
  /// Linux: the render nodes (`/dev/dri/renderD*`) exist, but this user may open none of them, so
  /// Vulkan falls back to a software GPU or finds none: the user needs the `render` group.
  NoAccess,
}

impl GpuProblem {
  /// The shell command that fixes it, when there is one, the same in every language:
  /// `sudo usermod -aG render,video $USER` for [`NoAccess`](Self::NoAccess) (then log in again),
  /// and for [`NoDriver`](Self::NoDriver) the command that installs the Vulkan driver on this
  /// Linux distribution, read from `/etc/os-release`: Debian, Ubuntu, Fedora, Arch, openSUSE and
  /// their derivatives. `NoDriver` has none for another distribution, inside a container (the
  /// host's driver serves it there: pass `--device /dev/dri`), and on Windows and macOS.
  ///
  /// # Examples
  ///
  /// ```
  /// let command = wavo::GpuProblem::NoAccess.command();
  /// assert_eq!(command.as_deref(), Some("sudo usermod -aG render,video $USER"));
  /// ```
  pub fn command(&self) -> Option<String> {
    let release = || std::fs::read_to_string("/etc/os-release").ok().filter(|_| !container());
    match self {
      Self::NoAccess => Some(ACCESS.into()),
      Self::NoDriver => installer(&release()?).map(Into::into),
    }
  }
}

impl std::fmt::Display for GpuProblem {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    let missing = "the GPU's Vulkan driver is missing";
    match self {
      Self::NoAccess => write!(f, "this user can't open the GPU: {ACCESS}, then log in again"),
      // Windows needs no Vulkan: the maker's driver brings it, and DX12 runs without.
      _ if cfg!(windows) => f.write_str("install the GPU's driver from NVIDIA, AMD or Intel"),
      _ if cfg!(target_os = "macos") => f.write_str("no Metal GPU (in a virtual machine?)"),
      _ if container() => write!(f, "{missing}: pass --device /dev/dri to the container"),
      _ => write!(f, "{missing}: {}", self.command().as_deref().unwrap_or("install it")),
    }
  }
}

/// [`GpuProblem::NoAccess`]'s command.
const ACCESS: &str = "sudo usermod -aG render,video $USER";

/// What keeps the GPU from running models, from files alone: no graphics API is called and no
/// device opened, so a sleeping discrete GPU stays asleep, and it takes microseconds. It never
/// fails: what it can't read counts as no problem.
///
/// On Linux, [`GpuProblem::NoDriver`] without a `/dev/dri/renderD*` render node, and
/// [`GpuProblem::NoAccess`] when this user may open none of them (by their permissions and ACLs).
/// `None` when they're fine, when `/dev/dri` can't be listed, under WSL (`/dev/dxg`) and with
/// NVIDIA's proprietary driver (`/dev/nvidiactl`), which reach the GPU another way, and on every
/// other OS: a missing driver there shows only once wgpu looks
/// for an adapter, in [`Device::gpu_problem`](crate::Device::gpu_problem) and
/// [`Error::gpu_problem`](crate::Error::gpu_problem).
pub fn gpu_problem() -> Option<GpuProblem> {
  #[cfg(target_os = "linux")]
  return render_nodes(std::path::Path::new("/dev"));
  #[cfg(not(target_os = "linux"))]
  None
}

/// Whether wavo runs in a Docker or Podman container, whose GPU comes from the host.
fn container() -> bool {
  ["/.dockerenv", "/run/.containerenv"].iter().any(|p| std::path::Path::new(p).exists())
}

/// The command that installs the Mesa Vulkan drivers for the distribution an `/etc/os-release`
/// names in `ID`, or failing that in `ID_LIKE`.
fn installer(release: &str) -> Option<&'static str> {
  let field = |key: &str| release.lines().find_map(|l| l.strip_prefix(key)).unwrap_or_default();
  let ids = [field("ID="), field("ID_LIKE=")].join(" ");
  ids.split([' ', '"', '\'']).find_map(|id| match id {
    "debian" | "ubuntu" => Some("sudo apt install mesa-vulkan-drivers"),
    "fedora" => Some("sudo dnf install mesa-vulkan-drivers"),
    // One package per GPU maker: Mesa's three, which Debian's and Fedora's one package holds.
    "arch" => Some("sudo pacman -S vulkan-intel vulkan-radeon vulkan-nouveau"),
    // Leap 15 has no libvulkan_nouveau.
    "suse" | "opensuse" => Some("sudo zypper install libvulkan_intel libvulkan_radeon"),
    _ => None,
  })
}

/// [`gpu_problem`] on Linux, with `/dev` at `dev`. `access` checks the permission bits and the ACLs
/// (logind grants the seat's user one) without opening the node.
#[cfg(any(target_os = "linux", all(unix, test)))]
fn render_nodes(dev: &std::path::Path) -> Option<GpuProblem> {
  use std::ffi::{CString, c_char, c_int};

  unsafe extern "C" {
    fn access(path: *const c_char, mode: c_int) -> c_int;
  }

  if dev.join("dxg").exists() || dev.join("nvidiactl").exists() {
    return None;
  }

  let nodes = match std::fs::read_dir(dev.join("dri")) {
    Ok(dir) => dir.flatten().filter(|e| e.file_name().to_string_lossy().starts_with("renderD")),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(GpuProblem::NoDriver),
    Err(_) => return None,
  };

  let paths = nodes.map(|e| e.path().into_os_string().into_encoded_bytes());
  let paths: Vec<_> = paths.filter_map(|p| CString::new(p).ok()).collect();
  if paths.is_empty() {
    return Some(GpuProblem::NoDriver);
  }

  // SAFETY: each path is a NUL-terminated string that outlives its call. 6: R_OK | W_OK.
  let open = paths.iter().any(|p| unsafe { access(p.as_ptr(), 6) } == 0);
  (!open).then_some(GpuProblem::NoAccess)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(unix)]
  fn render_nodes() {
    use std::os::unix::fs::PermissionsExt;

    let dev = std::env::temp_dir().join(format!("wavo-dev-{}", std::process::id()));
    let dri = dev.join("dri");
    std::fs::create_dir_all(&dev).unwrap();
    assert_eq!(super::render_nodes(&dev), Some(GpuProblem::NoDriver));
    // NVIDIA's own driver and WSL reach the GPU without a render node.
    for other in ["nvidiactl", "dxg"] {
      std::fs::write(dev.join(other), "").unwrap();
      assert_eq!(super::render_nodes(&dev), None, "{other}");
      std::fs::remove_file(dev.join(other)).unwrap();
    }

    std::fs::create_dir_all(&dri).unwrap();
    std::fs::write(dri.join("card0"), "").unwrap();
    assert_eq!(super::render_nodes(&dev), Some(GpuProblem::NoDriver));

    let node = dri.join("renderD128");
    std::fs::write(&node, "").unwrap();
    assert_eq!(super::render_nodes(&dev), None);

    let mode = |path: &std::path::Path, mode| {
      std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap()
    };
    mode(&node, 0o400);
    // Root may write it anyway.
    if std::fs::OpenOptions::new().write(true).open(&node).is_err() {
      assert_eq!(super::render_nodes(&dev), Some(GpuProblem::NoAccess));
      mode(&dri, 0o300); // can't be listed
      assert_eq!(super::render_nodes(&dev), None);
      mode(&dri, 0o700);
    }

    std::fs::remove_dir_all(&dev).unwrap();
  }

  #[test]
  fn installers() {
    let apt = Some("sudo apt install mesa-vulkan-drivers");
    let cases = [
      ("ID=debian\nVERSION_ID=\"13\"\n", apt),
      ("NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian\n", apt),
      ("ID=linuxmint\nID_LIKE=\"ubuntu debian\"\n", apt),
      ("ID=fedora\n", Some("sudo dnf install mesa-vulkan-drivers")),
      (
        "ID=\"rocky\"\nID_LIKE=\"rhel centos fedora\"\n",
        Some("sudo dnf install mesa-vulkan-drivers"),
      ),
      (
        "ID=endeavouros\nID_LIKE=arch\n",
        Some("sudo pacman -S vulkan-intel vulkan-radeon vulkan-nouveau"),
      ),
      (
        "ID=\"opensuse-tumbleweed\"\nID_LIKE=\"opensuse suse\"\n",
        Some("sudo zypper install libvulkan_intel libvulkan_radeon"),
      ),
      ("ID=alpine\n", None),
      ("ID_LIKE=gentoo\n", None),
      ("", None),
    ];
    for (release, want) in cases {
      assert_eq!(installer(release), want, "{release:?}");
    }
  }

  #[test]
  fn messages() {
    let vulkan = |p: GpuProblem| p.to_string().contains("Vulkan");
    assert_eq!(vulkan(GpuProblem::NoDriver), cfg!(target_os = "linux"));
    assert!(!vulkan(GpuProblem::NoAccess));
  }
}
