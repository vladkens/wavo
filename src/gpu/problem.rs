// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! What keeps a model off the GPU, as a code, for apps to explain in their own words.

/// What keeps the GPU from running models, when the person can fix it: a code for an app to show
/// its own message for, in any language, with [`command`](Self::command) as the fix. `Display`
/// gives one short English line for logs and the CLI.
///
/// [`Device::gpu_problem`](crate::Device::gpu_problem) tells it for an adapter that is a software
/// GPU, and [`Error::gpu_problem`](crate::Error::gpu_problem) for a load that found no adapter.
/// It is `#[non_exhaustive]`: new codes come in minor releases, so keep a `_` arm that falls back
/// to `Display`.
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
  /// The GPU's driver is missing, or on Linux this user can't open the GPU (not in the `render`
  /// group): no adapter at all, or a GPU emulated on the CPU
  /// ([`DeviceKind::SoftwareGpu`](crate::DeviceKind::SoftwareGpu): llvmpipe on Linux, WARP on
  /// Windows, 10–50× slower). On Linux the fix is a Vulkan driver, on Windows the GPU maker's
  /// driver.
  NoDriver,
}

impl GpuProblem {
  /// The shell command that fixes it, when there is one, the same in every language: on Linux the
  /// command that installs the Vulkan driver on this distribution, read from `/etc/os-release`
  /// (Debian, Ubuntu, Fedora, Arch, openSUSE and their derivatives). `None` for another
  /// distribution, and on Windows and macOS.
  pub fn command(&self) -> Option<String> {
    if !cfg!(target_os = "linux") {
      return None;
    }

    installer(&std::fs::read_to_string("/etc/os-release").ok()?).map(Into::into)
  }
}

impl std::fmt::Display for GpuProblem {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    if cfg!(windows) {
      // Windows needs no Vulkan: the maker's driver brings it, and DX12 runs without.
      return f.write_str("install the GPU's driver from NVIDIA, AMD or Intel");
    }

    if cfg!(target_os = "macos") {
      return f.write_str("no Metal GPU (in a virtual machine?)");
    }

    let fix = self.command().unwrap_or_else(|| "install it".into());
    let check = "check vulkaninfo and that you're in the render group";
    write!(f, "the GPU's Vulkan driver is missing or not accessible: {fix}; {check}")
  }
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn installers() {
    let apt = Some("sudo apt install mesa-vulkan-drivers");
    let dnf = Some("sudo dnf install mesa-vulkan-drivers");
    let cases = [
      ("ID=debian\nVERSION_ID=\"13\"\n", apt),
      ("NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian\n", apt),
      ("ID=linuxmint\nID_LIKE=\"ubuntu debian\"\n", apt),
      ("ID=fedora\n", dnf),
      ("ID=\"rocky\"\nID_LIKE=\"rhel centos fedora\"\n", dnf),
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
    let vulkan = GpuProblem::NoDriver.to_string().contains("Vulkan");
    assert_eq!(vulkan, cfg!(target_os = "linux"));
  }
}
