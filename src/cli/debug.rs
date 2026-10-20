// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! `wavo debug`: the machine, its GPU adapters and the downloaded models, to paste into a bug
//! report. It reads the adapters' info and features only: creating a device or compiling a kernel
//! could crash on a broken driver, in the very report about it. A section that fails prints
//! `error=…` and the report goes on.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::Result;
use wavo::hub::Hub;
use wavo::{Device, DeviceKind, Kernels};
use wgpu::{CooperativeMatrixProperties as Matrix, CooperativeScalarType as Scalar, Features};

/// How long a system command (sysctl, ioreg, lscpu, PowerShell) may take.
const TIMEOUT: Duration = Duration::from_secs(3);

pub fn print() -> Result<()> {
  let mut w = std::io::stdout().lock();
  let (os, machine) = system();
  writeln!(w, "wavo {} | OS: {os} | {}", env!("CARGO_PKG_VERSION"), std::env::consts::ARCH)?;
  section(&mut w, "Machine", |w| {
    for (key, value) in &machine {
      writeln!(w, "{key:<18} {value}")?;
    }

    writeln!(w, "{:<18} {}", "cpu features", cpu_features())?;
    Ok(writeln!(w, "{:<18} {}", "cpu kernels", wavo::cpu_path())?)
  })?;

  // Out before the adapters are listed: a broken driver can crash the process there.
  w.flush()?;
  let warnings = section(&mut w, "GPU", gpu)?.unwrap_or_default();
  section(&mut w, "Models", models)?;
  for (i, x) in warnings.iter().enumerate() {
    writeln!(w, "{}warning: {x}", if i == 0 { "\n" } else { "" })?;
  }

  Ok(())
}

/// A section's divider and body; a body that fails prints `error=…` instead of the rest, unless
/// stdout itself failed (a closed pipe).
fn section<W: Write, T>(
  w: &mut W,
  name: &str,
  body: impl FnOnce(&mut W) -> Result<T>,
) -> Result<Option<T>> {
  writeln!(w, "\n--- {name} {}", "-".repeat(75usize.saturating_sub(name.len())))?;
  match body(w) {
    Ok(x) => Ok(Some(x)),
    Err(e) if e.chain().any(|x| x.is::<std::io::Error>()) => Err(e),
    Err(e) => writeln!(w, "error={e:#}").map(|_| None).map_err(Into::into),
  }
}

/// The adapters, and the warnings for the one wavo uses.
fn gpu(w: &mut impl Write) -> Result<Vec<String>> {
  // vars_os: `vars` panics on any variable that isn't UTF-8.
  for (key, value) in std::env::vars_os() {
    let (key, value) = (key.to_string_lossy(), value.to_string_lossy());
    if key.starts_with("WGPU_") {
      writeln!(w, "{:<18} {key}={value}", "env")?;
    }
  }

  let devices = match wavo::devices() {
    Ok(devices) => devices,
    Err(e) => {
      writeln!(w, "error={e}")?;
      return Ok(vec!["no GPU adapter found: wavo can't load a model".into()]);
    }
  };

  adapters(w, &devices)?;
  let mut warnings = Vec::new();
  if let Some(d) = devices.first().filter(|d| d.kind == DeviceKind::SoftwareGpu) {
    warnings.extend(software(d));
    #[cfg(target_os = "linux")]
    warnings.extend(dri());
  }

  Ok(warnings)
}

fn models(w: &mut impl Write) -> Result<()> {
  let hub = Hub::new();
  writeln!(w, "{:<18} {hub}", "cache")?;
  for (m, path) in crate::downloaded(&hub) {
    let size = std::fs::metadata(path).map_or("-".into(), |x| crate::models::gb(x.len()));
    writeln!(w, "{:<18} {size}", m.name)?;
  }

  Ok(())
}

/// The warning for a device emulated on the CPU, for `wavo run` too.
pub fn software(d: &Device) -> Option<String> {
  let msg = "is a software GPU, emulated on the CPU: expect 10-50x slower transcription";
  let fix = "install the GPU's Vulkan driver";
  (d.kind == DeviceKind::SoftwareGpu).then(|| format!("{} {msg}; {fix}", d.name))
}

/// Each device with the features and limits its kernels depend on; the first is wavo's.
fn adapters(w: &mut impl Write, devices: &[Device]) -> Result<()> {
  let mut desc = wgpu::InstanceDescriptor::new_without_display_handle().with_env();
  desc.backends &= wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
  let backends = desc.backends;
  let mut raw = pollster::block_on(wgpu::Instance::new(desc).enumerate_adapters(backends));
  for (i, d) in devices.iter().enumerate() {
    writeln!(w, "[{i}] {d}{}", if i == 0 { "  <- wavo uses this" } else { "" })?;
    let mut rows =
      vec![("driver", if d.driver.is_empty() { "-".into() } else { d.driver.clone() })];
    // The same adapter in wgpu's list: its name, driver (two Vulkan drivers may share a GPU) and
    // API, which both name alike (`Dx12`).
    let api = format!("{:?}", d.api);
    let same = |a: &wgpu::Adapter| {
      let info = a.get_info();
      let driver = format!("{} {}", info.driver.trim(), info.driver_info.trim());
      info.name == d.name && driver.trim() == d.driver && format!("{:?}", info.backend) == api
    };
    if let Some(a) = raw.iter().position(same).map(|i| raw.remove(i)) {
      rows.extend(facts(d, &a));
    }

    for (key, value) in rows {
      writeln!(w, "  {key:<16} {value}")?;
    }
  }

  Ok(())
}

/// The rows of an adapter's kernel tier and the features and limits it depends on.
fn facts(d: &Device, a: &wgpu::Adapter) -> Vec<(&'static str, String)> {
  let (info, limits, features) = (a.get_info(), a.limits(), a.features());
  let subgroups = [info.subgroup_min_size, info.subgroup_max_size];
  let (vendor, matrices) = (info.vendor, a.cooperative_matrix_properties());
  let facts = Facts { vendor, features, subgroups, matrices };
  let has = |f, value: String| if features.contains(f) { value } else { "no".into() };
  let yes = |f| has(f, "yes".into());
  let [min, max] = subgroups;
  let barriers = yes(Features::SUBGROUP_BARRIER);
  let shared = bytes(limits.max_compute_workgroup_storage_size.into());
  let invocations = limits.max_compute_invocations_per_workgroup;
  let binding = bytes(limits.max_storage_buffer_binding_size);
  vec![
    ("ids", format!("vendor 0x{:04x}, device 0x{:04x}", info.vendor, info.device)),
    ("kernels", kernels(d.kernels, d.kind == DeviceKind::SoftwareGpu, &facts)),
    ("subgroups", has(Features::SUBGROUP, format!("{min}..{max}, barriers {barriers}"))),
    ("shader-f16", yes(Features::SHADER_F16)),
    ("coop matrix", has(Features::EXPERIMENTAL_COOPERATIVE_MATRIX, shapes(&facts.matrices))),
    ("passthrough", yes(Features::PASSTHROUGH_SHADERS)),
    ("mappable", yes(Features::MAPPABLE_PRIMARY_BUFFERS)),
    ("workgroup", format!("{invocations} invocations, {shared} shared")),
    ("buffer", format!("{}, storage binding {binding}", bytes(limits.max_buffer_size))),
    ("immediates", has(Features::IMMEDIATES, bytes(limits.max_immediate_size.into()))),
  ]
}

/// What the kernel tiers depend on, from an adapter's info and features.
struct Facts {
  vendor: u32,
  features: Features,
  subgroups: [u32; 2],
  matrices: Vec<Matrix>,
}

/// The kernels, on a CPU adapter or not, and why not a faster tier: the first need of wavo's choice
/// (in `src/gpu/mod.rs`) the adapter misses.
fn kernels(kernels: Kernels, cpu: bool, f: &Facts) -> String {
  let has = |n, ab| {
    let shape = (n, n, n, ab, Scalar::F32);
    f.matrices.iter().any(|p| (p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type) == shape)
  };
  let needs = [
    (Features::SUBGROUP, "subgroups"),
    (Features::SUBGROUP_BARRIER, "subgroup barriers"),
    (Features::SHADER_F16, "shader-f16"),
    (Features::EXPERIMENTAL_COOPERATIVE_MATRIX, "cooperative matrices"),
  ];
  let missing: Vec<_> =
    needs.iter().filter(|(x, _)| !f.features.contains(*x)).map(|x| x.1).collect();
  let [min, max] = f.subgroups;
  let (tier, why) = match kernels {
    Kernels::Tensor => ("8x8 matrices, Metal 4 tensor GEMM", None),
    Kernels::Matrix16 => ("16x16 matrices", None),
    Kernels::Matrix8 if !f.features.contains(Features::PASSTHROUGH_SHADERS) => {
      ("8x8 matrices", Some("no passthrough shaders for the tensor GEMM".into()))
    }
    Kernels::Matrix8 => {
      ("8x8 matrices", Some("the tensor GEMM is for Apple M5 / A19 and newer".into()))
    }
    _ if cpu => ("portable, CPU tiles", None),
    _ if !missing.is_empty() => ("portable", Some(format!("no {}", missing.join(", ")))),
    _ if !(min..=max).contains(&32) => {
      ("portable", Some("the fast kernels need 32-lane subgroups".into()))
    }
    _ if has(8, Scalar::F32) && has(8, Scalar::F16) => ("portable", None),
    _ if has(16, Scalar::F16) => (
      "portable",
      Some(format!("16x16 matrices run on NVIDIA only so far, vendor 0x{:04x}", f.vendor)),
    ),
    _ => ("portable", Some("no 8x8 or 16x16 matrices into f32".into())),
  };
  why.map_or(tier.into(), |why: String| format!("{tier}: {why}"))
}

/// `8x8x8 f16>f32, ...`: M×N×K, the type of A and B, then of C.
fn shapes(matrices: &[Matrix]) -> String {
  let ty = |x: Scalar| format!("{x:?}").to_lowercase();
  let all = matrices
    .iter()
    .map(|p| format!("{}x{}x{} {}>{}", p.m_size, p.n_size, p.k_size, ty(p.ab_type), ty(p.cr_type)));
  let all: Vec<_> = all.collect();
  if all.is_empty() { "none".into() } else { all.join(", ") }
}

/// The SIMD features the CPU reports at run time, of those that matter for matrix products.
fn cpu_features() -> String {
  #[cfg(target_arch = "aarch64")]
  let all = {
    use std::arch::is_aarch64_feature_detected as has;
    let all = [("neon", has!("neon")), ("fp16", has!("fp16")), ("dotprod", has!("dotprod"))];
    [all.to_vec(), vec![("i8mm", has!("i8mm")), ("sve", has!("sve"))]].concat()
  };
  #[cfg(target_arch = "x86_64")]
  let all = {
    use std::arch::is_x86_feature_detected as has;
    let all = [("sse4.2", has!("sse4.2")), ("avx", has!("avx")), ("avx2", has!("avx2"))];
    let more = [("fma", has!("fma")), ("f16c", has!("f16c")), ("avx512f", has!("avx512f"))];
    let wide = [("avx512bw", has!("avx512bw")), ("avx512vl", has!("avx512vl"))];
    [&all[..], &more, &wide, &[("avx-vnni", has!("avxvnni"))]].concat()
  };
  #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
  let all: Vec<(&str, bool)> = Vec::new();
  let all: Vec<_> = all.iter().filter(|x| x.1).map(|x| x.0).collect();
  if all.is_empty() { "-".into() } else { all.join(" ") }
}

/// `24.0 GiB`, `32.0 KiB`, `256 B`.
fn bytes(n: u64) -> String {
  let units = [(1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")];
  match units.into_iter().find(|(size, _)| n >= *size) {
    Some((size, unit)) => format!("{:.1} {unit}", n as f64 / size as f64),
    None => format!("{n} B"),
  }
}

/// The value of `key` in `key: value`, `key = value` or `key="value"` lines: sysctl, ioreg,
/// /proc files, os-release and the PowerShell script's.
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
  text.lines().find_map(|line| {
    let rest = line.trim_start().strip_prefix(key)?.trim_start();
    let value = rest.strip_prefix(':').or_else(|| rest.strip_prefix('='))?;
    Some(value.trim().trim_matches('"'))
  })
}

/// A command's stdout, in English. An error when it can't start or runs past `timeout`, which
/// kills it; its exit status doesn't count (sysctl fails on a key it lacks, after the others).
fn output(cmd: &str, args: &[&str], timeout: Duration) -> Result<String, String> {
  use std::process::{Command, Stdio};
  let mut child = (Command::new(cmd).args(args).env("LC_ALL", "C"))
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::null())
    .spawn()
    .map_err(|e| format!("{cmd}: {e}"))?;
  // Read on a thread, so a full pipe can't stall the command.
  let mut stdout = child.stdout.take().ok_or(format!("{cmd}: no stdout"))?;
  let (tx, rx) = std::sync::mpsc::channel();
  std::thread::spawn(move || {
    let mut out = Vec::new();
    let _ = std::io::Read::read_to_end(&mut stdout, &mut out);
    let _ = tx.send(out);
  });

  let start = Instant::now();
  while matches!(child.try_wait(), Ok(None)) && start.elapsed() < timeout {
    std::thread::sleep(Duration::from_millis(5));
  }

  let late = format!("{cmd}: timeout after {timeout:?}");
  if matches!(child.try_wait(), Ok(None)) {
    let _ = child.kill();
    let _ = child.wait();
    return Err(late);
  }

  // Its output may still be in flight, or held open by a child of its own.
  let out = rx.recv_timeout(timeout.saturating_sub(start.elapsed())).map_err(|_| late)?;
  Ok(String::from_utf8_lossy(&out).into_owned())
}

/// The value of `key` in a command's output: `-` when it lacks it, `error=…` when it failed.
#[cfg(any(target_os = "macos", windows, test))]
fn value(out: &Result<String, String>, key: &str) -> String {
  match out {
    Ok(text) => field(text, key).unwrap_or("-").into(),
    Err(e) => format!("error={e}"),
  }
}

#[cfg(not(target_os = "macos"))]
fn threads() -> String {
  std::thread::available_parallelism().map_or("-".into(), |n| format!("{n} threads"))
}

type System = (String, Vec<(&'static str, String)>);

/// The OS line and the machine's rows: model, CPU, memory.
#[cfg(target_os = "macos")]
fn system() -> System {
  let keys = "kern.osproductversion kern.osversion hw.model hw.memsize machdep.cpu.brand_string \
    hw.ncpu hw.perflevel0.physicalcpu hw.perflevel0.name hw.perflevel1.physicalcpu \
    hw.perflevel1.name hw.perflevel2.physicalcpu hw.perflevel2.name";
  let out = output("sysctl", &keys.split_whitespace().collect::<Vec<_>>(), TIMEOUT);
  let get = |key: &str| value(&out, key);
  // 4P + 4E: the cores of each performance level, by the level name's initial.
  let text = out.as_deref().unwrap_or_default();
  let tiers = (0..3).filter_map(|i| {
    let n = field(text, &format!("hw.perflevel{i}.physicalcpu"))?;
    Some(format!("{n}{}", field(text, &format!("hw.perflevel{i}.name"))?.get(..1)?))
  });
  let tiers = tiers.collect::<Vec<_>>().join(" + ");
  let cpu = format!("{}, {} cores ({tiers})", get("machdep.cpu.brand_string"), get("hw.ncpu"));
  let memory = get("hw.memsize");
  let gpu = output("ioreg", &["-rc", "AGXAccelerator", "-d1"], TIMEOUT);
  let rows = vec![
    ("model", get("hw.model")),
    ("cpu", cpu),
    ("memory", memory.parse().map_or(memory, bytes)),
    ("gpu cores", value(&gpu, "\"gpu-core-count\"")),
  ];
  (format!("macOS {} ({})", get("kern.osproductversion"), get("kern.osversion")), rows)
}

/// Linux, and other Unix systems as far as they share its files.
#[cfg(not(any(target_os = "macos", windows)))]
fn system() -> System {
  let read = |path: &str| std::fs::read_to_string(path).unwrap_or_default();
  let (release, kernel) = (read("/etc/os-release"), read("/proc/sys/kernel/osrelease"));
  let os = field(&release, "PRETTY_NAME").unwrap_or(std::env::consts::OS);
  // A board's device tree names it (Orange Pi 5); a PC's DMI its vendor and product.
  let model = match read("/sys/firmware/devicetree/base/model").trim_end_matches('\0') {
    "" => ["sys_vendor", "product_name"].map(|x| read(&format!("/sys/class/dmi/id/{x}"))).join(" "),
    tree => tree.into(),
  };
  // An ARM CPU has no model name in /proc/cpuinfo, but lscpu names its cores: ARM Cortex-A55 +
  // Cortex-A76.
  let cpu = match field(&read("/proc/cpuinfo"), "model name") {
    Some(cpu) => cpu.into(),
    None => match output("lscpu", &[], TIMEOUT) {
      Ok(lscpu) => {
        let cores = lscpu.lines().filter_map(|x| field(x, "Model name")).filter(|x| *x != "-");
        let cores = cores.collect::<Vec<_>>().join(" + ");
        format!("{} {cores}", field(&lscpu, "Vendor ID").unwrap_or("-")).trim().to_string()
      }
      Err(e) => format!("error={e}"),
    },
  };
  let memory = read("/proc/meminfo");
  let memory = field(&memory, "MemTotal").and_then(|x| x.trim_end_matches(" kB").parse().ok());
  let memory = memory.map_or("-".into(), |kb: u64| bytes(kb.saturating_mul(1024)));
  let rows = vec![
    ("model", model.replace('\n', "").trim().into()),
    ("cpu", format!("{cpu}, {}", threads())),
    ("memory", memory),
  ];
  (format!("{os}, kernel {}", kernel.trim()), rows)
}

#[cfg(windows)]
fn system() -> System {
  // Single quotes only: double ones would need escaping on powershell.exe's command line.
  let script = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; \
    $o = Get-CimInstance Win32_OperatingSystem; \
    $c = Get-CimInstance Win32_ComputerSystem; \
    'os=' + $o.Caption + ' ' + $o.Version; 'model=' + $c.Manufacturer + ' ' + $c.Model; \
    'cpu=' + @(Get-CimInstance Win32_Processor)[0].Name; 'memory=' + $c.TotalPhysicalMemory";
  let args = ["-NoProfile", "-NonInteractive", "-Command", script];
  let out = output("powershell", &args, TIMEOUT);
  let get = |key| value(&out, key);
  let memory = get("memory");
  let rows = vec![
    ("model", get("model")),
    ("cpu", format!("{}, {}", get("cpu"), threads())),
    ("memory", memory.parse().map_or(memory, bytes)),
  ];
  (get("os"), rows)
}

/// Why Linux may hide the GPU: no render node, or none this user may open.
#[cfg(target_os = "linux")]
fn dri() -> Option<String> {
  let nodes = std::fs::read_dir("/dev/dri").into_iter().flatten().flatten().map(|e| e.path());
  let nodes: Vec<_> =
    nodes.filter(|p| p.to_string_lossy().starts_with("/dev/dri/renderD")).collect();
  let Some(first) = nodes.first() else {
    return Some(
      "no /dev/dri/renderD*: no GPU driver loaded, or a container without --device /dev/dri".into(),
    );
  };

  let open = |p| std::fs::OpenOptions::new().read(true).write(true).open(p);
  let errors: Vec<_> = nodes.iter().map(open).filter_map(Result::err).collect();
  let err = errors.first().filter(|_| errors.len() == nodes.len())?;
  let fix = "sudo usermod -aG render,video $USER, then log in again";
  Some(format!("can't open {}: {err}; {fix}", first.display()))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fields() {
    let sysctl = "hw.model: Mac14,7\nhw.memsize: 25769803776\nhw.perflevel0.name: Performance";
    assert_eq!(field(sysctl, "hw.model"), Some("Mac14,7"));
    assert_eq!(field(sysctl, "hw.mem"), None);
    assert_eq!(field(sysctl, "hw.perflevel0.name"), Some("Performance"));
    let ioreg = "  | {\n    \"gpu-core-count\" = 10\n";
    assert_eq!(field(ioreg, "\"gpu-core-count\""), Some("10"));
    assert_eq!(field("model name\t: Intel(R) N100\n", "model name"), Some("Intel(R) N100"));
    assert_eq!(field("MemTotal:       16123456 kB\n", "MemTotal"), Some("16123456 kB"));
    let release = "NAME=x\nPRETTY_NAME=\"Ubuntu 24.04 LTS\"";
    assert_eq!(field(release, "PRETTY_NAME"), Some("Ubuntu 24.04 LTS"));
    assert_eq!(field("os=\r\nmemory=1\r\n", "os"), Some(""));
  }

  #[test]
  fn sizes() {
    assert_eq!(bytes(25769803776), "24.0 GiB");
    assert_eq!(bytes(4294967295), "4.0 GiB");
    assert_eq!(bytes(32768), "32.0 KiB");
    assert_eq!(bytes(256), "256 B");
  }

  #[test]
  #[cfg(unix)]
  fn commands() {
    assert_eq!(output("echo", &["a: b"], TIMEOUT), Ok("a: b\n".into()));
    let t = Instant::now();
    let slow = output("sleep", &["5"], Duration::from_millis(100));
    assert_eq!(slow, Err("sleep: timeout after 100ms".into()));
    assert!(t.elapsed() < Duration::from_secs(2));
    assert_eq!(value(&slow, "x"), "error=sleep: timeout after 100ms");
    assert!(output("wavo-no-such-command", &[], TIMEOUT).is_err());
  }

  #[test]
  fn sections() {
    let mut out = Vec::new();
    let failed = section(&mut out, "GPU", |_| -> Result<()> { anyhow::bail!("no driver") });
    assert!(matches!(failed, Ok(None)));
    assert!(String::from_utf8_lossy(&out).ends_with("---\nerror=no driver\n"));

    // A closed pipe ends the report, quietly.
    struct Closed;
    impl Write for Closed {
      fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
      }

      fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
      }
    }
    let closed = section(&mut Closed, "GPU", |_| Ok(()));
    assert!(closed.is_err_and(|e| crate::broken_pipe(&e)));
  }

  #[test]
  fn kernels_and_why() {
    let matrix = |n, ab_type| {
      let (m_size, n_size, k_size, cr_type) = (n, n, n, Scalar::F32);
      Matrix { m_size, n_size, k_size, ab_type, cr_type, saturating_accumulation: false }
    };
    let coop = Features::SUBGROUP
      | Features::SUBGROUP_BARRIER
      | Features::SHADER_F16
      | Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let pass = coop | Features::PASSTHROUGH_SHADERS;
    let apple = vec![matrix(8, Scalar::F32), matrix(8, Scalar::F16)];
    let nvidia = vec![matrix(16, Scalar::F16)];
    let cases = [
      (Kernels::Tensor, 0, pass, [4, 64], apple.clone(), "8x8 matrices, Metal 4 tensor GEMM"),
      (
        Kernels::Matrix8,
        0,
        pass,
        [4, 64],
        apple.clone(),
        "8x8 matrices: the tensor GEMM is for Apple M5 / A19 and newer",
      ),
      (
        Kernels::Matrix8,
        0,
        coop,
        [4, 64],
        apple.clone(),
        "8x8 matrices: no passthrough shaders for the tensor GEMM",
      ),
      (Kernels::Matrix16, 0x10de, coop, [32, 32], nvidia.clone(), "16x16 matrices"),
      // DX12 on Windows
      (
        Kernels::Portable,
        0x1002,
        Features::SUBGROUP | Features::SHADER_F16,
        [4, 128],
        vec![],
        "portable: no subgroup barriers, cooperative matrices",
      ),
      // AMD GCN
      (
        Kernels::Portable,
        0x1002,
        coop,
        [64, 64],
        vec![],
        "portable: the fast kernels need 32-lane subgroups",
      ),
      // AMD RDNA on Windows, whose driver crashes compiling 16×16 matrices
      (
        Kernels::Portable,
        0x1002,
        coop,
        [32, 64],
        nvidia,
        "portable: 16x16 matrices run on NVIDIA only so far, vendor 0x1002",
      ),
      (
        Kernels::Portable,
        0x8086,
        coop,
        [8, 32],
        vec![],
        "portable: no 8x8 or 16x16 matrices into f32",
      ),
    ];
    for (k, vendor, features, subgroups, matrices, want) in cases {
      assert_eq!(kernels(k, false, &Facts { vendor, features, subgroups, matrices }), want);
    }

    let llvmpipe = Facts { vendor: 0x10005, features: coop, subgroups: [8, 8], matrices: vec![] };
    assert_eq!(kernels(Kernels::Portable, true, &llvmpipe), "portable, CPU tiles");

    assert_eq!(shapes(&apple), "8x8x8 f32>f32, 8x8x8 f16>f32");
    assert_eq!(shapes(&[]), "none");
  }
}
