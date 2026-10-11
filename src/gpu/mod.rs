// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! wgpu device, weight upload, and a recorder that encodes kernel dispatches into compute passes.
//! Each kernel has a portable version; a fast one (subgroups, cooperative matrices) replaces it
//! when the device supports it and a probe dispatch of that pipeline succeeds.

use std::sync::{Arc, Mutex, OnceLock, PoisonError, mpsc};

use wgpu::BindGroup;
pub use wgpu::Buffer;

use crate::error::{Error, Result, bail};
use crate::gguf::{Tensor, Type};

type Pipeline = wgpu::ComputePipeline;

/// Bind groups cached per dispatch, each with the pipeline it was made for.
pub type Groups = Vec<(Pipeline, BindGroup)>;

/// Submissions a call may make (slots in `Gpu::marks`).
const MARKS: u64 = 256;

pub struct Gpu {
  device: wgpu::Device,
  queue: wgpu::Queue,
  /// Usage of weight buffers. On unified memory they are also `MAP_WRITE`, so filling their
  /// mapping writes the buffer itself: no staging copy at load, none queued for the first call.
  weights: wgpu::BufferUsages,
  kernels: Kernels,
  fast: Fast,
  /// Bound for a writable output a dispatch doesn't use, so no writable binding aliases another.
  spare: Buffer,
  /// One slot per submission of a call, set to the call's number when its work is done. A driver
  /// may cancel a long submission and report success (i915 does after 20 s), which would read
  /// back the previous call's output; a slot without the number makes the call an error instead.
  marks: Buffer,
  marks_read: Buffer,
  /// The last call's number. Calls share `marks`, so each holds this lock while it runs.
  calls: Mutex<u32>,
  /// The first device-lost or uncaptured wgpu error, reported by the next call.
  failure: Arc<Mutex<Option<String>>>,
}

struct Kernels {
  /// One per weight type.
  gemm: Vec<Pipeline>,
  layer_norm: Pipeline,
  im2col: Pipeline,
  conv_glu: Pipeline,
  flash: Pipeline,
  scores: Pipeline,
  softmax: Pipeline,
  values: Pipeline,
  depthwise: Pipeline,
  first_depthwise: Pipeline,
  flatten: Pipeline,
  rope: Pipeline,
  group_norm: Pipeline,
  swiglu: Pipeline,
  argmax: Pipeline,
  mark: Pipeline,
  gemm_module: wgpu::ShaderModule,
  /// Compiled on first use: only encoder–decoder models need them.
  decoder: OnceLock<Decoder>,
}

impl Kernels {
  fn new(device: &wgpu::Device) -> Self {
    let gemm = module(device, include_str!("gemm.wgsl"));
    let ops = module(device, include_str!("ops.wgsl"));
    let attention = module(device, include_str!("attention.wgsl"));
    let spatial = module(device, include_str!("spatial.wgsl"));
    Self {
      gemm: gemms(device, &gemm, "gemm"),
      layer_norm: pipeline(device, &ops, "layer_norm", &[]),
      im2col: pipeline(device, &ops, "im2col", &[]),
      conv_glu: pipeline(device, &ops, "conv_glu", &[]),
      flash: pipeline(device, &attention, "flash", &[]),
      scores: pipeline(device, &attention, "scores", &[]),
      softmax: pipeline(device, &ops, "softmax", &[]),
      values: pipeline(device, &attention, "values", &[]),
      depthwise: pipeline(device, &spatial, "depthwise", &[]),
      first_depthwise: pipeline(device, &spatial, "first_depthwise", &[]),
      flatten: pipeline(device, &spatial, "flatten", &[]),
      rope: pipeline(device, &ops, "rotary", &[]),
      group_norm: pipeline(device, &ops, "group_norm", &[]),
      swiglu: pipeline(device, &ops, "swiglu", &[]),
      argmax: pipeline(device, &ops, "argmax", &[]),
      mark: pipeline(device, &ops, "mark", &[]),
      gemm_module: gemm,
      decoder: OnceLock::new(),
    }
  }
}

/// The fast GEMMs (per weight type 4F- and 8F-row tiles), LayerNorm, conv_glu, flash attention and
/// relative attention, before their probes. Only the GEMM runs with F = 16: the flash kernels need
/// 8×8 F32 matrices, and on an RTX 4060 the fast LayerNorm and conv_glu were slower.
type FastPipelines = (Vec<Pipeline>, [Option<Pipeline>; 4]);

/// Compiles the fast kernels for F×F cooperative matrices; `None` if any fails.
fn compile_fast(device: &wgpu::Device, f: usize) -> Option<FastPipelines> {
  // Error scopes are per thread.
  let scopes = [wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation]
    .map(|f| device.push_error_scope(f));
  let source = include_str!("gemm_fast.wgsl")
    .replace("F: u32 = 8u", &format!("F: u32 = {f}u"))
    .replace("8x8", &format!("{f}x{f}"));
  let gemm = module(device, &source);
  let gemm: Vec<_> = (0..6)
    .map(|i| {
      let constants = [("WTYPE", (i / 2) as f64), ("ROWS", ((4 * f) << (i % 2)) as f64)];
      pipeline(device, &gemm, "gemm", &constants)
    })
    .collect();
  let kernels = (f == 8).then(|| {
    let ops = module(device, include_str!("ops_fast.wgsl"));
    let relative = module(device, include_str!("attention_fast.wgsl"));
    let flash = module(device, include_str!("flash.wgsl"));
    [
      pipeline(device, &ops, "layer_norm", &[]),
      pipeline(device, &ops, "conv_glu", &[]),
      pipeline(device, &flash, "flash", &[]),
      pipeline(device, &relative, "attention", &[]),
    ]
  });
  let kernels = kernels.map(|k| k.map(Some)).unwrap_or_default();
  let errors = scopes.into_iter().rev().filter_map(|s| pollster::block_on(s.pop())).count();
  (errors == 0).then_some((gemm, kernels))
}

/// Kernels for one decoder token: `gemv` and `embed` per weight type, `attend` and `pack`.
struct Decoder {
  gemv: Vec<Pipeline>,
  embed: Vec<Pipeline>,
  attend: Pipeline,
  pack: Pipeline,
}

#[derive(Default)]
struct Fast {
  /// F of the GEMM's F×F cooperative matrices: 8 (Apple) or 16; 0 without the fast GEMM.
  size: usize,
  gemm: Option<Vec<Pipeline>>,
  layer_norm: Option<Pipeline>,
  conv_glu: Option<Pipeline>,
  /// Flash attention for head_dim 32..64.
  attention: Option<Pipeline>,
  /// Flash attention with relative positions for head_dim ≤ 128.
  relative: Option<Pipeline>,
}

/// A linear layer `y = x·Wᵀ + b` with `W` row-major `[n][k]`, kept in its GGUF type. Q8_0 is
/// stored as int8 data (4 per u32, row-major) plus one f16 scale per 32-weight block.
pub struct Linear {
  /// Rows of `W`, padded to a multiple of 64 for the tile reads.
  n: usize,
  /// Output columns: the rows of `W` when a multiple of 32, else `n`.
  cols: usize,
  k: usize,
  /// Outputs written a second time after the `cols` others, with their own bias.
  dual: usize,
  /// GEMM pipeline index: 0: F32, 1: F16, 2: Q8_0
  ty: usize,
  weight: Buffer,
  scales: Option<Buffer>,
  bias: Buffer,
}

impl Linear {
  /// Output columns: the rows of `W` (padded to a multiple of 64 unless a multiple of 32), and
  /// the dual outputs.
  pub fn width(&self) -> usize {
    self.cols + self.dual
  }
}

/// Per-channel weights and biases: a LayerNorm's gain and bias, a per-channel affine `x·w + b`, or
/// a depthwise conv's taps and bias.
pub struct Channels {
  w: Buffer,
  b: Buffer,
}

/// The norm after the Conformer conv module's depthwise conv.
#[derive(Clone, Copy, PartialEq)]
pub enum ConvNorm {
  LayerNorm,
  /// `x·g + b`: BatchNorm folded at load.
  Affine,
}

#[derive(Clone, Copy)]
pub enum Epilogue {
  Bias,
  Silu,
  Relu,
  /// `c += alpha · (a·Wᵀ + b)`
  Residual(f32),
  Gelu,
  /// `c += GELU(a·Wᵀ + b)`
  GeluResidual,
  Tanh,
}

impl Epilogue {
  /// The kernels' mode and alpha.
  fn params(self) -> [u32; 2] {
    match self {
      Epilogue::Bias => [0, 0],
      Epilogue::Silu => [1, 0],
      Epilogue::Relu => [2, 0],
      Epilogue::Residual(alpha) => [3, alpha.to_bits()],
      Epilogue::Gelu => [4, 0],
      Epilogue::GeluResidual => [5, 0],
      Epilogue::Tanh => [6, 0],
    }
  }
}

impl Gpu {
  /// The device with the fast kernels that pass their probes, or only the portable ones.
  pub fn new(fast: bool) -> Result<Self> {
    let adapter = adapter()?;
    let size = if fast { fast_size(&adapter) } else { None };
    let fast = size.is_some();
    let mappable = adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu
      && adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
    let mut features = wgpu::Features::IMMEDIATES;
    if fast {
      features |= COOP;
    }
    if mappable {
      features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    }
    let supported = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
      required_features: features,
      required_limits: wgpu::Limits {
        max_immediate_size: 32,
        max_buffer_size: supported.max_buffer_size,
        max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
        max_compute_workgroup_storage_size: supported.max_compute_workgroup_storage_size,
        ..Default::default()
      },
      // SAFETY: cooperative matrices are only used by the bundled kernels, after a probe.
      experimental_features: if fast {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
      } else {
        wgpu::ExperimentalFeatures::disabled()
      },
      // Vulkan suballocates from smaller blocks (8–64 MiB, not 128–256): ~200 MiB less for
      // GigaAM on an Intel iGPU at the same speed. Metal ignores it.
      memory_hints: wgpu::MemoryHints::MemoryUsage,
      ..Default::default()
    }))?;
    let failure = Arc::new(Mutex::new(None));
    let f = failure.clone();
    device.on_uncaptured_error(Arc::new(move |e| fail(&f, e.to_string())));
    let f = failure.clone();
    device.set_device_lost_callback(move |_, message| fail(&f, format!("device lost: {message}")));

    // The fast kernels compile on a second thread meanwhile.
    let (kernels, compiled) = std::thread::scope(|s| {
      let compiled = s.spawn(|| size.and_then(|f| compile_fast(&device, f)));
      (Kernels::new(&device), compiled.join().unwrap())
    });
    let mut weights = wgpu::BufferUsages::STORAGE;
    if mappable {
      weights |= wgpu::BufferUsages::MAP_WRITE;
    }
    let buffer = |size, usage| {
      device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
      })
    };
    let spare = buffer(4, wgpu::BufferUsages::STORAGE);
    let marks = buffer(MARKS * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let marks_read = buffer(MARKS * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let mut gpu = Self {
      device,
      queue,
      weights,
      kernels,
      fast: Fast::default(),
      spare,
      marks,
      marks_read,
      calls: Mutex::new(0),
      failure,
    };
    if let (Some(size), Some(compiled)) = (size, compiled) {
      gpu.fast = gpu.probe(size, compiled);
    }
    Ok(gpu)
  }

  /// Keeps each fast kernel whose probe reports its subgroup layout. With all params zero a
  /// kernel sees an empty problem and writes its layout, `subgroups` of 32 lanes, to its output.
  /// The probes run in one submission; a device error turns them all off.
  fn probe(&self, size: usize, compiled: FastPipelines) -> Fast {
    let (gemm, [layer_norm, conv_glu, attention, relative]) = compiled;
    // Per kernel: bindings, the output's, params and subgroups.
    let mut probes: Vec<_> = gemm.into_iter().map(|k| (Some(k), 5, 2, 6, 4.0)).collect();
    probes.extend([
      (layer_norm, 6, 3, 4, 4.0),
      (conv_glu, 6, 5, 6, 4.0),
      (attention, 3, 2, 7, 4.0),
      (relative, 4, 3, 8, 1.0),
    ]);
    let read = self.readback(2 * probes.len());
    let mut groups = Vec::new();
    let mut p = Pass::new(self, &mut groups, 0);
    let outs: Vec<Buffer> = (probes.iter())
      .map(|(k, bindings, out, params, _)| {
        let mut buffers: Vec<Buffer> = (0..*bindings).map(|_| self.buffer(16)).collect();
        if let Some(k) = k {
          p.run(k, &buffers.iter().collect::<Vec<_>>(), &[0; 8][..*params], [1, 1, 1]);
        }
        buffers.swap_remove(*out)
      })
      .collect();
    let Pass { mut encoder, pass, .. } = p;
    drop(pass);
    for (i, out) in outs.iter().enumerate() {
      encoder.copy_buffer_to_buffer(out, 0, &read, 8 * i as u64, 8);
    }
    self.queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let mapped = self.device.poll(wgpu::PollType::wait_indefinitely()).is_ok()
      && rx.recv().unwrap().is_ok()
      && self.failure.lock().unwrap_or_else(PoisonError::into_inner).take().is_none();
    let range = mapped.then(|| read.slice(..).get_mapped_range().ok()).flatten();
    let got: Vec<f32> = range.map(|r| bytemuck::cast_slice(&r).to_vec()).unwrap_or_default();
    let mut kept = (probes.into_iter().enumerate()).map(|(i, (k, .., subgroups))| {
      k.filter(|_| got.get(2 * i..2 * i + 2) == Some(&[32.0, subgroups][..]))
    });
    let mut next = || kept.next().flatten();
    // All six GEMMs are taken before any is checked, so the kernels after them stay in place.
    let gemm: Vec<_> = (0..6).map(|_| next()).collect();
    let gemm: Option<Vec<_>> = gemm.into_iter().collect();
    Fast {
      size: if gemm.is_some() { size } else { 0 },
      gemm,
      layer_norm: next(),
      conv_glu: next(),
      attention: next(),
      relative: next(),
    }
  }

  /// An uninitialized storage buffer of `len` f32s.
  pub fn buffer(&self, len: usize) -> Buffer {
    self.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: (len.max(1) * 4) as u64,
      usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    })
  }

  /// A buffer the CPU reads results back from.
  pub fn readback(&self, len: usize) -> Buffer {
    self.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: (len.max(1) * 4) as u64,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    })
  }

  /// A weight buffer of `size` bytes, mapped for the caller to fill and unmap.
  fn mapped(&self, size: usize) -> Buffer {
    self.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: size.max(4).next_multiple_of(4) as u64,
      usage: self.weights,
      mapped_at_creation: true,
    })
  }

  pub fn upload(&self, data: &[f32]) -> Result<Buffer> {
    let buffer = self.mapped(data.len() * 4);
    let bytes: &[u8] = bytemuck::cast_slice(data);
    buffer.get_mapped_range_mut(..)?.slice(..bytes.len()).copy_from_slice(bytes);
    buffer.unmap();
    Ok(buffer)
  }

  pub fn write(&self, buffer: &Buffer, data: &[f32]) {
    self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
  }

  /// Uploads weights `[.., n_i]` of one type and inner size, stacked along n, with the bias of
  /// the stacked rows, or zeros when `bias` is empty. The GEMM computes 64 outputs at a time, so
  /// the rows are padded with zeros to a multiple of 64. Outputs past the rows are not written
  /// when the rows are a multiple of 32 (d 288), else they are `epilogue(0)` up to the padding. A
  /// bias with `dual` more values has the first `dual` outputs written again after the others
  /// with those biases (one q with two position biases).
  pub fn linear(&self, weights: &[Tensor], bias: &[f32]) -> Result<Linear> {
    self.upload_linear(weights, bias, false)
  }

  /// A conv1d weight `[kernel, ch, out]` as a linear layer over the columns of `Pass::im2col`,
  /// whose rows are padded with zeros to a multiple of 32, the GEMM's k step. W's rows get the
  /// same zero columns (an 80-mel conv: 240 → 256) in F32, which is exact as the GEMM rounds W to
  /// f16 either way.
  pub fn conv(&self, weight: Tensor, bias: &[f32]) -> Result<Linear> {
    self.upload_linear(&[weight], bias, true)
  }

  fn upload_linear(&self, weights: &[Tensor], bias: &[f32], pad: bool) -> Result<Linear> {
    let w0 = &weights[0];
    let k = w0.len() / w0.dims.last().unwrap();
    let rows = weights.iter().map(|w| w.dims.last().unwrap()).sum::<usize>();
    if weights.iter().any(|w| w.ty != w0.ty || w.len() / w.dims.last().unwrap() != k) {
      bail!("stacked weights differ in type or shape");
    }
    if !pad && !k.is_multiple_of(32) {
      bail!("linear layer {k}→{rows}: needs k % 32 == 0");
    }
    // Buffers start zeroed, so the padding rows only need the space.
    let n = rows.next_multiple_of(64);
    let cols = if rows.is_multiple_of(32) { rows } else { n };
    let kp = k.next_multiple_of(32);
    // The file is streamed in pieces of 8192 Q8_0 blocks straight into the mapped buffers.
    const PIECE: usize = 34 << 13;
    let (ty, weight, scales) = match w0.ty {
      _ if kp != k => {
        let weight = self.mapped(n * kp * 4);
        let mut view = weight.get_mapped_range_mut(..)?;
        let rows = weights.iter().map(|w| w.to_f32()).collect::<Result<Vec<_>>>()?;
        for (i, row) in rows.iter().flat_map(|w| w.chunks(k)).enumerate() {
          view.slice(i * kp * 4..(i * kp + k) * 4).copy_from_slice(bytemuck::cast_slice(row));
        }
        drop(view);
        weight.unmap();
        (0, weight, None)
      }
      Type::F32 | Type::F16 => {
        let (ty, width) = if w0.ty == Type::F32 { (0, 4) } else { (1, 2) };
        let weight = self.mapped(n * k * width);
        let mut view = weight.get_mapped_range_mut(..)?;
        let mut at = 0;
        for w in weights {
          w.read(PIECE, |i, piece| {
            view.slice(at + i..at + i + piece.len()).copy_from_slice(piece)
          })?;
          at += w.len() * width;
        }
        drop(view);
        weight.unmap();
        (ty, weight, None)
      }
      Type::Q8_0 => {
        let (data, scales) = (self.mapped(n * k), self.mapped(n * k / 16));
        let (mut dv, mut sv) = (data.get_mapped_range_mut(..)?, scales.get_mapped_range_mut(..)?);
        let mut block = 0;
        for w in weights {
          w.read(PIECE, |i, piece| {
            // Each block's scale and data go straight into the mappings.
            let blocks = piece.as_chunks::<34>().0;
            let (a, b) = (block + i / 34, block + i / 34 + blocks.len());
            let d = dv.slice(a * 32..b * 32).into_chunks::<32>().0;
            d.write_iter(blocks.iter().map(|q| <[u8; 32]>::try_from(&q[2..]).unwrap()));
            let s = sv.slice(a * 2..b * 2).into_chunks::<2>().0;
            s.write_iter(blocks.iter().map(|q| [q[0], q[1]]));
          })?;
          block += w.len() / 32;
        }
        drop((dv, sv));
        data.unmap();
        scales.unmap();
        (2, data, Some(scales))
      }
    };
    let dual = bias.len().saturating_sub(rows);
    if !bias.is_empty() && bias.len() < rows || dual > rows {
      bail!("linear layer {k}→{rows}: bias has {} values", bias.len());
    }
    let (own, second) = bias.split_at(bias.len() - dual);
    let mut padded = vec![0.0; n.max(cols + dual)];
    padded[..own.len()].copy_from_slice(own);
    padded[cols..cols + dual].copy_from_slice(second);
    Ok(Linear { n, cols, k: kp, dual, ty, weight, scales, bias: self.upload(&padded)? })
  }

  pub fn channels(&self, w: &[f32], b: &[f32]) -> Result<Channels> {
    Ok(Channels { w: self.upload(w)?, b: self.upload(b)? })
  }

  fn decoder(&self) -> &Decoder {
    self.kernels.decoder.get_or_init(|| {
      let (device, gemm) = (&self.device, &self.kernels.gemm_module);
      let decode = module(device, include_str!("decode.wgsl"));
      Decoder {
        gemv: gemms(device, gemm, "gemv"),
        embed: gemms(device, gemm, "embed"),
        attend: pipeline(device, &decode, "attend", &[]),
        pack: pipeline(device, &decode, "pack", &[]),
      }
    })
  }

  /// The rows of the fast GEMM's tile for `m` rows of `n` outputs, or `None` for the portable GEMM.
  fn tile_rows(&self, n: usize, m: usize) -> Option<usize> {
    self.fast.gemm.as_ref()?;
    let short = 4 * self.fast.size;
    let (cols, m8, m4) = (n.div_ceil(2 * short), m.div_ceil(2 * short), m.div_ceil(short));
    // Tiles 128 wide (F = 16) in fewer than 16 workgroups ran slower than the portable 64×64 ones
    // on an RTX 4060: GigaAM's 768-wide layers at 113 rows, 12 workgroups, by 1.5×.
    if short == 64 && cols * m4 < 16 {
      return None;
    }
    // 8F-row tiles unless they leave few workgroups or pad more rows than 4F-row ones.
    Some(if m8 * cols >= 64 && m8 * 20 <= m4 * 11 { 2 * short } else { short })
  }

  fn flash(&self, head_dim: usize, relative: bool) -> Option<&Pipeline> {
    let (k, min, max) =
      if relative { (&self.fast.relative, 8, 128) } else { (&self.fast.attention, 32, 64) };
    k.as_ref().filter(|_| (min..=max).contains(&head_dim) && head_dim.is_multiple_of(8))
  }

  /// Whether the portable path has a flash kernel for heads of `head_dim`.
  fn portable_flash(head_dim: usize, relative: bool) -> bool {
    !relative && head_dim <= 64 && head_dim.is_multiple_of(4)
  }

  /// Floats of attention scratch for `t` frames: the portable path without flash attention keeps
  /// all scores.
  pub fn scores_len(&self, t: usize, heads: usize, head_dim: usize, relative: bool) -> usize {
    match self.flash(head_dim, relative).is_some() || Self::portable_flash(head_dim, relative) {
      true => 1,
      false => heads * t * t,
    }
  }

  fn begin(&self) -> (wgpu::CommandEncoder, wgpu::ComputePass<'static>) {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    let pass = encoder.begin_compute_pass(&Default::default()).forget_lifetime();
    (encoder, pass)
  }

  /// Records the dispatches of `record` (one submission, or more with `Pass::layers`), runs them,
  /// and reads back `len` f32s of `out` through the `read` buffer. `groups` caches bind groups by
  /// dispatch index, so a caller must record the same dispatch sequence over the same buffers each
  /// time, and clear it when buffers change.
  pub fn run(
    &self,
    groups: &mut Groups,
    out: &Buffer,
    read: &Buffer,
    len: usize,
    record: impl FnOnce(&mut Pass),
  ) -> Result<Vec<f32>> {
    let mut calls = self.calls.lock().unwrap_or_else(PoisonError::into_inner);
    *calls = calls.wrapping_add(1);
    let call = *calls;
    let mut p = Pass::new(self, groups, call);
    record(&mut p);
    p.mark(out);
    let Pass { mut encoder, pass, marked, .. } = p;
    drop(pass);
    let (size, marks) = ((len * 4) as u64, marked as u64 * 4);
    encoder.copy_buffer_to_buffer(out, 0, read, 0, size);
    encoder.copy_buffer_to_buffer(&self.marks, 0, &self.marks_read, 0, marks);
    self.queue.submit([encoder.finish()]);

    let slices = [read.slice(..size), self.marks_read.slice(..marks)];
    let (tx, rx) = mpsc::channel();
    for slice in &slices {
      let tx = tx.clone();
      slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    }
    self.device.poll(wgpu::PollType::wait_indefinitely())?;
    for _ in &slices {
      rx.recv().unwrap()?;
    }
    let data = bytemuck::cast_slice(&slices[0].get_mapped_range()?).to_vec();
    let done =
      bytemuck::cast_slice::<_, u32>(&slices[1].get_mapped_range()?).iter().all(|&m| m == call);
    read.unmap();
    self.marks_read.unmap();
    if let Some(failure) = self.failure.lock().unwrap_or_else(PoisonError::into_inner).take() {
      return Err(Error::Gpu(failure));
    }
    if !done {
      return Err(Error::Gpu("the driver did not finish the work (cancelled as too long?)".into()));
    }
    Ok(data)
  }
}

fn fail(failure: &Mutex<Option<String>>, message: String) {
  failure.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(message);
}

/// Rows of projected positions for relative attention over `t` frames: positions t + 15 down to
/// 1 − t. The first 16 are only read by the fast kernel's partial tiles, as are 64 more rows past
/// the end, which a buffer must have (finite) too.
pub fn pos_rows(t: usize) -> usize {
  2 * t + 15
}

/// Output length of a stride-2 conv with kernel 3 and padding 1 (or kernel 5 and padding 2).
pub fn half(n: usize) -> usize {
  (n - 1) / 2 + 1
}

const COOP: wgpu::Features = wgpu::Features::SUBGROUP
  .union(wgpu::Features::SUBGROUP_BARRIER)
  .union(wgpu::Features::SHADER_F16)
  .union(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);

fn adapter() -> Result<wgpu::Adapter> {
  let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
  Ok(pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
    power_preference: wgpu::PowerPreference::HighPerformance,
    ..Default::default()
  }))?)
}

/// F of the F×F cooperative matrices the fast kernels use on `adapter`, which needs 32-lane
/// subgroups: 8 with 8×8 matrices of F32 and of F16 into F32 (Apple GPUs; all fast kernels), else
/// 16 with 16×16 F16 into F32 (only the GEMM), on NVIDIA only so far. AMD's Windows driver
/// (25.10, Radeon 610M) reports the same 16×16 shape but crashes the process while compiling any
/// shader that uses it, which no probe can catch; other drivers are untested.
fn fast_size(adapter: &wgpu::Adapter) -> Option<usize> {
  use wgpu::CooperativeScalarType::{F16, F32};
  let info = adapter.get_info();
  let matrices = adapter.cooperative_matrix_properties();
  let has = |f, ab| {
    let shape = (f, f, f, ab, F32);
    matrices.iter().any(|p| (p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type) == shape)
  };
  if !adapter.features().contains(COOP)
    || !(info.subgroup_min_size..=info.subgroup_max_size).contains(&32)
  {
    return None;
  }
  const NVIDIA: u32 = 0x10de;
  if has(8, F32) && has(8, F16) {
    Some(8)
  } else {
    (has(16, F16) && info.vendor == NVIDIA).then_some(16)
  }
}

fn module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
  let desc =
    wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(source.into()) };
  // SAFETY: the bundled kernels only index inside the buffers their callers size for them.
  unsafe { device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
}

fn pipeline(
  device: &wgpu::Device,
  module: &wgpu::ShaderModule,
  entry: &str,
  constants: &[(&str, f64)],
) -> Pipeline {
  device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some(entry),
    layout: None,
    module,
    entry_point: Some(entry),
    compilation_options: wgpu::PipelineCompilationOptions {
      constants,
      zero_initialize_workgroup_memory: false,
    },
    cache: None,
  })
}

/// The `entry` specialized for each weight type.
fn gemms(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str) -> Vec<Pipeline> {
  (0..3).map(|ty| pipeline(device, module, entry, &[("WTYPE", ty as f64)])).collect()
}

/// Records kernel dispatches into a compute pass.
pub struct Pass<'a> {
  gpu: &'a Gpu,
  encoder: wgpu::CommandEncoder,
  pass: wgpu::ComputePass<'static>,
  groups: &'a mut Groups,
  next: usize,
  /// The call's number and the submissions marked so far (see `Gpu::marks`).
  call: u32,
  marked: u32,
  /// The last submission `flush` made.
  queued: Option<wgpu::SubmissionIndex>,
}

impl<'a> Pass<'a> {
  fn new(gpu: &'a Gpu, groups: &'a mut Groups, call: u32) -> Self {
    let (encoder, pass) = gpu.begin();
    Self { gpu, encoder, pass, groups, next: 0, call, marked: 0, queued: None }
  }

  /// Records an encoder's layers, `layer(pass, item)` for each item, each returning the buffer it
  /// wrote last; on the portable path each layer is its own submission (see `flush`).
  pub fn layers<'b, T>(&mut self, items: &[T], mut layer: impl FnMut(&mut Self, &T) -> &'b Buffer) {
    for item in items {
      let out = layer(self, item);
      self.flush(out);
    }
  }

  /// Submits the work recorded so far, marked done once `after` is written, and waits for the
  /// submission before it. A driver may cancel a long request (i915 20 s after it was queued,
  /// while a slow GPU runs a minute of Parakeet for longer), so each request is short and at
  /// most two are queued. Errors show in `Gpu::run`'s last poll. With Apple's fast kernels (F = 8)
  /// a call stays one submission: splitting costs ~1% there, and a 20 s call ran fine.
  fn flush(&mut self, after: &Buffer) {
    if self.gpu.fast.size == 8 {
      return;
    }
    self.mark(after);
    let (encoder, pass) = self.gpu.begin();
    drop(std::mem::replace(&mut self.pass, pass));
    let done = std::mem::replace(&mut self.encoder, encoder);
    let index = self.gpu.queue.submit([done.finish()]);
    if let Some(previous) = self.queued.replace(index) {
      let wait = wgpu::PollType::Wait { submission_index: Some(previous), timeout: None };
      let _ = self.gpu.device.poll(wait);
    }
  }

  /// Marks the work recorded since the last mark done, once `after` is written.
  /// Its bind group stays out of `groups`, so callers may reuse a cache across recordings that
  /// share only a prefix of dispatches.
  fn mark(&mut self, after: &Buffer) {
    let gpu = self.gpu;
    let group = self.bind_group(&gpu.kernels.mark, &[after, &gpu.marks]);
    self.dispatch(&gpu.kernels.mark, &group, &[self.marked, self.call], [1, 1, 1]);
    self.marked += 1;
  }

  /// Dispatches `kernel` with `buffers` bound in order and `params` as immediates.
  fn run(&mut self, kernel: &Pipeline, buffers: &[&Buffer], params: &[u32], size: [usize; 3]) {
    // A group fits only the pipeline it was made for (the GEMM picks its tile per call).
    if self.groups.get(self.next).is_none_or(|(k, _)| k != kernel) {
      let group = (kernel.clone(), self.bind_group(kernel, buffers));
      match self.groups.get_mut(self.next) {
        Some(g) => *g = group,
        None => self.groups.push(group),
      }
    }
    let group = self.groups[self.next].1.clone();
    self.dispatch(kernel, &group, params, size);
    self.next += 1;
  }

  fn bind_group(&self, kernel: &Pipeline, buffers: &[&Buffer]) -> BindGroup {
    let entries: Vec<_> = buffers
      .iter()
      .enumerate()
      .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
      .collect();
    let layout = kernel.get_bind_group_layout(0);
    let desc = wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries };
    self.gpu.device.create_bind_group(&desc)
  }

  fn dispatch(
    &mut self,
    kernel: &Pipeline,
    group: &BindGroup,
    params: &[u32],
    [x, y, z]: [usize; 3],
  ) {
    self.pass.set_pipeline(kernel);
    self.pass.set_bind_group(0, group, &[]);
    self.pass.set_immediates(0, bytemuck::cast_slice(params));
    self.pass.dispatch_workgroups(x as u32, y as u32, z as u32);
  }

  /// `c = epilogue(a·Wᵀ + b)` for `m` rows of `a`.
  pub fn gemm(&mut self, a: &Buffer, l: &Linear, c: &Buffer, m: usize, epilogue: Epilogue) {
    let gpu = self.gpu;
    let [mode, alpha] = epilogue.params();
    let params = [m as u32, l.cols as u32, l.k as u32, mode, alpha, l.dual as u32];
    // Only Q8_0 reads scales; other types bind the bias in their place.
    let buffers = [a, &l.bias, c, &l.weight, l.scales.as_ref().unwrap_or(&l.bias)];
    let f = gpu.fast.size;
    match (&gpu.fast.gemm, gpu.tile_rows(l.n, m)) {
      (Some(k), Some(rows)) => {
        let tall = rows > 4 * f;
        self.run(
          &k[2 * l.ty + tall as usize],
          &buffers,
          &params,
          [l.n.div_ceil(8 * f), m.div_ceil(rows), 1],
        )
      }
      _ => self.run(&gpu.kernels.gemm[l.ty], &buffers, &params, [l.n / 64, m.div_ceil(64), 1]),
    }
  }

  /// LayerNorm over `rows` rows of `cols` into `y`; with `rope` (a table, `yr`, head_dim) also the
  /// same with rotary embedding into `yr`. The table holds per position head_dim / 2 cosines
  /// followed by as many sines.
  pub fn layer_norm(
    &mut self,
    x: &Buffer,
    n: &Channels,
    y: &Buffer,
    rows: usize,
    cols: usize,
    rope: Option<(&Buffer, &Buffer, usize)>,
  ) {
    let gpu = self.gpu;
    let (table, yr, head_dim) = rope.unwrap_or((x, &gpu.spare, 0));
    let params = [rows as u32, cols as u32, head_dim as u32];
    let buffers = [x, &n.w, &n.b, y, table, yr];
    match &gpu.fast.layer_norm {
      Some(k) => self.run(k, &buffers, &params, [rows.div_ceil(4), 1, 1]),
      None => self.run(&gpu.kernels.layer_norm, &buffers, &params, [rows, 1, 1]),
    }
  }

  /// `c[offset..]` = `epilogue(a·Wᵀ + b)` for one row `a`, W not rounded to f16.
  pub fn gemv(&mut self, a: &Buffer, l: &Linear, c: &Buffer, epilogue: Epilogue, offset: usize) {
    let [mode, alpha] = epilogue.params();
    let params = [1, l.cols as u32, l.k as u32, mode, alpha, 0, offset as u32];
    let buffers = [a, &l.bias, c, &l.weight, l.scales.as_ref().unwrap_or(&l.bias)];
    self.run(&self.gpu.decoder().gemv[l.ty], &buffers, &params, [l.cols / 32, 1, 1]);
  }

  /// `c` = row `id` of the table `l` + `pos[offset..]` + the table's bias, `l.k` values.
  pub fn embed(&mut self, l: &Linear, pos: &Buffer, c: &Buffer, id: usize, offset: usize) {
    let params = [id as u32, l.n as u32, l.k as u32, 0, 0, 0, offset as u32];
    let buffers = [pos, &l.bias, c, &l.weight, l.scales.as_ref().unwrap_or(&l.bias)];
    self.run(&self.gpu.decoder().embed[l.ty], &buffers, &params, [l.k.div_ceil(256), 1, 1]);
  }

  /// Attention of one query per head, `q` `[heads · head_dim]`, over keys 0..n of an F16 cache
  /// into `o`. A cache row is K of all heads, then V, as f16 pairs.
  pub fn attend(&mut self, q: &Buffer, kv: &Buffer, o: &Buffer, n: usize, heads: usize, hd: usize) {
    let scale = (1.0 / (hd as f32).sqrt()).to_bits();
    let (stride, v) = ((heads * hd) as u32, (heads * hd / 2) as u32);
    self.run(
      &self.gpu.decoder().attend,
      &[q, kv, o],
      &[n as u32, hd as u32, scale, stride, v],
      [heads, 1, 1],
    );
  }

  /// Packs `pairs` pairs of floats from `x[src..]` into f16 pairs at `y[dst..]`.
  pub fn pack(&mut self, x: &Buffer, y: &Buffer, pairs: usize, src: usize, dst: usize) {
    let params = [pairs as u32, 0, 0, 0, 0, src as u32, dst as u32];
    self.run(&self.gpu.decoder().pack, &[x, y], &params, [pairs.div_ceil(256), 1, 1]);
  }

  /// Columns for a conv1d with `[kernel, stride, pad]` over `x` `[t][ch]`, rows of `ch · kernel`
  /// padded with zeros to a multiple of 32.
  pub fn im2col(&mut self, x: &Buffer, col: &Buffer, t: usize, ch: usize, conv: [usize; 3]) {
    let [kernel, stride, pad] = conv;
    let rows = (t + 2 * pad - kernel) / stride + 1;
    let params = [t, ch, 0, 0, kernel, 0, stride, pad].map(|v| v as u32);
    let width = (ch * kernel).next_multiple_of(32);
    self.run(&self.gpu.kernels.im2col, &[x, col], &params, [width.div_ceil(256), rows, 1]);
  }

  /// Conformer conv module between the pointwise convs: GLU, depthwise conv (odd `kernel`,
  /// centred) + bias, `norm`, SiLU. `h` is `[t][2·ch]`, `y` is `[t][ch]`.
  #[allow(clippy::too_many_arguments)]
  pub fn conv_glu(
    &mut self,
    h: &Buffer,
    dw: &Channels,
    kernel: usize,
    n: &Channels,
    norm: ConvNorm,
    y: &Buffer,
    t: usize,
    ch: usize,
  ) {
    let gpu = self.gpu;
    let buffers = [h, &dw.w, &dw.b, &n.w, &n.b, y];
    let layer_norm = (norm == ConvNorm::LayerNorm) as u32;
    let params = [t as u32, ch as u32, 0, 0, kernel as u32, layer_norm];
    match &gpu.fast.conv_glu {
      Some(k) => self.run(k, &buffers, &params, [t.div_ceil(4), 1, 1]),
      None => self.run(&gpu.kernels.conv_glu, &buffers, &params, [t, 1, 1]),
    }
  }

  /// Multi-head attention over `qk` `[t][q | k]` and `v` `[t][heads · head_dim]` into `o`, or
  /// without `v` over `qk` `[t][q | k | v]`. With relative positions `pos`, the projected positions
  /// `[pos_rows(t)][heads · head_dim]` (row r is position t + 15 − r), it is over `qk`
  /// `[t][q + u | k | v | q + v]` (no `v`): score (i, j) is then `((q + u)_i·k_j + (q + v)_i·pos[j
  /// − i + t + 15]) / √head_dim`. `s` holds `scores_len` floats. The fast path reads rows of `qk`
  /// and `v` up to `t` rounded up to 64, which must be finite.
  #[allow(clippy::too_many_arguments)]
  pub fn attention(
    &mut self,
    qk: &Buffer,
    v: Option<&Buffer>,
    pos: Option<&Buffer>,
    s: &Buffer,
    o: &Buffer,
    t: usize,
    heads: usize,
    head_dim: usize,
  ) {
    let ch = (heads * head_dim) as u32;
    // attention.wgsl's params: stride, v_stride, v_offset and relative after t, ch, head_dim and
    // the scale. A flash workgroup takes 64 queries, with relative positions 16.
    let (v, [a, b, c, d], queries) = match (v, pos) {
      (Some(v), _) => (v, [2 * ch, ch, 0, 0], 64),
      (None, None) => (qk, [3 * ch, 3 * ch, 2 * ch, 0], 64),
      (None, Some(_)) => (qk, [4 * ch, 4 * ch, 2 * ch, 1], 16),
    };
    let scale = (1.0 / (head_dim as f32).sqrt()).to_bits();
    let params = [t as u32, ch, head_dim as u32, scale, a, b, c, d];
    let pos = pos.unwrap_or(qk);
    if let Some(k) = self.gpu.flash(head_dim, d == 1) {
      let buffers: &[&Buffer] = if d == 1 { &[qk, v, pos, o] } else { &[qk, v, o] };
      return self.run(k, buffers, &params[..7 + d as usize], [t.div_ceil(queries), heads, 1]);
    }
    let (k, tiles) = (&self.gpu.kernels, t.div_ceil(16));
    if Gpu::portable_flash(head_dim, d == 1) {
      return self.run(&k.flash, &[qk, v, o], &params[..7], [t.div_ceil(32), heads, 1]);
    }
    self.run(&k.scores, &[qk, pos, s], &params, [tiles, tiles, heads]);
    self.run(&k.softmax, &[s], &params[..1], [t, heads, 1]);
    self.run(&k.values, &[s, v, o], &params, [tiles, heads, 1]);
  }

  /// Depthwise 3×3 conv, stride 2, padding 1, + bias (weight `[9][ch]`) over the image
  /// `[t][f][ch]` into output rows t0..t0 + rows of `y` `[⌈t/2⌉][⌈f/2⌉][ch]`; `x` holds the
  /// image's rows from 2·t0 − 1 on. With `first` the image is ReLU(`first` + bias), a 3×3
  /// stride-2 conv from the one-channel image `x` `[t][f]` to ch channels that is never stored,
  /// and `y` gets `rows` rows of `[⌈f/4⌉][ch]`, the output rows from 2·t0 − 1 on.
  #[allow(clippy::too_many_arguments)]
  pub fn depthwise(
    &mut self,
    x: &Buffer,
    first: Option<&Channels>,
    dw: &Channels,
    y: &Buffer,
    image: [usize; 3],
    t0: usize,
    rows: usize,
  ) {
    let k = &self.gpu.kernels;
    let [t, f, ch] = image.map(|v| v as u32);
    let params = [t, f, ch, t0 as u32];
    match first {
      None => self.run(&k.depthwise, &[x, &dw.w, &dw.b, y], &params, [half(image[1]), rows, 1]),
      Some(c) => {
        let buffers = [x, &dw.w, &dw.b, y, &c.w, &c.b];
        self.run(&k.first_depthwise, &buffers, &params, [half(half(image[1])), rows, 1]);
      }
    }
  }

  /// Rotary embedding in place over `rows` rows `[q | k | v]` of `heads` heads of `hd`, at
  /// positions `pos..`: in q's and k's heads the pairs (2i, 2i + 1) for 2i < `rot` rotate. The
  /// table holds per position `rot` / 2 cosines, then as many sines.
  pub fn rope(&mut self, table: &Buffer, qkv: &Buffer, rows: usize, pos: usize, shape: [usize; 3]) {
    let [heads, hd, rot] = shape;
    let params = [rows, 3 * heads * hd, hd, 0, rot, 0, 2 * heads, pos].map(|v| v as u32);
    let size = [(heads * rot).div_ceil(256), rows, 1];
    self.run(&self.gpu.kernels.rope, &[table, qkv], &params, size);
  }

  /// GroupNorm with one group, in place over `x` `[t][ch]`: normalized over all its values, then
  /// times `n.w` and plus `n.b` per channel.
  pub fn group_norm(&mut self, x: &Buffer, n: &Channels, t: usize, ch: usize, eps: f32) {
    let params = [t as u32, ch as u32, 0, eps.to_bits()];
    self.run(&self.gpu.kernels.group_norm, &[x, &n.w, &n.b], &params, [1, 1, 1]);
  }

  /// SwiGLU in place over one row `h` `[x | gate]` of 2·`ch`: `h[..ch]` = SiLU(gate) · x.
  pub fn swiglu(&mut self, h: &Buffer, ch: usize) {
    self.run(&self.gpu.kernels.swiglu, &[h], &[0, ch as u32], [ch.div_ceil(256), 1, 1]);
  }

  /// The index of the largest of `x[..n]` (the lowest on a tie), as the bits of `out[0]`.
  pub fn argmax(&mut self, x: &Buffer, out: &Buffer, n: usize) {
    self.run(&self.gpu.kernels.argmax, &[x, out], &[n as u32], [1, 1, 1]);
  }

  /// `x` `[t][f][ch]` to `y` `[t][ch · f]`, channel-major.
  pub fn flatten(&mut self, x: &Buffer, y: &Buffer, t: usize, f: usize, ch: usize) {
    let k = &self.gpu.kernels.flatten;
    self.run(k, &[x, y], &[t as u32, f as u32, ch as u32], [t, 1, 1]);
  }
}

#[cfg(test)]
mod tests {
  use std::sync::OnceLock;

  use half::f16;

  use super::*;
  use crate::gguf::Gguf;

  /// The portable path, then the fast one (also portable where the adapter doesn't offer it).
  fn gpus() -> &'static [Gpu; 2] {
    static GPUS: OnceLock<[Gpu; 2]> = OnceLock::new();
    GPUS.get_or_init(|| {
      let fast = Gpu::new(true).unwrap();
      let f = &fast.fast;
      let ops = [&f.layer_norm, &f.conv_glu, &f.attention, &f.relative];
      // With 16×16 matrices only the GEMM.
      if let Some(size) = fast_size(&adapter().unwrap()) {
        assert!(
          f.gemm.is_some() && ops.iter().all(|k| k.is_some() == (size == 8)),
          "a fast kernel failed its probe"
        );
      }
      [Gpu::new(false).unwrap(), fast]
    })
  }

  fn random(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    (0..n)
      .map(|_| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x as f32 / u32::MAX as f32 * 2.0 - 1.0
      })
      .collect()
  }

  fn up(gpu: &Gpu, data: &[f32]) -> Buffer {
    gpu.upload(data).unwrap()
  }

  /// Runs `record` with an output buffer that starts as `init`, and returns the output.
  fn run(gpu: &Gpu, init: &[f32], record: impl FnOnce(&mut Pass, &Buffer)) -> Vec<f32> {
    let out = gpu.buffer(init.len());
    gpu.write(&out, init);
    let read = gpu.readback(init.len());
    gpu.run(&mut Vec::new(), &out, &read, init.len(), |p| record(p, &out)).unwrap()
  }

  fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1f32, |m, x| m.max(x.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
      assert!((g - w).abs() <= 2e-5 * scale, "[{i}]: {g} vs {w}");
    }
  }

  fn layer_norm(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let n = g.len() as f32;
    (x.chunks(g.len()))
      .flat_map(|row| {
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        row.iter().zip(g).zip(b).map(move |((x, g), b)| (x - mean) * rstd * g + b)
      })
      .collect()
  }

  fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
  }

  /// As gemm.wgsl's (its constants rounded to f32).
  fn gelu(x: f32) -> f32 {
    let z = (std::f32::consts::FRAC_1_SQRT_2 * x).abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let p = ((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_8) * t - 0.284_496_74) * t;
    0.5 * x * (1.0 + x.signum() * (1.0 - (p + 0.254_829_6) * t * (-z * z).exp()))
  }

  fn epilogue(e: Epilogue, v: f32, c: f32) -> f32 {
    match e {
      Epilogue::Bias => v,
      Epilogue::Silu => silu(v),
      Epilogue::Relu => v.max(0.0),
      Epilogue::Residual(alpha) => c + alpha * v,
      Epilogue::Gelu => gelu(v),
      Epilogue::GeluResidual => c + gelu(v),
      Epilogue::Tanh => v.tanh(),
    }
  }

  const EPILOGUES: [Epilogue; 7] = [
    Epilogue::Bias,
    Epilogue::Silu,
    Epilogue::Relu,
    Epilogue::Residual(0.5),
    Epilogue::Gelu,
    Epilogue::GeluResidual,
    Epilogue::Tanh,
  ];

  /// A random [n][k] weight as F32, F16 and Q8_0 GGUF tensor data.
  fn weights(n: usize, k: usize) -> [(Type, Vec<u8>); 3] {
    let w = random(n * k, 1);
    let f32s: Vec<u8> = bytemuck::cast_slice(&w).to_vec();
    let f16s: Vec<u8> = w.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect();
    let q8: Vec<u8> = (w.chunks(32))
      .flat_map(|block| {
        let d = block.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
        let q = block.iter().map(move |x| (x / d).round() as i8 as u8);
        f16::from_f32(d).to_le_bytes().into_iter().chain(q)
      })
      .collect();
    [(Type::F32, f32s), (Type::F16, f16s), (Type::Q8_0, q8)]
  }

  #[test]
  fn failed_gemm_probe() {
    // A GEMM that fails its probe turns the fast GEMM off and leaves the kernels after it in place.
    let gpu = &gpus()[1];
    let Some(size) = fast_size(&adapter().unwrap()) else { return };
    let (mut gemm, kernels) = compile_fast(&gpu.device, size).unwrap();
    // The portable GEMM writes nothing for m = 0.
    gemm[0] = gpu.kernels.gemm[0].clone();
    let fast = gpu.probe(size, (gemm, kernels.clone()));
    assert!(fast.gemm.is_none() && fast.size == 0);
    assert!([fast.layer_norm, fast.conv_glu, fast.attention, fast.relative] == kernels);
  }

  #[test]
  fn gemm() {
    // 164 rows: the GEMM pads them to 192 output columns, which come out as epilogue(0); a
    // 128-wide tile of the fast GEMM passes them. 288 rows (a multiple of 32, not of 64): the rows
    // of c are 288 wide, and a tile passes them too.
    for (n, cols) in [(164, 192), (288, 288)] {
      let k = 96;
      let w = random(n * k, 1);
      let bias = random(n, 2);
      // 64 more biases: the first 64 outputs again at columns cols.., with no epilogue.
      let dual: Vec<f32> = bias.iter().copied().chain(random(64, 5)).collect();
      let f32s: &[u8] = bytemuck::cast_slice(&w);
      let f16s: Vec<u8> = w.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect();
      let q8: Vec<u8> = (w.chunks(32))
        .flat_map(|block| {
          let d = block.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
          let q = block.iter().map(move |x| (x / d).round() as i8 as u8);
          f16::from_f32(d).to_le_bytes().into_iter().chain(q)
        })
        .collect();
      // Each weight is stored as two halves stacked along n, like q over k.
      let wd = [k, n / 2];
      for (ty, bytes) in [(Type::F32, f32s), (Type::F16, &f16s), (Type::Q8_0, &q8)] {
        let (w1, w2) = bytes.split_at(bytes.len() / 2);
        let file = Gguf::with_tensors(&[("w1", &wd, ty, w1), ("w2", &wd, ty, w2)]);
        let t = |name| file.tensor(name, &wd).unwrap();
        // The GEMM rounds W to f16.
        let half = |x: &f32| f16::from_f32(*x).to_f32();
        let wf: Vec<f32> = [t("w1").to_f32().unwrap(), t("w2").to_f32().unwrap()].concat();
        let wf: Vec<f32> = wf.iter().map(half).collect();
        // With the bias, without (zeros), and with dual outputs.
        for b in [&bias[..], &[], &dual] {
          let width = if b.len() > n { cols + 64 } else { cols };
          for gpu in gpus() {
            let l = gpu.linear(&[t("w1"), t("w2")], b).unwrap();
            assert_eq!(l.width(), width);
            // 2100 rows take the fast path's 8F-row tiles.
            for m in [1, 37, 70, 2100] {
              let a = random(m * k, 3);
              // The fast GEMM rounds a to f16 as well.
              let fast = gpu.tile_rows(n.next_multiple_of(64), m).is_some();
              let ar: Vec<f32> = if fast { a.iter().map(half).collect() } else { a.clone() };
              // One row more than written: it must stay as it was.
              let c = random(m * width + 1, 4);
              let ab = up(gpu, &a);
              for e in EPILOGUES {
                let dot =
                  |r: usize, j: usize| (0..k).map(|x| ar[r * k + x] * wf[j * k + x]).sum::<f32>();
                let want: Vec<f32> = (0..c.len())
                  .map(|i| {
                    let (r, j) = (i / width, i % width);
                    if r == m {
                      c[i]
                    } else if j >= cols {
                      b[n + j - cols] + dot(r, j - cols)
                    } else {
                      let v = if j < n { b.get(j).unwrap_or(&0.0) + dot(r, j) } else { 0.0 };
                      epilogue(e, v, c[i])
                    }
                  })
                  .collect();
                assert_close(&run(gpu, &c, |p, out| p.gemm(&ab, &l, out, m, e)), &want);
              }
            }
          }
        }
      }
    }
  }

  #[test]
  fn layer_norm_and_rope() {
    let (rows, cols, hd) = (37, 192, 48);
    let x = random(rows * cols, 5);
    let (g, b) = (random(cols, 6), random(cols, 7));
    let rope = random(rows * hd, 8);
    let y = layer_norm(&x, &g, &b);
    let yr: Vec<f32> = (0..rows * cols)
      .map(|i| {
        let (t, d, half) = (i / cols, i % hd, hd / 2);
        let (cos, sin) = (rope[t * hd + d % half], rope[t * hd + half + d % half]);
        if d < half { y[i] * cos - y[i + half] * sin } else { y[i] * cos + y[i - half] * sin }
      })
      .collect();
    for gpu in gpus() {
      let n = gpu.channels(&g, &b).unwrap();
      let (xb, rb) = (up(gpu, &x), up(gpu, &rope));
      let zeros = vec![0.0; rows * cols];
      assert_close(&run(gpu, &zeros, |p, y| p.layer_norm(&xb, &n, y, rows, cols, None)), &y);
      let yb = gpu.buffer(rows * cols);
      let got =
        run(gpu, &zeros, |p, yr| p.layer_norm(&xb, &n, &yb, rows, cols, Some((&rb, yr, hd))));
      assert_close(&got, &yr);
    }
  }

  #[test]
  fn im2col() {
    // Moonshine's first conv over raw samples: kernel 127, stride 64.
    let shapes =
      [(37, 20, [5, 2, 2]), (37, 20, [3, 1, 1]), (37, 20, [3, 2, 1]), (900, 1, [127, 64, 0])];
    for (t_in, ch, [kernel, stride, pad]) in shapes {
      let x = random(t_in * ch, 9);
      let t_out = (t_in + 2 * pad - kernel) / stride + 1;
      // Rows of 100, 60 or 127 columns, padded with zeros to 128, 64 or 128.
      let width = (ch * kernel).next_multiple_of(32);
      let want: Vec<f32> = (0..t_out * width)
        .map(|i| {
          let (t, c, k) = (i / width, i % width / kernel, i % width % kernel);
          let s = (stride * t + k).wrapping_sub(pad);
          if s < t_in && c < ch { x[s * ch + c] } else { 0.0 }
        })
        .collect();
      for gpu in gpus() {
        let xb = up(gpu, &x);
        let shape = [kernel, stride, pad];
        let got = run(gpu, &vec![0.0; want.len()], |p, col| p.im2col(&xb, col, t_in, ch, shape));
        assert_close(&got, &want);
      }
    }
  }

  #[test]
  fn gemm_padded_k() {
    // k = 40 is padded to 64 at load; input rows are 64 wide with zeros after 40, as from im2col.
    // A linear layer refuses it: its inputs are not padded.
    let (n, k, m) = (64, 40, 37);
    let half = |x: &f32| f16::from_f32(*x).to_f32();
    let mut a = random(m * 64, 3);
    a.chunks_mut(64).for_each(|row| row[k..].fill(0.0));
    for (ty, bytes) in weights(n, k).into_iter().take(2) {
      let file = Gguf::with_tensors(&[("w", &[k, n], ty, &bytes)]);
      let t = file.tensor("w", &[k, n]).unwrap();
      let wf: Vec<f32> = t.to_f32().unwrap().iter().map(half).collect();
      for gpu in gpus() {
        assert!(gpu.linear(std::slice::from_ref(&t), &[]).is_err());
        let l = gpu.conv(file.tensor("w", &[k, n]).unwrap(), &[]).unwrap();
        let fast = gpu.tile_rows(n, m).is_some();
        let ar: Vec<f32> = if fast { a.iter().map(half).collect() } else { a.clone() };
        let want: Vec<f32> = (0..m * n)
          .map(|i| (0..k).map(|x| ar[i / n * 64 + x] * wf[i % n * k + x]).sum())
          .collect();
        let ab = up(gpu, &a);
        let got = run(gpu, &vec![0.0; m * n], |p, c| p.gemm(&ab, &l, c, m, Epilogue::Bias));
        assert_close(&got, &want);
      }
    }
  }

  #[test]
  fn gemv_and_embed() {
    // 100 rows padded to 128 by `linear`; 288, a multiple of 32, written as they are.
    for (n, cols) in [(100, 128), (288, 288)] {
      let k = 96;
      let bias = random(n, 2);
      let a = random(k, 3);
      let pos = random(2 * k, 6);
      for (ty, bytes) in weights(n, k) {
        let file = Gguf::with_tensors(&[("w", &[k, n], ty, &bytes)]);
        let t = file.tensor("w", &[k, n]).unwrap();
        let w = t.to_f32().unwrap();
        for gpu in gpus() {
          let l = gpu.linear(&[file.tensor("w", &[k, n]).unwrap()], &bias).unwrap();
          let ab = up(gpu, &a);
          // Outputs from 3 on, and one past them that stays; the gemv doesn't round W.
          let c = random(3 + cols + 1, 4);
          for e in EPILOGUES {
            let want: Vec<f32> = (0..c.len())
              .map(|i| match i.checked_sub(3) {
                Some(j) if j < n => {
                  epilogue(e, bias[j] + (0..k).map(|x| a[x] * w[j * k + x]).sum::<f32>(), c[i])
                }
                Some(j) if j < cols => epilogue(e, 0.0, c[i]),
                _ => c[i],
              })
              .collect();
            assert_close(&run(gpu, &c, |p, out| p.gemv(&ab, &l, out, e, 3)), &want);
          }
          let pb = up(gpu, &pos);
          let want: Vec<f32> = (0..k).map(|i| w[7 * k + i] + pos[k + i] + bias[i]).collect();
          assert_close(&run(gpu, &vec![0.0; k], |p, out| p.embed(&l, &pb, out, 7, k)), &want);
        }
      }
    }
  }

  #[test]
  fn attend_and_pack() {
    let half = |x: f32| f16::from_f32(x).to_f32();
    // Moonshine's heads of 36; past 2048 keys the kernel goes in blocks.
    let keys = 4100;
    for (heads, hd) in [(2, 40), (3, 36)] {
      let d = heads * hd;
      // Rows of K then V of all heads, f16.
      let kv = random(keys * 2 * d, 7);
      let q = random(d, 8);
      for gpu in gpus() {
        let (qb, xb) = (up(gpu, &q), up(gpu, &kv));
        let cache = gpu.buffer(keys * d);
        for n in [1, 37, 1500, 2049, keys] {
          let mut want = vec![0.0; d];
          for h in 0..heads {
            let s: Vec<f32> = (0..n)
              .map(|j| {
                let k = |c: usize| half(kv[j * 2 * d + h * hd + c]);
                (0..hd).map(|c| half(q[h * hd + c]) * k(c)).sum::<f32>()
              })
              .map(|v| v / (hd as f32).sqrt())
              .collect();
            let mx = s.iter().fold(f32::MIN, |m, x| m.max(*x));
            let e: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            for c in 0..hd {
              want[h * hd + c] =
                (0..n).map(|j| e[j] * half(kv[j * 2 * d + d + h * hd + c])).sum::<f32>() / sum;
            }
          }
          let got = run(gpu, &vec![0.0; d], |p, o| {
            p.pack(&xb, &cache, keys * d, 0, 0);
            p.attend(&qb, &cache, o, n, heads, hd);
          });
          assert_close(&got, &want);
        }
      }
    }
    let kv = random(40, 7);
    for gpu in gpus() {
      let xb = up(gpu, &kv);
      // Pairs from x[3..] into y[5..].
      let y = random(5 + 10, 9);
      let got = run(gpu, &y, |p, out| p.pack(&xb, out, 10, 3, 5));
      for (i, (g, y)) in got.iter().zip(&y).enumerate() {
        let want = match i.checked_sub(5) {
          Some(j) => {
            let h = |x: f32| f16::from_f32(x).to_bits() as u32;
            h(kv[3 + 2 * j]) | h(kv[4 + 2 * j]) << 16
          }
          None => y.to_bits(),
        };
        assert_eq!(g.to_bits(), want, "pack [{i}]");
      }
    }
  }

  #[test]
  fn conv_glu() {
    let (t, ch) = (37, 192);
    let h = random(t * 2 * ch, 10);
    let (g, b) = (random(ch, 13), random(ch, 14));
    let glu = |s: usize, c: usize| h[s * 2 * ch + c] / (1.0 + (-h[s * 2 * ch + ch + c]).exp());
    for (kernel, norm) in [(5, ConvNorm::LayerNorm), (9, ConvNorm::Affine)] {
      let (w, wb) = (random(ch * kernel, 11), random(ch, 12));
      let pad = kernel / 2;
      let dw: Vec<f32> = (0..t * ch)
        .map(|i| {
          let (r, c) = (i / ch, i % ch);
          let taps = (0..kernel).filter(|k| (r + k).wrapping_sub(pad) < t);
          wb[c] + taps.map(|k| w[c * kernel + k] * glu(r + k - pad, c)).sum::<f32>()
        })
        .collect();
      let want: Vec<f32> = match norm {
        ConvNorm::LayerNorm => layer_norm(&dw, &g, &b),
        ConvNorm::Affine => dw.iter().enumerate().map(|(i, x)| x * g[i % ch] + b[i % ch]).collect(),
      };
      let want: Vec<f32> = want.into_iter().map(silu).collect();
      for gpu in gpus() {
        let (n, dw, hb) =
          (gpu.channels(&g, &b).unwrap(), gpu.channels(&w, &wb).unwrap(), up(gpu, &h));
        let got =
          run(gpu, &vec![0.0; t * ch], |p, y| p.conv_glu(&hb, &dw, kernel, &n, norm, y, t, ch));
        assert_close(&got, &want);
      }
    }
  }

  #[test]
  fn attention() {
    // head_dim 56: the flash kernel's last 8 columns stay out.
    for (t, hd) in [(37usize, 48), (100, 48), (100, 56), (5, 36), (70, 52), (100, 64), (40, 80)] {
      let heads = 2;
      let d = heads * hd;
      // The fast path reads rows up to t rounded up to 64.
      let qk = random(t.next_multiple_of(64) * 2 * d, 15);
      let v = random(t.next_multiple_of(64) * d, 16);
      let scale = 1.0 / (hd as f32).sqrt();
      let mut want = vec![0.0; t * d];
      for h in 0..heads {
        for i in 0..t {
          let q = &qk[i * 2 * d + h * hd..][..hd];
          let s: Vec<f32> = (0..t)
            .map(|j| {
              scale * q.iter().zip(&qk[j * 2 * d + d + h * hd..]).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect();
          let mx = s.iter().fold(f32::MIN, |m, x| m.max(*x));
          let e: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
          let sum: f32 = e.iter().sum();
          for c in 0..hd {
            want[i * d + h * hd + c] = (0..t).map(|j| e[j] / sum * v[j * d + h * hd + c]).sum();
          }
        }
      }
      for gpu in gpus() {
        let (qb, vb) = (up(gpu, &qk), up(gpu, &v));
        let s = gpu.buffer(gpu.scores_len(t, heads, hd, false));
        let got = run(gpu, &vec![0.0; t * d], |p, o| {
          p.attention(&qb, Some(&vb), None, &s, o, t, heads, hd)
        });
        assert_close(&got, &want);
      }
    }
  }

  #[test]
  fn relative_attention() {
    for (t, heads, hd) in [(5usize, 2, 40), (37, 2, 40), (100, 3, 128)] {
      let d = heads * hd;
      // The fast path reads rows of qkv up to t rounded up to 64 and 64 rows of pos past the end.
      let (qkv, pos) =
        (random(t.next_multiple_of(64) * 4 * d, 17), random((pos_rows(t) + 64) * d, 18));
      let scale = 1.0 / (hd as f32).sqrt();
      let dot = |a: &[f32], b: &[f32]| a[..hd].iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
      let mut want = vec![0.0; t * d];
      for h in 0..heads {
        for i in 0..t {
          let row = |j: usize, part: usize| &qkv[j * 4 * d + part * d + h * hd..];
          let s: Vec<f32> = (0..t)
            .map(|j| {
              (dot(row(i, 0), row(j, 1)) + dot(row(i, 3), &pos[(j + t + 15 - i) * d + h * hd..]))
                * scale
            })
            .collect();
          let mx = s.iter().fold(f32::MIN, |m, x| m.max(*x));
          let e: Vec<f32> = s.iter().map(|x| (x - mx).exp()).collect();
          let sum: f32 = e.iter().sum();
          for c in 0..hd {
            want[i * d + h * hd + c] = (0..t).map(|j| e[j] / sum * row(j, 2)[c]).sum();
          }
        }
      }
      for gpu in gpus() {
        let (qb, pb) = (up(gpu, &qkv), up(gpu, &pos));
        let s = gpu.buffer(gpu.scores_len(t, heads, hd, true));
        let got = run(gpu, &vec![0.0; t * d], |p, o| {
          p.attention(&qb, None, Some(&pb), &s, o, t, heads, hd)
        });
        assert_close(&got, &want);
      }
    }
  }

  /// 3×3 stride-2 conv with padding 1 per channel over `x` `[t][f][ch]`, `w` `[9][ch]`.
  fn depthwise(x: &[f32], w: &[f32], b: &[f32], t: usize, f: usize, ch: usize) -> Vec<f32> {
    let fo = half(f);
    (0..half(t) * fo * ch)
      .map(|e| {
        let (r, j, c) = (e / (fo * ch), e / ch % fo, e % ch);
        let taps = (0..9).filter_map(|k| {
          let (ti, fi) = ((2 * r + k / 3).checked_sub(1)?, (2 * j + k % 3).checked_sub(1)?);
          (ti < t && fi < f).then(|| w[k * ch + c] * x[(ti * f + fi) * ch + c])
        });
        b[c] + taps.sum::<f32>()
      })
      .collect()
  }

  #[test]
  fn subsampling() {
    let (t, f, ch) = (29, 11, 5);
    let mel = random(t * f, 19);
    let ws: Vec<_> = (0..6).map(|i| random(ch * 9, 20 + i)).collect();
    let [w0, b0, w2, b2, w5, b5] =
      std::array::from_fn(|i| &ws[i][..if i % 2 == 0 { ch * 9 } else { ch }]);
    let wide: Vec<f32> = mel.iter().flat_map(|&v| std::iter::repeat_n(v, ch)).collect();
    let first: Vec<f32> =
      depthwise(&wide, w0, b0, t, f, ch).into_iter().map(|v| v.max(0.0)).collect();
    let (t2, f2) = (half(half(t)), half(half(f)));
    let second = depthwise(&first, w2, b2, half(t), half(f), ch);
    // Four output rows of conv5.
    let want = depthwise(&second, w5, b5, t2, f2, ch);
    let x = random(t * f * ch, 26);
    let flat: Vec<f32> =
      (0..t * f * ch).map(|e| x[(e / (f * ch) * f + e % f) * ch + e % (f * ch) / f]).collect();
    for gpu in gpus() {
      let [c0, c2, c5] = [(w0, b0), (w2, b2), (w5, b5)].map(|(w, b)| gpu.channels(w, b).unwrap());
      let (mb, xb) = (up(gpu, &mel), up(gpu, &x));
      // Two chunks of two rows, each from five rows of conv2's output.
      let chunk = gpu.buffer(5 * f2 * ch);
      let got = run(gpu, &vec![0.0; want.len()], |p, y| {
        for t0 in [0, 2] {
          p.depthwise(&mb, Some(&c0), &c2, &chunk, [t, f, ch], t0, 5);
          p.depthwise(&chunk, None, &c5, y, [t2, f2, ch], t0, 2);
        }
      });
      assert_close(&got, &want);
      assert_close(&run(gpu, &vec![0.0; flat.len()], |p, y| p.flatten(&xb, y, t, f, ch)), &flat);
    }
  }

  #[test]
  fn rope() {
    // Rows [q | k | v] of 3 heads of 10, the first 6 dims of q's and k's heads rotating, from
    // position 2.
    let (rows, heads, hd, rot, pos) = (5, 3, 10, 6, 2);
    let stride = 3 * heads * hd;
    let x = random(rows * stride, 30);
    let table = random((pos + rows) * rot, 31);
    let want: Vec<f32> = (0..x.len())
      .map(|e| {
        let (r, i) = (e / stride, e % stride);
        let d = i % hd;
        if i >= 2 * heads * hd || d >= rot {
          return x[e];
        }
        let t = &table[(pos + r) * rot..];
        let (c, s) = (t[d / 2], t[rot / 2 + d / 2]);
        if d % 2 == 0 { x[e] * c - x[e + 1] * s } else { x[e - 1] * s + x[e] * c }
      })
      .collect();
    for gpu in gpus() {
      let tb = up(gpu, &table);
      let got = run(gpu, &x, |p, xb| p.rope(&tb, xb, rows, pos, [heads, hd, rot]));
      assert_close(&got, &want);
    }
  }

  #[test]
  fn group_norm_swiglu_argmax() {
    let (t, ch) = (37, 20);
    let x = random(t * ch, 32);
    let (w, b) = (random(ch, 33), random(ch, 34));
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let rstd = 1.0 / (var + 1e-5).sqrt();
    let norm: Vec<f32> =
      x.iter().enumerate().map(|(i, v)| (v - mean) * rstd * w[i % ch] + b[i % ch]).collect();
    // One row of 300 values and 300 gates, in place: the gates stay.
    let h = &x[..600];
    let glu: Vec<f32> =
      (0..600).map(|i| if i < 300 { h[i] * silu(h[300 + i]) } else { h[i] }).collect();
    // The largest value twice: the lower index wins.
    let mut logits = random(1000, 35);
    logits[700] = 2.0;
    logits[300] = 2.0;
    for gpu in gpus() {
      let c = gpu.channels(&w, &b).unwrap();
      assert_close(&run(gpu, &x, |p, xb| p.group_norm(xb, &c, t, ch, 1e-5)), &norm);
      assert_close(&run(gpu, h, |p, hb| p.swiglu(hb, 300)), &glu);
      let lb = up(gpu, &logits);
      let got = run(gpu, &[0.0], |p, out| p.argmax(&lb, out, 1000));
      assert_eq!(got[0].to_bits(), 300);
      let got = run(gpu, &[0.0], |p, out| p.argmax(&lb, out, 500));
      assert_eq!(got[0].to_bits(), 300);
    }
  }
}
