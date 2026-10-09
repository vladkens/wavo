// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! wgpu device, weight upload, and a recorder that encodes kernel dispatches into one compute pass.
//! Each kernel has a portable version; a fast one (subgroups, cooperative matrices) replaces it
//! when the device supports it and a probe dispatch of that pipeline succeeds.

use std::sync::mpsc;

pub use wgpu::{BindGroup, Buffer};

use crate::error::{Result, bail};
use crate::gguf::{Tensor, Type};

type Pipeline = wgpu::ComputePipeline;

pub struct Gpu {
  device: wgpu::Device,
  queue: wgpu::Queue,
  /// Usage of weight buffers. On unified memory they are also `MAP_WRITE`, so filling their
  /// mapping writes the buffer itself: no staging copy at load, none queued for the first call.
  weights: wgpu::BufferUsages,
  kernels: Kernels,
  fast: Fast,
}

struct Kernels {
  /// One per weight type.
  gemm: Vec<Pipeline>,
  layer_norm: Pipeline,
  layer_norm_rope: Pipeline,
  im2col: Pipeline,
  conv_glu: Pipeline,
  scores: Pipeline,
  pos_scores: Pipeline,
  softmax: Pipeline,
  values: Pipeline,
  depthwise: Pipeline,
  first_depthwise: Pipeline,
  flatten: Pipeline,
}

#[derive(Default)]
struct Fast {
  gemm: Option<Vec<Pipeline>>,
  layer_norm: Option<Pipeline>,
  layer_norm_rope: Option<Pipeline>,
  conv_glu: Option<Pipeline>,
  /// Flash attention for head_dim ≤ 64.
  attention: Option<Pipeline>,
}

/// A linear layer `y = x·Wᵀ + b` with `W` row-major `[n][k]`, kept in its GGUF type. Q8_0 is
/// stored as int8 data (4 per u32, row-major) plus one f16 scale per 32-weight block.
pub struct Linear {
  n: usize,
  k: usize,
  /// GEMM pipeline index: 0: F32, 1: F16, 2: Q8_0
  ty: usize,
  weight: Buffer,
  scales: Option<Buffer>,
  bias: Buffer,
}

impl Linear {
  /// Output columns: the rows of `W` padded to a multiple of 64.
  pub fn width(&self) -> usize {
    self.n
  }
}

/// LayerNorm gain and bias, or a per-channel affine `x·g + b`.
pub struct Norm {
  g: Buffer,
  b: Buffer,
}

/// A depthwise conv's weight `[ch][taps]` and bias.
pub struct Depthwise {
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
}

impl Gpu {
  pub fn new() -> Result<Self> {
    Self::with_fast_paths(true)
  }

  pub(crate) fn with_fast_paths(fast: bool) -> Result<Self> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
      power_preference: wgpu::PowerPreference::HighPerformance,
      ..Default::default()
    }))?;
    let coop = wgpu::Features::SUBGROUP
      | wgpu::Features::SUBGROUP_BARRIER
      | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let fast = fast
      && adapter.features().contains(coop)
      && adapter.cooperative_matrix_properties().iter().any(|p| {
        (p.m_size, p.n_size, p.k_size) == (8, 8, 8)
          && p.ab_type == wgpu::CooperativeScalarType::F32
          && p.cr_type == wgpu::CooperativeScalarType::F32
      });
    let mappable = adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu
      && adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
    let mut features = wgpu::Features::IMMEDIATES;
    if fast {
      features |= coop;
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
      ..Default::default()
    }))?;

    let gemm = module(&device, include_str!("gemm.wgsl"));
    let ops = module(&device, include_str!("ops.wgsl"));
    let attention = module(&device, include_str!("attention.wgsl"));
    let spatial = module(&device, include_str!("spatial.wgsl"));
    let kernels = Kernels {
      gemm: gemms(&device, &gemm),
      layer_norm: pipeline(&device, &ops, "layer_norm", &[]),
      layer_norm_rope: pipeline(&device, &ops, "layer_norm_rope", &[]),
      im2col: pipeline(&device, &ops, "im2col", &[]),
      conv_glu: pipeline(&device, &ops, "conv_glu", &[]),
      scores: pipeline(&device, &attention, "scores", &[]),
      pos_scores: pipeline(&device, &attention, "pos_scores", &[]),
      softmax: pipeline(&device, &ops, "softmax", &[]),
      values: pipeline(&device, &attention, "values", &[]),
      depthwise: pipeline(&device, &spatial, "depthwise", &[]),
      first_depthwise: pipeline(&device, &spatial, "first_depthwise", &[]),
      flatten: pipeline(&device, &spatial, "flatten", &[]),
    };
    let mut weights = wgpu::BufferUsages::STORAGE;
    if mappable {
      weights |= wgpu::BufferUsages::MAP_WRITE;
    }
    let mut gpu = Self { device, queue, weights, kernels, fast: Fast::default() };
    if fast {
      gpu.fast = gpu.fast_kernels();
    }
    Ok(gpu)
  }

  /// Compiles the fast kernels and keeps each one whose probe reports four 32-lane subgroups.
  fn fast_kernels(&self) -> Fast {
    let scopes = [wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation]
      .map(|f| self.device.push_error_scope(f));
    let gemm = module(&self.device, include_str!("gemm_fast.wgsl"));
    let ops = module(&self.device, include_str!("ops_fast.wgsl"));
    let gemm = gemms(&self.device, &gemm);
    let kernels = ["layer_norm", "layer_norm_rope", "conv_glu", "attention"]
      .map(|entry| pipeline(&self.device, &ops, entry, &[]));
    let errors = scopes.into_iter().rev().filter_map(|s| pollster::block_on(s.pop())).count();
    if errors > 0 {
      return Fast::default();
    }
    let [layer_norm, layer_norm_rope, conv_glu, attention] = kernels;
    // All params zero: the kernel sees an empty problem and reports its subgroup layout.
    let probe = |k: Pipeline, bindings: usize, out: usize, params: usize| {
      let buffers: Vec<Buffer> = (0..bindings).map(|_| self.buffer(16)).collect();
      let refs: Vec<&Buffer> = buffers.iter().collect();
      let read = self.readback(2);
      let r = self.run(&mut Vec::new(), &buffers[out], &read, 2, |p| {
        p.run(&k, &refs, &[0; 6][..params], [1, 1, 1]);
      });
      matches!(r.as_deref(), Ok([32.0, 4.0])).then_some(k)
    };
    Fast {
      gemm: gemm.into_iter().map(|k| probe(k, 5, 2, 5)).collect(),
      layer_norm: probe(layer_norm, 4, 3, 4),
      layer_norm_rope: probe(layer_norm_rope, 6, 3, 4),
      conv_glu: probe(conv_glu, 6, 5, 6),
      attention: probe(attention, 3, 2, 4),
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
  /// the rows are padded with zeros to a multiple of 64; the extra outputs are `epilogue(0)`.
  pub fn linear(&self, weights: &[Tensor], bias: &[f32]) -> Result<Linear> {
    let w0 = &weights[0];
    let k = w0.len() / w0.dims.last().unwrap();
    let rows = weights.iter().map(|w| w.dims.last().unwrap()).sum::<usize>();
    if weights.iter().any(|w| w.ty != w0.ty || w.len() / w.dims.last().unwrap() != k) {
      bail!("stacked weights differ in type or shape");
    }
    if !k.is_multiple_of(32) {
      bail!("linear layer {k}→{rows}: needs k % 32 == 0");
    }
    // Buffers start zeroed, so the padding rows only need the space.
    let n = rows.next_multiple_of(64);
    // The file is streamed in pieces of 8192 Q8_0 blocks straight into the mapped buffers.
    const PIECE: usize = 34 << 13;
    let (ty, weight, scales) = match w0.ty {
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
        let (mut d, mut s) =
          (Vec::with_capacity(PIECE / 34 * 32), Vec::with_capacity(PIECE / 34 * 2));
        let mut block = 0;
        for w in weights {
          w.read(PIECE, |i, piece| {
            d.clear();
            s.clear();
            for b in piece.as_chunks::<34>().0 {
              s.extend_from_slice(&b[..2]);
              d.extend_from_slice(&b[2..]);
            }
            let first = block + i / 34;
            dv.slice(first * 32..first * 32 + d.len()).copy_from_slice(&d);
            sv.slice(first * 2..first * 2 + s.len()).copy_from_slice(&s);
          })?;
          block += w.len() / 32;
        }
        drop((dv, sv));
        data.unmap();
        scales.unmap();
        (2, data, Some(scales))
      }
    };
    if !bias.is_empty() && bias.len() != rows {
      bail!("linear layer {k}→{rows}: bias has {} values", bias.len());
    }
    let mut padded = vec![0.0; n];
    padded[..bias.len()].copy_from_slice(bias);
    Ok(Linear { n, k, ty, weight, scales, bias: self.upload(&padded)? })
  }

  pub fn norm(&self, g: &[f32], b: &[f32]) -> Result<Norm> {
    Ok(Norm { g: self.upload(g)?, b: self.upload(b)? })
  }

  pub fn depthwise(&self, w: &[f32], b: &[f32]) -> Result<Depthwise> {
    Ok(Depthwise { w: self.upload(w)?, b: self.upload(b)? })
  }

  fn flash(&self, head_dim: usize) -> Option<&Pipeline> {
    self.fast.attention.as_ref().filter(|_| head_dim <= 64 && head_dim.is_multiple_of(8))
  }

  /// Floats of attention scratch for `t` frames: the portable path keeps all scores.
  pub fn scores_len(&self, t: usize, heads: usize, head_dim: usize) -> usize {
    if self.flash(head_dim).is_some() { 1 } else { heads * t * t }
  }

  /// Encodes one compute pass with `record`, runs it, and reads back `len` f32s of `out` through
  /// the `read` buffer. `groups` caches bind groups by dispatch index, so a caller must record the
  /// same dispatch sequence over the same buffers each time, and clear it when buffers change.
  pub fn run(
    &self,
    groups: &mut Vec<BindGroup>,
    out: &Buffer,
    read: &Buffer,
    len: usize,
    record: impl FnOnce(&mut Pass),
  ) -> Result<Vec<f32>> {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    record(&mut Pass {
      gpu: self,
      pass: encoder.begin_compute_pass(&Default::default()),
      groups,
      next: 0,
    });
    let size = (len * 4) as u64;
    encoder.copy_buffer_to_buffer(out, 0, read, 0, size);
    self.queue.submit([encoder.finish()]);

    let slice = read.slice(..size);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    self.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv().unwrap()?;
    let data = bytemuck::cast_slice(&slice.get_mapped_range()?).to_vec();
    read.unmap();
    Ok(data)
  }
}

/// Output length of a stride-2 conv with kernel 3 and padding 1 (or kernel 5 and padding 2).
pub fn half(n: usize) -> usize {
  (n - 1) / 2 + 1
}

/// attention.wgsl's params: `[stride, v_stride, v_offset, relative]` after t, ch, head_dim and the
/// scale.
fn attention_params(t: usize, ch: usize, head_dim: usize, rest: [usize; 4]) -> [u32; 8] {
  let scale = 1.0 / (head_dim as f32).sqrt();
  let [a, b, c, d] = rest.map(|v| v as u32);
  [t as u32, ch as u32, head_dim as u32, scale.to_bits(), a, b, c, d]
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

/// The `gemm` entry specialized for each weight type.
fn gemms(device: &wgpu::Device, module: &wgpu::ShaderModule) -> Vec<Pipeline> {
  (0..3).map(|ty| pipeline(device, module, "gemm", &[("WTYPE", ty as f64)])).collect()
}

/// Records kernel dispatches into one compute pass.
pub struct Pass<'a> {
  gpu: &'a Gpu,
  pass: wgpu::ComputePass<'a>,
  groups: &'a mut Vec<BindGroup>,
  next: usize,
}

impl Pass<'_> {
  /// Dispatches `kernel` with `buffers` bound in order and `params` as immediates.
  fn run(&mut self, kernel: &Pipeline, buffers: &[&Buffer], params: &[u32], [x, y, z]: [usize; 3]) {
    if self.next == self.groups.len() {
      let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
      let layout = kernel.get_bind_group_layout(0);
      let desc = wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries };
      self.groups.push(self.gpu.device.create_bind_group(&desc));
    }
    self.pass.set_pipeline(kernel);
    self.pass.set_bind_group(0, &self.groups[self.next], &[]);
    self.pass.set_immediates(0, bytemuck::cast_slice(params));
    self.pass.dispatch_workgroups(x as u32, y as u32, z as u32);
    self.next += 1;
  }

  /// `c = epilogue(a·Wᵀ + b)` for `m` rows of `a`.
  pub fn gemm(&mut self, a: &Buffer, l: &Linear, c: &Buffer, m: usize, epilogue: Epilogue) {
    let gpu = self.gpu;
    let (mode, alpha) = match epilogue {
      Epilogue::Bias => (0, 0.0f32),
      Epilogue::Silu => (1, 0.0),
      Epilogue::Relu => (2, 0.0),
      Epilogue::Residual(alpha) => (3, alpha),
    };
    let params = [m as u32, l.n as u32, l.k as u32, mode, alpha.to_bits()];
    // Only Q8_0 reads scales; other types bind the bias in their place.
    let buffers = [a, &l.bias, c, &l.weight, l.scales.as_ref().unwrap_or(&l.bias)];
    match &gpu.fast.gemm {
      Some(k) => self.run(&k[l.ty], &buffers, &params, [l.n / 64, m.div_ceil(32), 1]),
      None => self.run(&gpu.kernels.gemm[l.ty], &buffers, &params, [l.n / 64, m.div_ceil(64), 1]),
    }
  }

  /// LayerNorm over `rows` rows of `cols`.
  pub fn layer_norm(&mut self, x: &Buffer, n: &Norm, y: &Buffer, rows: usize, cols: usize) {
    let gpu = self.gpu;
    let params = [rows as u32, cols as u32];
    match &gpu.fast.layer_norm {
      Some(k) => self.run(k, &[x, &n.g, &n.b, y], &params, [rows.div_ceil(4), 1, 1]),
      None => self.run(&gpu.kernels.layer_norm, &[x, &n.g, &n.b, y], &params, [rows, 1, 1]),
    }
  }

  /// LayerNorm into `y`, and the same with rotary embedding into `yr`. `rope` holds per position
  /// `head_dim / 2` cosines followed by as many sines.
  #[allow(clippy::too_many_arguments)]
  pub fn layer_norm_rope(
    &mut self,
    x: &Buffer,
    n: &Norm,
    y: &Buffer,
    rope: &Buffer,
    yr: &Buffer,
    rows: usize,
    cols: usize,
    head_dim: usize,
  ) {
    let gpu = self.gpu;
    let params = [rows as u32, cols as u32, head_dim as u32];
    let buffers = [x, &n.g, &n.b, y, rope, yr];
    match &gpu.fast.layer_norm_rope {
      Some(k) => self.run(k, &buffers, &params, [rows.div_ceil(4), 1, 1]),
      None => self.run(&gpu.kernels.layer_norm_rope, &buffers, &params, [rows, 1, 1]),
    }
  }

  /// Columns for a conv1d with kernel 5, stride 2, padding 2 over `x` `[t_in][ch]`.
  pub fn im2col(&mut self, x: &Buffer, col: &Buffer, t_in: usize, ch: usize, t_out: usize) {
    let k = &self.gpu.kernels.im2col;
    self.run(k, &[x, col], &[t_in as u32, ch as u32], [(ch * 5).div_ceil(256), t_out, 1]);
  }

  /// Conformer conv module between the pointwise convs: GLU, depthwise conv (odd `kernel`,
  /// centred) + bias, `norm`, SiLU. `h` is `[t][2·ch]`, `y` is `[t][ch]`.
  #[allow(clippy::too_many_arguments)]
  pub fn conv_glu(
    &mut self,
    h: &Buffer,
    dw: &Depthwise,
    kernel: usize,
    n: &Norm,
    norm: ConvNorm,
    y: &Buffer,
    t: usize,
    ch: usize,
  ) {
    let gpu = self.gpu;
    let buffers = [h, &dw.w, &dw.b, &n.g, &n.b, y];
    let layer_norm = (norm == ConvNorm::LayerNorm) as u32;
    let params = [t as u32, ch as u32, 0, 0, kernel as u32, layer_norm];
    match &gpu.fast.conv_glu {
      Some(k) => self.run(k, &buffers, &params, [t.div_ceil(4), 1, 1]),
      None => self.run(&gpu.kernels.conv_glu, &buffers, &params, [t, 1, 1]),
    }
  }

  /// Multi-head attention: `qk` is `[t][q | k]`, `v` and `o` are `[t][heads · head_dim]`, `s`
  /// holds `scores_len` floats. The fast path reads rows of `qk` and `v` up to `t` rounded up to
  /// 64, which must be finite.
  #[allow(clippy::too_many_arguments)]
  pub fn attention(
    &mut self,
    qk: &Buffer,
    v: &Buffer,
    s: &Buffer,
    o: &Buffer,
    t: usize,
    heads: usize,
    head_dim: usize,
  ) {
    let ch = heads * head_dim;
    let params = attention_params(t, ch, head_dim, [2 * ch, ch, 0, 0]);
    if let Some(k) = self.gpu.flash(head_dim) {
      return self.run(k, &[qk, v, o], &params[..4], [t.div_ceil(64), heads, 1]);
    }
    self.scores_softmax_values([qk, qk, v], s, o, t, heads, &params);
  }

  /// Multi-head attention with relative positions: `qkv` is `[t][q + u | k | v | q + v]`, `pos`
  /// the projected positions `[2t − 1][heads · head_dim]` (row r is position t − 1 − r), `ps`
  /// holds `heads · t · (2t − 1)` floats and `s` `heads · t · t`. Score (i, j) is
  /// `((q + u)_i·k_j + (q + v)_i·pos[j − i + t − 1]) / √head_dim`.
  #[allow(clippy::too_many_arguments)]
  pub fn relative_attention(
    &mut self,
    qkv: &Buffer,
    pos: &Buffer,
    ps: &Buffer,
    s: &Buffer,
    o: &Buffer,
    t: usize,
    heads: usize,
    head_dim: usize,
  ) {
    let ch = heads * head_dim;
    let params = attention_params(t, ch, head_dim, [4 * ch, 4 * ch, 2 * ch, 1]);
    let k = &self.gpu.kernels.pos_scores;
    self.run(k, &[qkv, pos, ps], &params, [(2 * t - 1).div_ceil(16), t.div_ceil(16), heads]);
    self.scores_softmax_values([qkv, ps, qkv], s, o, t, heads, &params);
  }

  /// The portable attention: scores from `[qk, ps]` into `s`, softmax, values from `v` into `o`.
  fn scores_softmax_values(
    &mut self,
    [qk, ps, v]: [&Buffer; 3],
    s: &Buffer,
    o: &Buffer,
    t: usize,
    heads: usize,
    params: &[u32],
  ) {
    let (k, tiles) = (&self.gpu.kernels, t.div_ceil(16));
    self.run(&k.scores, &[qk, ps, s], params, [tiles, tiles, heads]);
    self.run(&k.softmax, &[s], &params[..1], [t, heads, 1]);
    self.run(&k.values, &[s, v, o], params, [tiles, heads, 1]);
  }

  /// Depthwise 3×3 conv, stride 2, padding 1, + bias over `x` `[t][f][ch]` into `y`
  /// `[⌈t/2⌉][⌈f/2⌉][ch]`.
  pub fn depthwise(
    &mut self,
    x: &Buffer,
    dw: &Depthwise,
    y: &Buffer,
    t: usize,
    f: usize,
    ch: usize,
  ) {
    let k = &self.gpu.kernels.depthwise;
    self.run(k, &[x, &dw.w, &dw.b, y], &[t as u32, f as u32, ch as u32], [half(f), half(t), 1]);
  }

  /// `depthwise` over ReLU(`first` + bias), a 3×3 stride-2 conv from the one-channel image `x`
  /// `[t][f]` to ch channels, whose output is never stored. `y` is `[⌈t/4⌉][⌈f/4⌉][ch]`.
  #[allow(clippy::too_many_arguments)]
  pub fn first_depthwise(
    &mut self,
    x: &Buffer,
    first: &Depthwise,
    dw: &Depthwise,
    y: &Buffer,
    t: usize,
    f: usize,
    ch: usize,
  ) {
    let k = &self.gpu.kernels.first_depthwise;
    let buffers = [x, &dw.w, &dw.b, y, &first.w, &first.b];
    let grid = [half(half(f)), half(half(t)), 1];
    self.run(k, &buffers, &[t as u32, f as u32, ch as u32], grid);
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

  /// The portable path, then the fast one.
  fn gpus() -> &'static [Gpu; 2] {
    static GPUS: OnceLock<[Gpu; 2]> = OnceLock::new();
    GPUS.get_or_init(|| {
      let fast = Gpu::new().unwrap();
      let f = &fast.fast;
      let ops = [&f.layer_norm, &f.layer_norm_rope, &f.conv_glu, &f.attention];
      assert!(
        f.gemm.is_some() && ops.iter().all(|k| k.is_some()),
        "a fast kernel failed its probe"
      );
      [Gpu::with_fast_paths(false).unwrap(), fast]
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

  #[test]
  fn gemm() {
    // 100 rows: the GEMM pads them to 128 output columns, which come out as epilogue(0).
    let (n, k, width) = (100, 96, 128);
    let w = random(n * k, 1);
    let bias = random(n, 2);
    let f32s: &[u8] = bytemuck::cast_slice(&w);
    let f16s: Vec<u8> = w.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect();
    let q8: Vec<u8> = (w.chunks(32))
      .flat_map(|block| {
        let d = block.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
        let q = block.iter().map(move |x| (x / d).round() as i8 as u8);
        f16::from_f32(d).to_le_bytes().into_iter().chain(q)
      })
      .collect();
    let epilogues = [Epilogue::Bias, Epilogue::Silu, Epilogue::Relu, Epilogue::Residual(0.5)];
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
      // With the bias, and without: zeros.
      for (bias, b) in [(&bias, &bias[..]), (&vec![0.0; n], &[][..])] {
        for gpu in gpus() {
          let l = gpu.linear(&[t("w1"), t("w2")], b).unwrap();
          assert_eq!(l.width(), width);
          for m in [1, 37, 70] {
            let a = random(m * k, 3);
            let c = random(m * width, 4);
            let ab = up(gpu, &a);
            for e in epilogues {
              let want: Vec<f32> = (0..m * width)
                .map(|i| {
                  let (r, j) = (i / width, i % width);
                  let dot = |j: usize| (0..k).map(|x| a[r * k + x] * wf[j * k + x]).sum::<f32>();
                  let v = if j < n { bias[j] + dot(j) } else { 0.0 };
                  match e {
                    Epilogue::Bias => v,
                    Epilogue::Silu => silu(v),
                    Epilogue::Relu => v.max(0.0),
                    Epilogue::Residual(alpha) => c[i] + alpha * v,
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
      let n = gpu.norm(&g, &b).unwrap();
      let (xb, rb) = (up(gpu, &x), up(gpu, &rope));
      let zeros = vec![0.0; rows * cols];
      assert_close(&run(gpu, &zeros, |p, y| p.layer_norm(&xb, &n, y, rows, cols)), &y);
      let yb = gpu.buffer(rows * cols);
      let got = run(gpu, &zeros, |p, yr| p.layer_norm_rope(&xb, &n, &yb, &rb, yr, rows, cols, hd));
      assert_close(&got, &yr);
    }
  }

  #[test]
  fn im2col() {
    let (t_in, ch) = (37, 20);
    let t_out = (t_in - 1) / 2 + 1;
    let x = random(t_in * ch, 9);
    let want: Vec<f32> = (0..t_out * ch * 5)
      .map(|i| {
        let (t, c, k) = (i / (ch * 5), i % (ch * 5) / 5, i % 5);
        let s = (2 * t + k).wrapping_sub(2);
        if s < t_in { x[s * ch + c] } else { 0.0 }
      })
      .collect();
    for gpu in gpus() {
      let xb = up(gpu, &x);
      let got = run(gpu, &vec![0.0; want.len()], |p, col| p.im2col(&xb, col, t_in, ch, t_out));
      assert_close(&got, &want);
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
        let (n, dw, hb) = (gpu.norm(&g, &b).unwrap(), gpu.depthwise(&w, &wb).unwrap(), up(gpu, &h));
        let got =
          run(gpu, &vec![0.0; t * ch], |p, y| p.conv_glu(&hb, &dw, kernel, &n, norm, y, t, ch));
        assert_close(&got, &want);
      }
    }
  }

  #[test]
  fn attention() {
    let (heads, hd) = (2, 48);
    let d = heads * hd;
    for t in [37usize, 100] {
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
        let s = gpu.buffer(gpu.scores_len(t, heads, hd));
        let got = run(gpu, &vec![0.0; t * d], |p, o| p.attention(&qb, &vb, &s, o, t, heads, hd));
        assert_close(&got, &want);
      }
    }
  }

  #[test]
  fn relative_attention() {
    let (t, heads, hd) = (37, 2, 40);
    let d = heads * hd;
    let (qkv, pos) = (random(t * 4 * d, 17), random((2 * t - 1) * d, 18));
    let scale = 1.0 / (hd as f32).sqrt();
    let dot = |a: &[f32], b: &[f32]| a[..hd].iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut want = vec![0.0; t * d];
    for h in 0..heads {
      for i in 0..t {
        let row = |j: usize, part: usize| &qkv[j * 4 * d + part * d + h * hd..];
        let s: Vec<f32> = (0..t)
          .map(|j| {
            (dot(row(i, 0), row(j, 1)) + dot(row(i, 3), &pos[(j + t - 1 - i) * d + h * hd..]))
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
      let (ps, s) = (gpu.buffer(heads * t * (2 * t - 1)), gpu.buffer(heads * t * t));
      let got = run(gpu, &vec![0.0; t * d], |p, o| {
        p.relative_attention(&qb, &pb, &ps, &s, o, t, heads, hd)
      });
      assert_close(&got, &want);
    }
  }

  /// 3×3 stride-2 conv with padding 1 per channel over `x` `[t][f][ch]`, `w` `[ch][9]`.
  fn depthwise(x: &[f32], w: &[f32], b: &[f32], t: usize, f: usize, ch: usize) -> Vec<f32> {
    let fo = half(f);
    (0..half(t) * fo * ch)
      .map(|e| {
        let (r, j, c) = (e / (fo * ch), e / ch % fo, e % ch);
        let taps = (0..9).filter_map(|k| {
          let (ti, fi) = ((2 * r + k / 3).checked_sub(1)?, (2 * j + k % 3).checked_sub(1)?);
          (ti < t && fi < f).then(|| w[c * 9 + k] * x[(ti * f + fi) * ch + c])
        });
        b[c] + taps.sum::<f32>()
      })
      .collect()
  }

  #[test]
  fn subsampling() {
    let (t, f, ch) = (13, 11, 5);
    let mel = random(t * f, 19);
    let ws: Vec<_> = (0..6).map(|i| random(ch * 9, 20 + i)).collect();
    let [w0, b0, w2, b2, w5, b5] =
      std::array::from_fn(|i| &ws[i][..if i % 2 == 0 { ch * 9 } else { ch }]);
    let wide: Vec<f32> = mel.iter().flat_map(|&v| std::iter::repeat_n(v, ch)).collect();
    let first: Vec<f32> =
      depthwise(&wide, w0, b0, t, f, ch).into_iter().map(|v| v.max(0.0)).collect();
    let want = depthwise(&first, w2, b2, half(t), half(f), ch);
    let x = random(t * f * ch, 26);
    let want5 = depthwise(&x, w5, b5, t, f, ch);
    let flat: Vec<f32> =
      (0..t * f * ch).map(|e| x[(e / (f * ch) * f + e % f) * ch + e % (f * ch) / f]).collect();
    for gpu in gpus() {
      let [c0, c2, c5] = [(w0, b0), (w2, b2), (w5, b5)].map(|(w, b)| gpu.depthwise(w, b).unwrap());
      let (mb, xb) = (up(gpu, &mel), up(gpu, &x));
      let n = want.len();
      assert_close(
        &run(gpu, &vec![0.0; n], |p, y| p.first_depthwise(&mb, &c0, &c2, y, t, f, ch)),
        &want,
      );
      assert_close(
        &run(gpu, &vec![0.0; want5.len()], |p, y| p.depthwise(&xb, &c5, y, t, f, ch)),
        &want5,
      );
      assert_close(&run(gpu, &vec![0.0; flat.len()], |p, y| p.flatten(&xb, y, t, f, ch)), &flat);
    }
  }
}
