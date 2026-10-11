// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Row-wise and convolution kernels over time-major [t][ch] F32 activations.

// One immediate struct for all kernels: naga sizes a module's immediates from its first one.
struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  // conv_glu: the depthwise kernel size, and 1 for LayerNorm or 0 for the affine x·g + b.
  kernel: u32,
  layer_norm: u32,
  // im2col: the conv's stride and padding (and `kernel`).
  stride: u32,
  pad: u32,
}

var<immediate> p: Params;

var<workgroup> red: array<f32, 256>;
// One row of up to 1024 channels.
var<workgroup> row: array<f32, 1024>;

// Sum (or max) over a 256-thread workgroup.
fn reduce(v: f32, lid: u32, is_max: bool) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      if (is_max) {
        red[lid] = max(red[lid], red[lid + s]);
      } else {
        red[lid] += red[lid + s];
      }
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

// LayerNorm (eps 1e-5) of row t, one workgroup per row. With head_dim > 0 it also writes the
// normalized row with rotary embedding: NEOX split-half rotation inside each head, using a
// [t][cos | sin] table of head_dim / 2 frequencies each.

@group(0) @binding(0) var<storage, read> norm_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_g: array<f32>;
@group(0) @binding(2) var<storage, read> norm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_y: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_yr: array<f32>;

@compute @workgroup_size(256)
fn layer_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x;
  let base = t * p.ch;
  var s = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    s += norm_x[base + i];
  }
  let mean = reduce(s, lid, false) / f32(p.ch);
  var q = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    let d = norm_x[base + i] - mean;
    q += d * d;
  }
  let rstd = inverseSqrt(reduce(q, lid, false) / f32(p.ch) + 1e-5);
  for (var i = lid; i < p.ch; i += 256u) {
    let v = (norm_x[base + i] - mean) * rstd * norm_g[i] + norm_b[i];
    norm_y[base + i] = v;
    if (p.head_dim != 0u) {
      row[i] = v;
    }
  }
  if (p.head_dim == 0u) {
    return;
  }
  workgroupBarrier();
  let half = p.head_dim / 2u;
  for (var i = lid; i < p.ch; i += 256u) {
    let d = i % p.head_dim;
    let f = d % half;
    let sn = rope[t * p.head_dim + half + f];
    var v = row[i] * rope[t * p.head_dim + f];
    if (d < half) {
      v -= row[i + half] * sn;
    } else {
      v += row[i - half] * sn;
    }
    norm_yr[base + i] = v;
  }
}

// im2col for a conv1d over x [t][ch]: col[i][c · kernel + k] = x[stride · i + k − pad][c], matching
// a ggml conv weight [kernel, ch, out]. Rows are padded with zeros to a multiple of 32 columns, as
// `Gpu::conv` pads the weight (80 mels: 240 → 256).

@group(0) @binding(0) var<storage, read> col_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> col_y: array<f32>;

@compute @workgroup_size(256)
fn im2col(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wg.y;
  let e = wg.x * 256u + lid;
  let width = (p.ch * p.kernel + 31u) / 32u * 32u;
  if (e >= width) {
    return;
  }
  // Source frame plus the padding.
  let s = p.stride * i + e % p.kernel;
  var v = 0.0;
  if (e < p.ch * p.kernel && s >= p.pad && s - p.pad < p.t) {
    v = col_x[(s - p.pad) * p.ch + e / p.kernel];
  }
  col_y[i * width + e] = v;
}

// Conformer conv module between the pointwise convs, one workgroup per frame: GLU (first half ·
// sigmoid(second half)) of h [t][2·ch], depthwise conv (odd kernel, centred, weight [ch][kernel])
// plus bias, LayerNorm over channels or a per-channel affine, SiLU.

@group(0) @binding(0) var<storage, read> conv_h: array<f32>;
@group(0) @binding(1) var<storage, read> conv_w: array<f32>;
@group(0) @binding(2) var<storage, read> conv_b: array<f32>;
@group(0) @binding(3) var<storage, read> conv_g: array<f32>;
@group(0) @binding(4) var<storage, read> conv_beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_y: array<f32>;

@compute @workgroup_size(256)
fn conv_glu(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x;
  let pad = p.kernel / 2u;
  var s = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    var acc = conv_b[i];
    for (var k = 0u; k < p.kernel; k++) {
      // Source frame plus the padding.
      let src = t + k;
      if (src >= pad && src - pad < p.t) {
        let base = (src - pad) * 2u * p.ch;
        acc += conv_w[i * p.kernel + k] * conv_h[base + i] / (1.0 + exp(-conv_h[base + p.ch + i]));
      }
    }
    row[i] = acc;
    s += acc;
  }
  let mean = reduce(s, lid, false) / f32(p.ch);
  var q = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    let d = row[i] - mean;
    q += d * d;
  }
  let rstd = inverseSqrt(reduce(q, lid, false) / f32(p.ch) + 1e-5);
  for (var i = lid; i < p.ch; i += 256u) {
    var y = row[i] * conv_g[i] + conv_beta[i];
    if (p.layer_norm != 0u) {
      y = (row[i] - mean) * rstd * conv_g[i] + conv_beta[i];
    }
    conv_y[t * p.ch + i] = y / (1.0 + exp(-y));
  }
}

// Softmax of each row of s [heads][t][t] in place (see attention.wgsl).

@group(0) @binding(0) var<storage, read_write> sm_s: array<f32>;

// One workgroup per row (query wg.x, head wg.y).
@compute @workgroup_size(256)
fn softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let base = (wg.y * p.t + wg.x) * p.t;
  var mx = -3.0e38;
  for (var j = lid; j < p.t; j += 256u) {
    mx = max(mx, sm_s[base + j]);
  }
  mx = reduce(mx, lid, true);
  var sum = 0.0;
  for (var j = lid; j < p.t; j += 256u) {
    let e = exp(sm_s[base + j] - mx);
    sm_s[base + j] = e;
    sum += e;
  }
  let inv = 1.0 / reduce(sum, lid, false);
  for (var j = lid; j < p.t; j += 256u) {
    sm_s[base + j] *= inv;
  }
}

// Writes the call's number p.ch to slot p.t of mk_marks once mk_after (bound only to order this
// after its last writer) is written: proof that a submission ran to its end.

@group(0) @binding(0) var<storage, read> mk_after: array<f32>;
@group(0) @binding(1) var<storage, read_write> mk_marks: array<u32>;

@compute @workgroup_size(1)
fn mark() {
  mk_marks[p.t] = p.ch | (bitcast<u32>(mk_after[0]) & 0u);
}
