// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Row-wise, convolution and attention kernels over time-major [t][ch] F32 activations.

// One immediate struct for all kernels: naga sizes a module's immediates from its first one.
struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
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

// LayerNorm (eps 1e-5) of row t, one workgroup per row. layer_norm_rope also writes the normalized
// row with rotary embedding: NEOX split-half rotation inside each head, using a [t][cos | sin]
// table of head_dim / 2 frequencies each.

@group(0) @binding(0) var<storage, read> norm_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_g: array<f32>;
@group(0) @binding(2) var<storage, read> norm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_y: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_yr: array<f32>;

fn normalize_row(t: u32, lid: u32) {
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
    row[i] = v;
  }
}

@compute @workgroup_size(256)
fn layer_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  normalize_row(wg.x, lid);
}

@compute @workgroup_size(256)
fn layer_norm_rope(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x;
  normalize_row(t, lid);
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
    norm_yr[t * p.ch + i] = v;
  }
}

// im2col for a conv1d with kernel 5, stride 2, padding 2 over x [t][ch]:
// col[i][c * 5 + k] = x[2i + k - 2][c], matching a ggml conv weight [k, ch, out].

@group(0) @binding(0) var<storage, read> col_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> col_y: array<f32>;

@compute @workgroup_size(256)
fn im2col(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wg.y;
  let e = wg.x * 256u + lid;
  if (e >= p.ch * 5u) {
    return;
  }
  // Source frame plus the padding of 2.
  let s = 2u * i + e % 5u;
  var v = 0.0;
  if (s >= 2u && s - 2u < p.t) {
    v = col_x[(s - 2u) * p.ch + e / 5u];
  }
  col_y[i * p.ch * 5u + e] = v;
}

// Conformer conv module between the pointwise convs, one workgroup per frame: GLU (first half ·
// sigmoid(second half)) of h [t][2·ch], depthwise conv (kernel 5, padding 2, weight [ch][5]) plus
// bias, LayerNorm over channels, SiLU.

@group(0) @binding(0) var<storage, read> conv_h: array<f32>;
@group(0) @binding(1) var<storage, read> conv_w: array<f32>;
@group(0) @binding(2) var<storage, read> conv_b: array<f32>;
@group(0) @binding(3) var<storage, read> conv_g: array<f32>;
@group(0) @binding(4) var<storage, read> conv_beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_y: array<f32>;

@compute @workgroup_size(256)
fn conv_glu(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x;
  var s = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    var acc = conv_b[i];
    for (var k = 0u; k < 5u; k++) {
      // Source frame plus the padding of 2.
      let src = t + k;
      if (src >= 2u && src - 2u < p.t) {
        let base = (src - 2u) * 2u * p.ch;
        acc += conv_w[i * 5u + k] * conv_h[base + i] / (1.0 + exp(-conv_h[base + p.ch + i]));
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
    let y = (row[i] - mean) * rstd * conv_g[i] + conv_beta[i];
    conv_y[t * p.ch + i] = y / (1.0 + exp(-y));
  }
}

// Attention with heads of head_dim ≤ 128 (ch = heads · head_dim): qk is [t][q | k] (2·ch wide),
// v and o are [t][ch], s is [heads][t][t]. scores writes scale · q·k, softmax normalizes each row
// of s in place, values computes o = s·v.

@group(0) @binding(0) var<storage, read> sc_qk: array<f32>;
@group(0) @binding(1) var<storage, read_write> sc_s: array<f32>;

// [16][head_dim + 1] tiles of q and k rows.
var<workgroup> qs: array<f32, 2064>;
var<workgroup> ks: array<f32, 2064>;

// One workgroup per 16×16 scores of one head.
@compute @workgroup_size(16, 16)
fn scores(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let hd = p.head_dim;
  let h = wg.z;
  let i0 = wg.y * 16u;
  let j0 = wg.x * 16u;
  for (var e = l.y * 16u + l.x; e < 16u * hd; e += 256u) {
    let r = e / hd;
    let c = e % hd;
    var q = 0.0;
    var k = 0.0;
    if (i0 + r < p.t) {
      q = sc_qk[(i0 + r) * 2u * p.ch + h * hd + c];
    }
    if (j0 + r < p.t) {
      k = sc_qk[(j0 + r) * 2u * p.ch + p.ch + h * hd + c];
    }
    qs[r * (hd + 1u) + c] = q;
    ks[r * (hd + 1u) + c] = k;
  }
  workgroupBarrier();
  var acc = 0.0;
  for (var c = 0u; c < hd; c++) {
    acc += qs[l.y * (hd + 1u) + c] * ks[l.x * (hd + 1u) + c];
  }
  let i = i0 + l.y;
  let j = j0 + l.x;
  if (i < p.t && j < p.t) {
    sc_s[(h * p.t + i) * p.t + j] = acc * p.scale;
  }
}

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

@group(0) @binding(0) var<storage, read> va_s: array<f32>;
@group(0) @binding(1) var<storage, read> va_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> va_o: array<f32>;

// [16 queries][16 keys] probabilities (padded) and [16 keys][head_dim] values.
var<workgroup> ps: array<f32, 272>;
var<workgroup> vs: array<f32, 2048>;

// One workgroup per 16 queries (wg.x) of one head (wg.y); thread (x, y) owns query y and
// columns x + 16u.
@compute @workgroup_size(16, 16)
fn values(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let hd = p.head_dim;
  let h = wg.y;
  let i = wg.x * 16u + l.y;
  var acc: array<f32, 8>;
  for (var j0 = 0u; j0 < p.t; j0 += 16u) {
    var pv = 0.0;
    if (i < p.t && j0 + l.x < p.t) {
      pv = va_s[(h * p.t + i) * p.t + j0 + l.x];
    }
    ps[l.y * 17u + l.x] = pv;
    for (var e = l.y * 16u + l.x; e < 16u * hd; e += 256u) {
      var x = 0.0;
      if (j0 + e / hd < p.t) {
        x = va_v[(j0 + e / hd) * p.ch + h * hd + e % hd];
      }
      vs[e] = x;
    }
    workgroupBarrier();
    for (var jj = 0u; jj < 16u; jj++) {
      let pj = ps[l.y * 17u + jj];
      for (var u = 0u; u < 8u; u++) {
        let c = l.x + 16u * u;
        if (c < hd) {
          acc[u] += pj * vs[jj * hd + c];
        }
      }
    }
    workgroupBarrier();
  }
  if (i < p.t) {
    for (var u = 0u; u < 8u; u++) {
      let c = l.x + 16u * u;
      if (c < hd) {
        va_o[i * p.ch + h * hd + c] = acc[u];
      }
    }
  }
}
