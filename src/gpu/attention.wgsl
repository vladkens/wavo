// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Portable multi-head attention, heads of head_dim ≤ 128 (ch = heads · head_dim). The projections
// are rows of `stride` floats with q at 0 and k at ch: [q | k], or [q + u | k | v | q + v] for
// relative positions. scores writes scale · q·k into s [heads][t][t], softmax (in ops.wgsl)
// normalizes each row in place, values computes o [t][ch] = s·v. For relative positions,
// pos_scores first writes ps [heads][t][2t − 1] = (q + v)·P, P being the projected positions
// [2t − 1][ch] (row r is position t − 1 − r), and scores adds ps[h][i][j − i + t − 1] before the
// scale.

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  stride: u32,
  // v [t][..] rows: their stride and the offset of v in them.
  v_stride: u32,
  v_offset: u32,
  relative: u32,
}

var<immediate> p: Params;

// [16][head_dim + 1] tiles of query rows and key (or position) rows.
var<workgroup> qs: array<f32, 2064>;
var<workgroup> ks: array<f32, 2064>;

// The dot of tile rows l.y of qs and l.x of ks.
fn tile_dot(l: vec3<u32>) -> f32 {
  let hd = p.head_dim;
  workgroupBarrier();
  var acc = 0.0;
  for (var c = 0u; c < hd; c++) {
    acc += qs[l.y * (hd + 1u) + c] * ks[l.x * (hd + 1u) + c];
  }
  return acc;
}

@group(0) @binding(0) var<storage, read> sc_qk: array<f32>;
@group(0) @binding(1) var<storage, read> sc_ps: array<f32>;
@group(0) @binding(2) var<storage, read_write> sc_s: array<f32>;

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
      q = sc_qk[(i0 + r) * p.stride + h * hd + c];
    }
    if (j0 + r < p.t) {
      k = sc_qk[(j0 + r) * p.stride + p.ch + h * hd + c];
    }
    qs[r * (hd + 1u) + c] = q;
    ks[r * (hd + 1u) + c] = k;
  }
  var acc = tile_dot(l);
  let i = i0 + l.y;
  let j = j0 + l.x;
  if (i < p.t && j < p.t) {
    if (p.relative != 0u) {
      acc += sc_ps[(h * p.t + i) * (2u * p.t - 1u) + j + p.t - 1u - i];
    }
    sc_s[(h * p.t + i) * p.t + j] = acc * p.scale;
  }
}

@group(0) @binding(0) var<storage, read> pos_q: array<f32>;
@group(0) @binding(1) var<storage, read> pos_p: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_s: array<f32>;

// One workgroup per 16 queries × 16 positions of one head; q + v is the last ch of each row.
@compute @workgroup_size(16, 16)
fn pos_scores(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let hd = p.head_dim;
  let h = wg.z;
  let i0 = wg.y * 16u;
  let r0 = wg.x * 16u;
  let n = 2u * p.t - 1u;
  for (var e = l.y * 16u + l.x; e < 16u * hd; e += 256u) {
    let r = e / hd;
    let c = e % hd;
    var q = 0.0;
    var k = 0.0;
    if (i0 + r < p.t) {
      q = pos_q[(i0 + r) * p.stride + p.stride - p.ch + h * hd + c];
    }
    if (r0 + r < n) {
      k = pos_p[(r0 + r) * p.ch + h * hd + c];
    }
    qs[r * (hd + 1u) + c] = q;
    ks[r * (hd + 1u) + c] = k;
  }
  let acc = tile_dot(l);
  let i = i0 + l.y;
  let r = r0 + l.x;
  if (i < p.t && r < n) {
    pos_s[(h * p.t + i) * n + r] = acc;
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
        x = va_v[(j0 + e / hd) * p.v_stride + p.v_offset + h * hd + e % hd];
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
