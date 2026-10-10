// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Portable multi-head attention, heads of head_dim ≤ 128 (ch = heads · head_dim). The projections
// are rows of `stride` floats with q at 0 and k at ch: [q | k], or [q + u | k | v | q + v] for
// relative positions. scores writes scale · q·k into s [heads][t][t], softmax (in ops.wgsl)
// normalizes each row in place, values computes o [t][ch] = s·v. For relative positions, scores
// adds (q + v)_i·P[j − i + t − 1] before the scale, P being the projected positions [2t − 1][ch]
// from row 16 on (row 16 + r is position t − 1 − r).

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

@group(0) @binding(0) var<storage, read> sc_qk: array<f32>;
@group(0) @binding(1) var<storage, read> sc_pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> sc_s: array<f32>;

// [16][head_dim + 1] tiles of query rows and key rows.
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
      q = sc_qk[(i0 + r) * p.stride + h * hd + c];
    }
    if (j0 + r < p.t) {
      k = sc_qk[(j0 + r) * p.stride + p.ch + h * hd + c];
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
    if (p.relative != 0u) {
      let qv = i * p.stride + 3u * p.ch + h * hd;
      let pr = (j + p.t + 15u - i) * p.ch + h * hd;
      var pos = 0.0;
      for (var c = 0u; c < hd; c++) {
        pos += sc_qk[qv + c] * sc_pos[pr + c];
      }
      acc += pos;
    }
    sc_s[(h * p.t + i) * p.t + j] = acc * p.scale;
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
