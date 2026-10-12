// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Decoder kernels for one token: attention of one query per head over an F16 K/V cache, split over
// workgroups of keys, and F32 rows packed into such a cache. A cache of `cols` positions with d
// dims (all heads) holds f16 pairs: K transposed, word i·cols + j for pair i of position j, then
// V by position, word (d/2)·cols + j·d/2 + i. With K transposed, the threads of `attend` (one key
// each) read consecutive words.

struct Params {
  // attend: keys; pack: rows.
  n: u32,
  // attend: dims per head; pack: d.
  dim: u32,
  scale: f32,
  // The cache's positions.
  cols: u32,
  // pack: the position of the first row, and the first source float and floats per source row.
  col: u32,
  src: u32,
  stride: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> at_q: array<f32>;
@group(0) @binding(1) var<storage, read> at_kv: array<u32>;
@group(0) @binding(2) var<storage, read_write> at_part: array<f32>;

// Keys per workgroup.
const KEYS: u32 = 256u;

// A key per thread, head_dim up to 128.
var<workgroup> q: array<f32, 128>;
var<workgroup> s: array<f32, 256>;
var<workgroup> rs: array<f32, 256>;
var<workgroup> rv: array<vec2<f32>, 256>;

// Sum (or max) over the 256-thread workgroup.
fn reduce(v: f32, lid: u32, is_max: bool) -> f32 {
  rs[lid] = v;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (lid < h) {
      if (is_max) {
        rs[lid] = max(rs[lid], rs[lid + h]);
      } else {
        rs[lid] += rs[lid + h];
      }
    }
    workgroupBarrier();
  }
  let r = rs[0];
  workgroupBarrier();
  return r;
}

// Workgroup (c, h): head h's query against its keys c·KEYS.. (up to n): the scores' maximum m,
// the sum l of exp(score − m) and the sum of exp(score − m)·V, as part [h][c][head_dim + 2]:
// the sums of V, then m and l. `combine` merges the parts of each head. The query is rounded to
// f16, as the reference's flash attention stores it.
@compute @workgroup_size(256)
fn attend(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let hd = p.dim;
  let pairs = hd / 2u;
  let h = wg.y;
  let k0 = h * pairs * p.cols;
  let v_stride = num_workgroups.y * pairs;
  let v0 = v_stride * p.cols + h * pairs;
  for (var i = lid; i < hd; i += 256u) {
    q[i] = unpack2x16float(pack2x16float(vec2(at_q[h * hd + i], 0.0))).x;
  }
  workgroupBarrier();
  let j0 = wg.x * KEYS;
  let n = min(p.n - j0, KEYS);
  var x = -3.0e38;
  if (lid < n) {
    var acc = 0.0;
    for (var i = 0u; i < pairs; i++) {
      let k = unpack2x16float(at_kv[k0 + i * p.cols + j0 + lid]);
      acc += q[2u * i] * k.x + q[2u * i + 1u] * k.y;
    }
    x = acc * p.scale;
  }
  let m = reduce(x, lid, true);
  var e = 0.0;
  if (lid < n) {
    e = exp(x - m);
  }
  s[lid] = e;
  let l = reduce(e, lid, false);
  // Thread (g, c) sums V pair c over keys g, g + groups, ...
  let groups = 256u / pairs;
  let c = lid % pairs;
  let g = lid / pairs;
  var o = vec2(0.0);
  if (g < groups) {
    for (var j = g; j < n; j += groups) {
      o += s[j] * unpack2x16float(at_kv[v0 + (j0 + j) * v_stride + c]);
    }
  }
  rv[lid] = o;
  workgroupBarrier();
  let part = (h * ((p.n + KEYS - 1u) / KEYS) + wg.x) * (hd + 2u);
  if (lid < pairs) {
    var t = vec2(0.0);
    for (var k = 0u; k < groups; k++) {
      t += rv[k * pairs + lid];
    }
    at_part[part + 2u * lid] = t.x;
    at_part[part + 2u * lid + 1u] = t.y;
  }
  if (lid == 0u) {
    at_part[part + hd] = m;
    at_part[part + hd + 1u] = l;
  }
}

@group(0) @binding(0) var<storage, read> cb_part: array<f32>;
@group(0) @binding(1) var<storage, read_write> cb_o: array<f32>;

// One workgroup per head: attend's parts rescaled to their common maximum, summed and normalized.
@compute @workgroup_size(64)
fn combine(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let hd = p.dim;
  let parts = (p.n + KEYS - 1u) / KEYS;
  let base = wg.x * parts * (hd + 2u);
  var m = -3.0e38;
  for (var c = 0u; c < parts; c++) {
    m = max(m, cb_part[base + c * (hd + 2u) + hd]);
  }
  var l = 0.0;
  for (var c = 0u; c < parts; c++) {
    let at = base + c * (hd + 2u) + hd;
    l += exp(cb_part[at] - m) * cb_part[at + 1u];
  }
  for (var i = lid; i < hd; i += 64u) {
    var o = 0.0;
    for (var c = 0u; c < parts; c++) {
      let at = base + c * (hd + 2u);
      o += exp(cb_part[at + hd] - m) * cb_part[at + i];
    }
    cb_o[wg.x * hd + i] = o / l;
  }
}

@group(0) @binding(0) var<storage, read> pk_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> pk_y: array<u32>;

// n rows of x, each K then V (2·d floats from src + j·stride), into the cache at positions col..:
// pair i of row j is x[2i], x[2i + 1] (of V for i ≥ d/2). One row: a thread per pair; more: 64
// rows × 4 pairs per workgroup, so the transposed K is written in runs of 64 positions.
@compute @workgroup_size(256)
fn pack(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  var i = wg.x * 256u + lid;
  var j = 0u;
  if (p.n > 1u) {
    i = wg.y * 4u + lid / 64u;
    j = wg.x * 64u + lid % 64u;
  }
  let half = p.dim / 2u;
  if (i >= p.dim || j >= p.n) {
    return;
  }
  let x = p.src + j * p.stride + 2u * i;
  var at = i * p.cols + p.col + j;
  if (i >= half) {
    at = half * p.cols + (p.col + j) * half + i - half;
  }
  pk_y[at] = pack2x16float(vec2(pk_x[x], pk_x[x + 1u]));
}
