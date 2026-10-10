// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Decoder kernels for one token: attention of one query per head over an F16 K/V cache, and F32
// rows packed into such a cache. A cache row holds f16 pairs: K of all heads, then V.

struct Params {
  // attend: keys; pack: pairs.
  n: u32,
  head_dim: u32,
  scale: f32,
  // attend: words (f16 pairs) per cache row, and V's first word in a row.
  stride: u32,
  v_offset: u32,
  // pack: the first source float and destination word.
  src: u32,
  dst: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> at_q: array<f32>;
@group(0) @binding(1) var<storage, read> at_kv: array<u32>;
@group(0) @binding(2) var<storage, read_write> at_o: array<f32>;

// Up to 2048 keys and head_dim 128.
var<workgroup> q: array<f32, 128>;
var<workgroup> s: array<f32, 2048>;
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

// One workgroup per head: scores of the head's query against keys 0..n, softmax, the weighted sum
// of V. The query is rounded to f16, as the reference's flash attention stores it.
@compute @workgroup_size(256)
fn attend(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let hd = p.head_dim;
  let pairs = hd / 2u;
  let h = wg.x;
  for (var i = lid; i < hd; i += 256u) {
    q[i] = unpack2x16float(pack2x16float(vec2(at_q[h * hd + i], 0.0))).x;
  }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var j = lid; j < p.n; j += 256u) {
    let base = j * p.stride + h * pairs;
    var acc = 0.0;
    for (var c = 0u; c < pairs; c++) {
      let k = unpack2x16float(at_kv[base + c]);
      acc += q[2u * c] * k.x + q[2u * c + 1u] * k.y;
    }
    s[j] = acc * p.scale;
    mx = max(mx, s[j]);
  }
  mx = reduce(mx, lid, true);
  var sum = 0.0;
  for (var j = lid; j < p.n; j += 256u) {
    let e = exp(s[j] - mx);
    s[j] = e;
    sum += e;
  }
  sum = reduce(sum, lid, false);
  // Thread (g, c) sums V pair c over keys g, g + groups, ...
  let groups = 256u / pairs;
  let c = lid % pairs;
  let g = lid / pairs;
  var o = vec2(0.0);
  if (g < groups) {
    for (var j = g; j < p.n; j += groups) {
      o += s[j] * unpack2x16float(at_kv[j * p.stride + p.v_offset + h * pairs + c]);
    }
  }
  rv[lid] = o;
  workgroupBarrier();
  if (lid < pairs) {
    var t = vec2(0.0);
    for (var k = 0u; k < groups; k++) {
      t += rv[k * pairs + lid];
    }
    at_o[h * hd + 2u * lid] = t.x / sum;
    at_o[h * hd + 2u * lid + 1u] = t.y / sum;
  }
}

@group(0) @binding(0) var<storage, read> pk_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> pk_y: array<u32>;

// y[dst + i] = f16 pair of x[src + 2i], x[src + 2i + 1] for i < n.
@compute @workgroup_size(256)
fn pack(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wg.x * 256u + lid;
  if (i < p.n) {
    pk_y[p.dst + i] = pack2x16float(vec2(pk_x[p.src + 2u * i], pk_x[p.src + 2u * i + 1u]));
  }
}
