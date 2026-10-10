// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Linear layer c = epilogue(a·Wᵀ + bias): `a` is row-major [m][k] F32, W is [n][k] in its GGUF
// type. A workgroup computes a 64×64 tile of c, each thread 4×4 outputs strided by 16. K advances
// 32 per step (one Q8_0 block). n % 64 == 0 and k % 32 == 0. W is rounded to f16 after
// dequantization, as the reference's Metal matmul (`kernel_mul_mm`) does; without it, near-ties
// (one CTC frame of e2e-ctc on ru-short) fall on the other side. The reference also rounds `a`,
// which costs ~2% here and no fixture needs. The module also holds `gemv`, for one row of `a`, and
// `embed`.

// Weight type, one pipeline each: 0: F32, 1: F16, 2: Q8_0.
override WTYPE: u32;

struct Params {
  m: u32,
  n: u32,
  k: u32,
  // With v = a·Wᵀ + bias, c = v, 1: SiLU(v), 2: ReLU(v), 3: c + alpha·v, 4: GELU(v), 5: c + GELU(v).
  mode: u32,
  alpha: f32,
  // Rows of c are n + dual wide: outputs j < dual are written again at n + j with bias[n + j].
  dual: u32,
  // gemv: c's first output; embed: the first float of the position's row in `a`.
  offset: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
// F32 as bits, F16 as pairs, Q8_0 as 4 int8 per word.
@group(0) @binding(3) var<storage, read> w: array<u32>;
// Q8_0 block scales as f16 pairs (any buffer for other types).
@group(0) @binding(4) var<storage, read> scales: array<u32>;

fn to_half(x: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2(x, 0.0))).x;
}

fn scale(block: u32) -> f32 {
  let s = unpack2x16float(scales[block / 2u]);
  return select(s.x, s.y, (block & 1u) == 1u);
}

// [64][32] tiles of a and W; the padded row stride avoids shared-memory bank conflicts.
const S: u32 = 33u;
var<workgroup> at: array<f32, 2112>;
var<workgroup> wt: array<f32, 2112>;

// GELU with erf as ggml-metal computes it: `erf_approx` (Abramowitz–Stegun 7.1.26) in
// ggml/src/ggml-metal/kernels/common.h, `0.5·x·(1 + erf(x/√2))` in unary.metal.
fn gelu(x: f32) -> f32 {
  let z = abs(0.70710678118654752440 * x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return 0.5 * x * (1.0 + sign(x) * e);
}

fn store_one(i: u32, v: f32) {
  switch p.mode {
    case 1u: { c[i] = v / (1.0 + exp(-v)); }
    case 2u: { c[i] = max(v, 0.0); }
    case 3u: { c[i] += p.alpha * v; }
    case 4u: { c[i] = gelu(v); }
    case 5u: { c[i] += gelu(v); }
    default: { c[i] = v; }
  }
}

fn epilogue(m: u32, n: u32, acc: f32) {
  let i = m * (p.n + p.dual) + n;
  store_one(i, acc + bias[n]);
  if (n < p.dual) {
    c[i + p.n] = acc + bias[p.n + n];
  }
}

fn store(m: u32, n0: u32, acc: vec4<f32>) {
  if (m < p.m) {
    epilogue(m, n0, acc.x);
    epilogue(m, n0 + 16u, acc.y);
    epilogue(m, n0 + 32u, acc.z);
    epilogue(m, n0 + 48u, acc.w);
  }
}

@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let n0 = wg.x * 64u;
  let m0 = wg.y * 64u;
  let tx = lid % 16u;
  let ty = lid / 16u;
  var acc0: vec4<f32>;
  var acc1: vec4<f32>;
  var acc2: vec4<f32>;
  var acc3: vec4<f32>;
  for (var k0 = 0u; k0 < p.k; k0 += 32u) {
    for (var i = 0u; i < 8u; i++) {
      let e = lid + i * 256u;
      let r = e / 32u;
      var v = 0.0;
      if (m0 + r < p.m) {
        v = a[(m0 + r) * p.k + k0 + e % 32u];
      }
      at[r * S + e % 32u] = v;
    }
    // Each thread loads 8 weights: element e of the tile is W[n0 + e / 32][k0 + e % 32].
    for (var i = 0u; i < 2u; i++) {
      let e = (lid + i * 256u) * 4u;
      let x = (n0 + e / 32u) * p.k + k0 + e % 32u;
      var v: vec4<f32>;
      switch WTYPE {
        case 0u: { v = bitcast<vec4<f32>>(vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u])); }
        case 1u: { v = vec4(unpack2x16float(w[x / 2u]), unpack2x16float(w[x / 2u + 1u])); }
        default: { v = vec4<f32>(unpack4xI8(w[x / 4u])) * scale(x / 32u); }
      }
      let o = e / 32u * S + e % 32u;
      wt[o] = to_half(v.x);
      wt[o + 1u] = to_half(v.y);
      wt[o + 2u] = to_half(v.z);
      wt[o + 3u] = to_half(v.w);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let b = vec4(wt[tx * S + kk], wt[(tx + 16u) * S + kk], wt[(tx + 32u) * S + kk], wt[(tx + 48u) * S + kk]);
      acc0 += at[ty * S + kk] * b;
      acc1 += at[(ty + 16u) * S + kk] * b;
      acc2 += at[(ty + 32u) * S + kk] * b;
      acc3 += at[(ty + 48u) * S + kk] * b;
    }
    workgroupBarrier();
  }
  store(m0 + ty, n0 + tx, acc0);
  store(m0 + ty + 16u, n0 + tx, acc1);
  store(m0 + ty + 32u, n0 + tx, acc2);
  store(m0 + ty + 48u, n0 + tx, acc3);
}

// a[8j..8j + 8]·W[row][8j..8j + 8]. Q8_0 scales the int8 sum once per 8 weights, as ggml's
// `mul_mv` does; W is not rounded.
fn dot8(row: u32, j: u32, a0: vec4<f32>, a1: vec4<f32>) -> f32 {
  let x = row * p.k + 8u * j;
  switch WTYPE {
    case 0u: {
      let w0 = vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u]);
      let w1 = vec4(w[x + 4u], w[x + 5u], w[x + 6u], w[x + 7u]);
      return dot(bitcast<vec4<f32>>(w0), a0) + dot(bitcast<vec4<f32>>(w1), a1);
    }
    case 1u: {
      let h = x / 2u;
      let w0 = vec4(unpack2x16float(w[h]), unpack2x16float(w[h + 1u]));
      return dot(w0, a0) + dot(vec4(unpack2x16float(w[h + 2u]), unpack2x16float(w[h + 3u])), a1);
    }
    default: {
      let q = x / 4u;
      let d = dot(vec4<f32>(unpack4xI8(w[q])), a0) + dot(vec4<f32>(unpack4xI8(w[q + 1u])), a1);
      return d * scale(x / 32u);
    }
  }
}

var<workgroup> red: array<f32, 1024>;

// c[offset + n] = epilogue(a·W[n] + bias[n]) for one row of a: 8 groups of 32 threads, 4 rows of W
// per group, each thread over 8 weights at a time (8 contiguous words per 4 threads and row).
@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let lane = lid % 32u;
  let row = (wg.x * 8u + lid / 32u) * 4u;
  var s = vec4(0.0);
  for (var j = lane; j < p.k / 8u; j += 32u) {
    let a0 = vec4(a[8u * j], a[8u * j + 1u], a[8u * j + 2u], a[8u * j + 3u]);
    let a1 = vec4(a[8u * j + 4u], a[8u * j + 5u], a[8u * j + 6u], a[8u * j + 7u]);
    s += vec4(dot8(row, j, a0, a1), dot8(row + 1u, j, a0, a1), dot8(row + 2u, j, a0, a1), dot8(row + 3u, j, a0, a1));
  }
  // Rows [4][32] per group.
  let o = (lid - lane) * 4u + lane;
  red[o] = s.x;
  red[o + 32u] = s.y;
  red[o + 64u] = s.z;
  red[o + 96u] = s.w;
  workgroupBarrier();
  for (var h = 16u; h > 0u; h >>= 1u) {
    if (lane < h) {
      for (var r = 0u; r < 128u; r += 32u) {
        red[o + r] += red[o + r + h];
      }
    }
    workgroupBarrier();
  }
  if (lane < 4u) {
    store_one(p.offset + row + lane, red[o - lane + lane * 32u] + bias[row + lane]);
  }
}

// c = W[m] + a[offset..] + bias for k values: row m of an embedding table in its GGUF type, plus a
// position's row.
@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= p.k) {
    return;
  }
  let x = p.m * p.k + i;
  var v: f32;
  switch WTYPE {
    case 0u: { v = bitcast<f32>(w[x]); }
    case 1u: { v = unpack2x16float(w[x / 2u])[x % 2u]; }
    default: { v = f32(unpack4xI8(w[x / 4u])[x % 4u]) * scale(x / 32u); }
  }
  c[i] = v + a[p.offset + i] + bias[i];
}
