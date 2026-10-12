// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Linear layer c = epilogue(a·Wᵀ + bias): `a` is row-major [m][k] F32, W is [n][k] in its GGUF
// type. A workgroup computes a ROWS×128 tile of c. K advances 32 per step (one Q8_0 block). W
// has a multiple of 64 rows and k % 32 == 0; c's rows are n + dual
// wide, and outputs from n on (W's zero rows) are not written. W is rounded to f16 after
// dequantization, as ggml-metal's `kernel_mul_mm` (`ggml/src/ggml-metal/kernels/mul_mm.metal`)
// does; without it, near-ties (one CTC frame of e2e-ctc on ru-short) fall on the other side. The
// reference also rounds `a`, which costs ~2% here and no fixture needs. The module also holds
// `gemv`, for one row of `a`, and `embed`.

// Weight type, one pipeline each: 0: F32, 1: F16, 2: Q8_0.
override WTYPE: u32;
// Rows of the GEMM's tile: 64, or 32 (more workgroups).
override ROWS: u32 = 64u;

struct Params {
  m: u32,
  // Output columns per row of c (before the dual ones).
  n: u32,
  k: u32,
  // With v = a·Wᵀ + bias, c = v, 1: SiLU(v), 2: ReLU(v), 3: c + alpha·v, 4: GELU(v), 5: c + GELU(v),
  // 6: tanh(v).
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
// `a` for the GEMM's 16-byte loads.
@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;

// Rounds to the nearest f16, ties to even, by hand: NVIDIA's Vulkan driver folds
// unpack2x16float(pack2x16float(x)) into x and AMD's truncates. Normal halves keep 10 mantissa
// bits, subnormal ones are multiples of 2^-24. Weights never exceed the f16 range.
fn to_half(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  let normal = bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
  return select(normal, round(x * 0x1p24) * 0x1p-24, abs(x) < 0x1p-14);
}

fn scale(block: u32) -> f32 {
  let s = unpack2x16float(scales[block / 2u]);
  return select(s.x, s.y, (block & 1u) == 1u);
}

// The [ROWS][32] tile of a as [row][k quad], rows padded to 9 quads against bank conflicts; the
// [32][128] tile of W as [k][column quad].
var<workgroup> at: array<vec4<f32>, 576>;
var<workgroup> wt: array<vec4<f32>, 1024>;

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
    // tanh(±10) is ±1 in f32; some drivers' tanh overflows to NaN on large inputs.
    case 6u: { c[i] = tanh(clamp(v, -10.0, 10.0)); }
    default: { c[i] = v; }
  }
}

fn epilogue(m: u32, n: u32, acc: f32) {
  if (n >= p.n) {
    return;
  }
  let i = m * (p.n + p.dual) + n;
  store_one(i, acc + bias[n]);
  if (n < p.dual) {
    c[i + p.n] = acc + bias[p.n + n];
  }
}

fn store(m: u32, n0: u32, acc: vec4<f32>) {
  if (m < p.m) {
    epilogue(m, n0, acc.x);
    epilogue(m, n0 + 1u, acc.y);
    epilogue(m, n0 + 2u, acc.z);
    epilogue(m, n0 + 3u, acc.w);
  }
}

// The words of W[n][k..k + 4]; for Q8_0 one, the word of its block's scale and the block's
// parity. Zeros from row p.n on.
fn w_words(n: u32, k: u32) -> vec4<u32> {
  if (n >= p.n) {
    return vec4(0u);
  }
  let x = n * p.k + k;
  switch WTYPE {
    case 0u: { return vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u]); }
    case 1u: { return vec4(w[x / 2u], w[x / 2u + 1u], 0u, 0u); }
    default: { return vec4(w[x / 4u], scales[x / 64u], (x / 32u) & 1u, 0u); }
  }
}

// The 4 weights of `w_words`, rounded.
fn w_floats(v: vec4<u32>) -> vec4<f32> {
  var x: vec4<f32>;
  switch WTYPE {
    case 0u: { x = bitcast<vec4<f32>>(v); }
    case 1u: { x = vec4(unpack2x16float(v.x), unpack2x16float(v.y)); }
    default: {
      let s = unpack2x16float(v.y);
      x = vec4<f32>(unpack4xI8(v.x)) * select(s.x, s.y, v.z == 1u);
    }
  }
  return vec4(to_half(x.x), to_half(x.y), to_half(x.z), to_half(x.w));
}

// What a thread stages per K step: a quads of rows r and r + 32, and W words of 4 rows.
struct Fetch {
  a0: vec4<f32>,
  a1: vec4<f32>,
  w0: vec4<u32>,
  w1: vec4<u32>,
  w2: vec4<u32>,
  w3: vec4<u32>,
}

// Thread lid's share of the K step from k0: a rows m0 + lid / 8 (and + 32) at quad lid % 8, W rows
// from n0 + 4·(lid / 8) at the quad lid % 8.
fn fetch(m0: u32, n0: u32, k0: u32, lid: u32) -> Fetch {
  var f: Fetch;
  let r = m0 + lid / 8u;
  let k = k0 + 4u * (lid % 8u);
  if (r < p.m) {
    f.a0 = a4[(r * p.k + k) / 4u];
  }
  if (ROWS == 64u && r + 32u < p.m) {
    f.a1 = a4[((r + 32u) * p.k + k) / 4u];
  }
  let n = n0 + 4u * (lid / 8u);
  f.w0 = w_words(n, k);
  f.w1 = w_words(n + 1u, k);
  f.w2 = w_words(n + 2u, k);
  f.w3 = w_words(n + 3u, k);
  return f;
}

// A workgroup computes a ROWS×128 tile of c: thread (tx, ty) = (lid % 16, lid / 16) rows ty + 16i
// and columns 4tx.. and 64 + 4tx.., four each, summed over k in order. The next K step's loads
// are issued before this one's products, which hides their latency on GPUs with few workgroups.
@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let n0 = wg.x * 128u;
  let m0 = wg.y * ROWS;
  let tx = lid % 16u;
  let ty = lid / 16u;
  var c00: vec4<f32>;
  var c01: vec4<f32>;
  var c10: vec4<f32>;
  var c11: vec4<f32>;
  var c20: vec4<f32>;
  var c21: vec4<f32>;
  var c30: vec4<f32>;
  var c31: vec4<f32>;
  let r = lid / 8u;
  let kq = 4u * (lid % 8u);
  var f = fetch(m0, n0, 0u, lid);
  for (var k0 = 0u; k0 < p.k; k0 += 32u) {
    at[r * 9u + kq / 4u] = f.a0;
    if (ROWS == 64u) {
      at[(r + 32u) * 9u + kq / 4u] = f.a1;
    }
    // W transposed to [k][column quad].
    let x0 = w_floats(f.w0);
    let x1 = w_floats(f.w1);
    let x2 = w_floats(f.w2);
    let x3 = w_floats(f.w3);
    wt[kq * 32u + r] = vec4(x0.x, x1.x, x2.x, x3.x);
    wt[(kq + 1u) * 32u + r] = vec4(x0.y, x1.y, x2.y, x3.y);
    wt[(kq + 2u) * 32u + r] = vec4(x0.z, x1.z, x2.z, x3.z);
    wt[(kq + 3u) * 32u + r] = vec4(x0.w, x1.w, x2.w, x3.w);
    if (k0 + 32u < p.k) {
      f = fetch(m0, n0, k0 + 32u, lid);
    }
    workgroupBarrier();
    for (var kq = 0u; kq < 8u; kq++) {
      let a0 = at[ty * 9u + kq];
      let a1 = at[(ty + 16u) * 9u + kq];
      var a2: vec4<f32>;
      var a3: vec4<f32>;
      if (ROWS == 64u) {
        a2 = at[(ty + 32u) * 9u + kq];
        a3 = at[(ty + 48u) * 9u + kq];
      }
      for (var j = 0u; j < 4u; j++) {
        let b0 = wt[(4u * kq + j) * 32u + tx];
        let b1 = wt[(4u * kq + j) * 32u + 16u + tx];
        c00 += a0[j] * b0;
        c01 += a0[j] * b1;
        c10 += a1[j] * b0;
        c11 += a1[j] * b1;
        if (ROWS == 64u) {
          c20 += a2[j] * b0;
          c21 += a2[j] * b1;
          c30 += a3[j] * b0;
          c31 += a3[j] * b1;
        }
      }
    }
    workgroupBarrier();
  }
  let n = n0 + 4u * tx;
  store(m0 + ty, n, c00);
  store(m0 + ty, n + 64u, c01);
  store(m0 + ty + 16u, n, c10);
  store(m0 + ty + 16u, n + 64u, c11);
  if (ROWS == 64u) {
    store(m0 + ty + 32u, n, c20);
    store(m0 + ty + 32u, n + 64u, c21);
    store(m0 + ty + 48u, n, c30);
    store(m0 + ty + 48u, n + 64u, c31);
  }
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

// The sums of s over each group of 32 threads (lid / 32), for all its threads. subgroup.wgsl has
// the fast version.
fn group_sums(s: vec4<f32>, lid: u32) -> vec4<f32> {
  let lane = lid % 32u;
  // [4][32] per group.
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
  return vec4(red[o - lane], red[o - lane + 32u], red[o - lane + 64u], red[o - lane + 96u]);
}

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
  let t = group_sums(s, lid);
  if (lane < 4u) {
    store_one(p.offset + row + lane, t[lane] + bias[row + lane]);
  }
}

@group(0) @binding(5) var<storage, read> ids: array<u32>;

// c = W[ids[m]] + a[offset..] + bias for k values: a row of an embedding table in its GGUF type,
// plus a position's row.
@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= p.k) {
    return;
  }
  let x = ids[p.m] * p.k + i;
  var v: f32;
  switch WTYPE {
    case 0u: { v = bitcast<f32>(w[x]); }
    case 1u: { v = unpack2x16float(w[x / 2u])[x % 2u]; }
    default: { v = f32(unpack4xI8(w[x / 4u])[x % 4u]) * scale(x / 32u); }
  }
  c[i] = v + a[p.offset + i] + bias[i];
}
