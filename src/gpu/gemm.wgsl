// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Linear layer c = epilogue(a·Wᵀ + bias): `a` is row-major [m][k] F32, W is [n][k] in its GGUF
// type. A workgroup computes a 64×64 tile of c, each thread 4×4 outputs strided by 16. K advances
// 32 per step (one Q8_0 block). n % 64 == 0 and k % 32 == 0. W is rounded to f16 after
// dequantization, as the reference's Metal matmul (`kernel_mul_mm`) does; without it, near-ties
// (one CTC frame of e2e-ctc on ru-short) fall on the other side. The reference also rounds `a`,
// which costs ~2% here and no fixture needs.

// Weight type, one pipeline each: 0: F32, 1: F16, 2: Q8_0.
override WTYPE: u32;

struct Params {
  m: u32,
  n: u32,
  k: u32,
  // 0: bias, 1: bias + SiLU, 2: bias + ReLU, 3: c += alpha · (a·Wᵀ + bias)
  mode: u32,
  alpha: f32,
  // Rows of c are n + dual wide: outputs j < dual are written again at n + j with bias[n + j].
  dual: u32,
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

fn epilogue(m: u32, n: u32, acc: f32) {
  let v = acc + bias[n];
  let i = m * (p.n + p.dual) + n;
  switch p.mode {
    case 1u: { c[i] = v / (1.0 + exp(-v)); }
    case 2u: { c[i] = max(v, 0.0); }
    case 3u: { c[i] += p.alpha * v; }
    default: { c[i] = v; }
  }
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
