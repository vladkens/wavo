// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// gemm.wgsl on 8×8 cooperative matrices with f16 operands and F32 accumulators, as the reference's
// `kernel_mul_mm` multiplies (a and W both rounded to f16): 128 threads (four 32-lane subgroups)
// compute a ROWS×64 tile of c, each subgroup a (ROWS / 2)×32 quadrant. K advances 512 / ROWS per
// step, so staging takes 2–3 KB of shared memory, and the c tile goes out in rounds of 16 rows:
// shared memory sets occupancy. 32-row tiles serve short inputs, where 64-row ones leave too few
// workgroups or too much padding. A dispatch with m == 0 is a probe: it writes the subgroup size
// and count to c[0..2].

enable f16;
enable wgpu_cooperative_matrix;

override WTYPE: u32;
// 32 or 64, and the K step.
override ROWS: u32;
override KS: u32 = 512u / ROWS;

struct Params {
  m: u32,
  n: u32,
  k: u32,
  mode: u32,
  alpha: f32,
  dual: u32,
  offset: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<u32>;
// Q8_0 block scales as f16 pairs.
@group(0) @binding(4) var<storage, read> scales: array<u32>;

alias AM = coop_mat8x8<f16, A>;
alias BM = coop_mat8x8<f16, B>;
alias CM = coop_mat8x8<f32, C>;

// a as 8×8 row-major fragments [k / 8][row / 8] at 0, Wᵀ as fragments [k / 8][col / 8] (k as rows)
// at 512.
var<workgroup> tile: array<f16, 512u + 64u * KS>;
// 16 rows of the c tile: 8 of the upper quadrants, then 8 of the lower ones.
var<workgroup> ct: array<f32, 1024>;

// GELU as gemm.wgsl's.
fn gelu(x: f32) -> f32 {
  let z = abs(0.70710678118654752440 * x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return 0.5 * x * (1.0 + sign(x) * e);
}

fn scale(block: u32) -> f32 {
  let s = unpack2x16float(scales[block / 2u]);
  return select(s.x, s.y, (block & 1u) == 1u);
}

// Stages a[m0 + i / (KS / 4)][k0 + i % (KS / 4) · 4 ..+4].
fn stage_a(m0: u32, k0: u32, i: u32) {
  let r = i / (KS / 4u);
  let kk = i % (KS / 4u) * 4u;
  var v = vec4<f32>(0.0);
  if (m0 + r < p.m) {
    v = a[((m0 + r) * p.k + k0 + kk) / 4u];
  }
  let o = (kk / 8u * ROWS + r) * 8u + kk % 8u;
  tile[o] = f16(v.x);
  tile[o + 1u] = f16(v.y);
  tile[o + 2u] = f16(v.z);
  tile[o + 3u] = f16(v.w);
}

// Stages W[n0 + i / (KS / 4)][k0 + i % (KS / 4) · 4 ..+4], dequantized.
fn stage_w(n0: u32, k0: u32, i: u32) {
  let r = i / (KS / 4u);
  let kk = i % (KS / 4u) * 4u;
  let x = (n0 + r) * p.k + k0 + kk;
  var v: vec4<f32>;
  switch WTYPE {
    case 0u: { v = bitcast<vec4<f32>>(vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u])); }
    case 1u: { v = vec4(unpack2x16float(w[x / 2u]), unpack2x16float(w[x / 2u + 1u])); }
    default: { v = vec4<f32>(unpack4xI8(w[x / 4u])) * scale(x / 32u); }
  }
  let t = 512u + (kk / 8u * 8u + r / 8u) * 64u + kk % 8u * 8u + r % 8u;
  tile[t] = f16(v.x);
  tile[t + 8u] = f16(v.y);
  tile[t + 16u] = f16(v.z);
  tile[t + 24u] = f16(v.w);
}

// Writes `ct`, rows q·8.. of each quadrant row, through the epilogue.
fn flush(m0: u32, n0: u32, q: u32, lane: u32) {
  workgroupBarrier();
  for (var i = lane; i < 1024u; i += 128u) {
    let m = m0 + i / 512u * ROWS / 2u + q * 8u + i / 64u % 8u;
    if (m < p.m) {
      let n = n0 + i % 64u;
      let v = ct[i] + bias[n];
      let o = m * (p.n + p.dual) + n;
      switch p.mode {
        case 1u: { c[o] = v / (1.0 + exp(-v)); }
        case 2u: { c[o] = max(v, 0.0); }
        case 3u: { c[o] += p.alpha * v; }
        case 4u: { c[o] = gelu(v); }
        case 5u: { c[o] += gelu(v); }
        default: { c[o] = v; }
      }
      if (n < p.dual) {
        c[o + p.n] = ct[i] + bias[p.n + n];
      }
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(128)
fn gemm(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.m == 0u) {
    if (lane == 0u) {
      c[0] = f32(width);
      c[1] = f32(count);
    }
    return;
  }
  let n0 = wg.x * 64u;
  let m0 = wg.y * ROWS;
  // This subgroup's quadrant: rows sr..sr + ROWS / 2, columns sc..sc + 32 of the tile.
  let sr = sg / 2u * ROWS / 2u;
  let sc = sg % 2u * 32u;
  var c00: CM;
  var c01: CM;
  var c02: CM;
  var c03: CM;
  var c10: CM;
  var c11: CM;
  var c12: CM;
  var c13: CM;
  var c20: CM;
  var c21: CM;
  var c22: CM;
  var c23: CM;
  var c30: CM;
  var c31: CM;
  var c32: CM;
  var c33: CM;
  for (var k0 = 0u; k0 < p.k; k0 += KS) {
    stage_a(m0, k0, lane);
    stage_w(n0, k0, lane);
    if (KS > 8u) {
      stage_w(n0, k0, lane + 128u);
    }
    workgroupBarrier();
    for (var f = 0u; f < KS / 8u; f++) {
      let ta = (f * ROWS + sr) * 8u;
      let tb = 512u + (f * 8u + sc / 8u) * 64u;
      let a0 = coopLoadT<AM>(&tile[ta], 8u);
      let a1 = coopLoadT<AM>(&tile[ta + 64u], 8u);
      let b0 = coopLoadT<BM>(&tile[tb], 8u);
      let b1 = coopLoadT<BM>(&tile[tb + 64u], 8u);
      let b2 = coopLoadT<BM>(&tile[tb + 128u], 8u);
      let b3 = coopLoadT<BM>(&tile[tb + 192u], 8u);
      c00 = coopMultiplyAdd(a0, b0, c00);
      c01 = coopMultiplyAdd(a0, b1, c01);
      c02 = coopMultiplyAdd(a0, b2, c02);
      c03 = coopMultiplyAdd(a0, b3, c03);
      c10 = coopMultiplyAdd(a1, b0, c10);
      c11 = coopMultiplyAdd(a1, b1, c11);
      c12 = coopMultiplyAdd(a1, b2, c12);
      c13 = coopMultiplyAdd(a1, b3, c13);
      if (ROWS > 32u) {
        let a2 = coopLoadT<AM>(&tile[ta + 128u], 8u);
        let a3 = coopLoadT<AM>(&tile[ta + 192u], 8u);
        c20 = coopMultiplyAdd(a2, b0, c20);
        c21 = coopMultiplyAdd(a2, b1, c21);
        c22 = coopMultiplyAdd(a2, b2, c22);
        c23 = coopMultiplyAdd(a2, b3, c23);
        c30 = coopMultiplyAdd(a3, b0, c30);
        c31 = coopMultiplyAdd(a3, b1, c31);
        c32 = coopMultiplyAdd(a3, b2, c32);
        c33 = coopMultiplyAdd(a3, b3, c33);
      }
    }
    workgroupBarrier();
  }
  let o = sg / 2u * 512u + sc;
  coopStoreT(c00, &ct[o], 64u);
  coopStoreT(c01, &ct[o + 8u], 64u);
  coopStoreT(c02, &ct[o + 16u], 64u);
  coopStoreT(c03, &ct[o + 24u], 64u);
  flush(m0, n0, 0u, lane);
  coopStoreT(c10, &ct[o], 64u);
  coopStoreT(c11, &ct[o + 8u], 64u);
  coopStoreT(c12, &ct[o + 16u], 64u);
  coopStoreT(c13, &ct[o + 24u], 64u);
  flush(m0, n0, 1u, lane);
  if (ROWS == 32u) {
    return;
  }
  coopStoreT(c20, &ct[o], 64u);
  coopStoreT(c21, &ct[o + 8u], 64u);
  coopStoreT(c22, &ct[o + 16u], 64u);
  coopStoreT(c23, &ct[o + 24u], 64u);
  flush(m0, n0, 2u, lane);
  coopStoreT(c30, &ct[o], 64u);
  coopStoreT(c31, &ct[o + 8u], 64u);
  coopStoreT(c32, &ct[o + 16u], 64u);
  coopStoreT(c33, &ct[o + 24u], 64u);
  flush(m0, n0, 3u, lane);
}
