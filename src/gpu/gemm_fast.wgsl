// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// gemm.wgsl on F×F cooperative matrices with f16 operands and F32 accumulators, as the reference's
// `kernel_mul_mm` multiplies (a and W both rounded to f16): 128 threads (four 32-lane subgroups)
// compute a ROWS×8F tile of c, each subgroup a (ROWS / 2)×4F quadrant. F is 8 (Apple) or 16: the
// module is built with F and the matrix types replaced. ROWS is 4F or 8F, and K advances 8F² / ROWS
// per step, so staging takes 2–3 KB of shared memory at F = 8, and the c tile goes out in rounds of
// 2F rows: shared memory sets occupancy. 4F-row tiles serve short inputs, where 8F-row ones leave
// too few workgroups or too much padding. A dispatch with m == 0 is a probe: it writes the subgroup
// size and count to c[0..2].

enable f16;
enable wgpu_cooperative_matrix;

override WTYPE: u32;
const F: u32 = 8u;
// 4F or 8F, and the K step.
override ROWS: u32;
override KS: u32 = 8u * F * F / ROWS;
// W has rows up to n rounded up to 64, so a tile 128 wide may pass them. Checking at F = 8 too cost
// GigaAM on 34 s 8% on an M2.
const WIDE: bool = F > 8u;
// Halves of the a tile, where the Wᵀ tile starts.
const AT: u32 = 8u * F * F;

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

// a as F×F row-major fragments [k / F][row / F] at 0, Wᵀ as fragments [k / F][col / F] (k as rows)
// at AT.
var<workgroup> tile: array<f16, AT + 8u * F * KS>;
// 2F rows of the c tile: F of the upper quadrants, then F of the lower ones.
var<workgroup> ct: array<f32, 16u * F * F>;

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
  let o = (kk / F * ROWS + r) * F + kk % F;
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
  var v = vec4<f32>(0.0);
  if (!WIDE || n0 + r < p.n) {
    switch WTYPE {
      case 0u: { v = bitcast<vec4<f32>>(vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u])); }
      case 1u: { v = vec4(unpack2x16float(w[x / 2u]), unpack2x16float(w[x / 2u + 1u])); }
      default: { v = vec4<f32>(unpack4xI8(w[x / 4u])) * scale(x / 32u); }
    }
  }
  let t = AT + (kk / F * 8u + r / F) * F * F + kk % F * F + r % F;
  tile[t] = f16(v.x);
  tile[t + F] = f16(v.y);
  tile[t + 2u * F] = f16(v.z);
  tile[t + 3u * F] = f16(v.w);
}

// Writes `ct`, rows q·F.. of each quadrant row, through the epilogue.
fn flush(m0: u32, n0: u32, q: u32, lane: u32) {
  workgroupBarrier();
  for (var i = lane; i < 16u * F * F; i += 128u) {
    let m = m0 + i / (8u * F * F) * ROWS / 2u + q * F + i / (8u * F) % F;
    let n = n0 + i % (8u * F);
    if (m < p.m && (!WIDE || n < p.n)) {
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
  let n0 = wg.x * 8u * F;
  let m0 = wg.y * ROWS;
  // This subgroup's quadrant: rows sr..sr + ROWS / 2, columns sc..sc + 4F of the tile.
  let sr = sg / 2u * ROWS / 2u;
  let sc = sg % 2u * 4u * F;
  let ff = F * F;
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
    // Each thread stages a vec4 per 128 of a and of W. The first ones stay out of the loops: with
    // them inside, GigaAM on 4.5 s took 3% longer on an M2.
    stage_a(m0, k0, lane);
    stage_w(n0, k0, lane);
    for (var j = 1u; j < AT / 512u; j++) {
      stage_a(m0, k0, lane + j * 128u);
    }
    for (var j = 1u; j < F * KS / 64u; j++) {
      stage_w(n0, k0, lane + j * 128u);
    }
    workgroupBarrier();
    for (var f = 0u; f < KS / F; f++) {
      let ta = (f * ROWS + sr) * F;
      let tb = AT + (f * 8u + sc / F) * ff;
      let a0 = coopLoadT<AM>(&tile[ta], F);
      let a1 = coopLoadT<AM>(&tile[ta + ff], F);
      let b0 = coopLoadT<BM>(&tile[tb], F);
      let b1 = coopLoadT<BM>(&tile[tb + ff], F);
      let b2 = coopLoadT<BM>(&tile[tb + 2u * ff], F);
      let b3 = coopLoadT<BM>(&tile[tb + 3u * ff], F);
      c00 = coopMultiplyAdd(a0, b0, c00);
      c01 = coopMultiplyAdd(a0, b1, c01);
      c02 = coopMultiplyAdd(a0, b2, c02);
      c03 = coopMultiplyAdd(a0, b3, c03);
      c10 = coopMultiplyAdd(a1, b0, c10);
      c11 = coopMultiplyAdd(a1, b1, c11);
      c12 = coopMultiplyAdd(a1, b2, c12);
      c13 = coopMultiplyAdd(a1, b3, c13);
      if (ROWS > 4u * F) {
        let a2 = coopLoadT<AM>(&tile[ta + 2u * ff], F);
        let a3 = coopLoadT<AM>(&tile[ta + 3u * ff], F);
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
  let o = sg / 2u * 8u * ff + sc;
  coopStoreT(c00, &ct[o], 8u * F);
  coopStoreT(c01, &ct[o + F], 8u * F);
  coopStoreT(c02, &ct[o + 2u * F], 8u * F);
  coopStoreT(c03, &ct[o + 3u * F], 8u * F);
  flush(m0, n0, 0u, lane);
  coopStoreT(c10, &ct[o], 8u * F);
  coopStoreT(c11, &ct[o + F], 8u * F);
  coopStoreT(c12, &ct[o + 2u * F], 8u * F);
  coopStoreT(c13, &ct[o + 3u * F], 8u * F);
  flush(m0, n0, 1u, lane);
  if (ROWS == 4u * F) {
    return;
  }
  coopStoreT(c20, &ct[o], 8u * F);
  coopStoreT(c21, &ct[o + F], 8u * F);
  coopStoreT(c22, &ct[o + 2u * F], 8u * F);
  coopStoreT(c23, &ct[o + 3u * F], 8u * F);
  flush(m0, n0, 2u, lane);
  coopStoreT(c30, &ct[o], 8u * F);
  coopStoreT(c31, &ct[o + F], 8u * F);
  coopStoreT(c32, &ct[o + 2u * F], 8u * F);
  coopStoreT(c33, &ct[o + 3u * F], 8u * F);
  flush(m0, n0, 3u, lane);
}
