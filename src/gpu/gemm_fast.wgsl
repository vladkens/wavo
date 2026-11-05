// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// gemm.wgsl on F×F cooperative matrices with f16 operands and F32 accumulators, as ggml-metal's
// `kernel_mul_mm` (`ggml/src/ggml-metal/kernels/mul_mm.metal`) multiplies (a and W both rounded to
// f16): 128 threads (four 32-lane subgroups) compute a ROWS×8F tile of c, each subgroup a
// (ROWS / 2)×4F quadrant. F is 8 (Apple) or 16: the module is built with F and the matrix types
// replaced. ROWS is 4F or 8F, and K advances 8F² / ROWS per step, so staging takes 2–3 KB of shared
// memory at F = 8, and the c tile goes out in rounds of 2F rows: shared memory sets occupancy.
// 4F-row tiles serve short inputs, where 8F-row ones leave too few workgroups or too much padding.
// A dispatch with m == 0 is a probe: it writes the subgroup size and count to c[0..2].

enable f16;
enable wgpu_cooperative_matrix;

// 0: F32, 1: F16, 2: Q8_0, 3: Q5_0, 4: Q4_0, 5: Q4_1, 6: Q5_1.
override WTYPE: u32;
const F: u32 = 8u;
// 4F or 8F, and the K step.
override ROWS: u32;
override KS: u32 = 8u * F * F / ROWS;
// W has rows up to n rounded up to 64, so a tile 128 wide may pass them: its loads past them are
// skipped. Checking at F = 8 too cost GigaAM on 34 s 8% on an M2. Stores past n (c's row width,
// which may be a multiple of 32 only) are skipped at any F.
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
  // Q4 and Q5 conv rows shorter than `k`: the file's blocks use this stride, and columns from
  // here to `k` are zero. 0 means the stored row is already `k` wide.
  src_k: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<u32>;
// Q8_0, Q5_0 and Q4_0: one f16 scale per block, two per word. Q4_1 and Q5_1: scale and minimum.
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

// Same load as gemm.wgsl's q5_word and q5_4. This module is not concatenated with that one.
fn q5_word(x: u32) -> u32 {
  let block = x / 32u;
  let group = (x % 32u) / 4u;
  let bit = group * 20u;
  let byte = block * 24u + (bit >> 3u);
  let i = byte >> 2u;
  let s = (byte & 3u) * 8u;
  var word = w[i];
  if (s != 0u) {
    word = (word >> s) | (w[i + 1u] << (32u - s));
  }
  return word >> (bit & 7u);
}

fn q5_codes4(x: u32) -> vec4<f32> {
  let packed = q5_word(x);
  return vec4(f32(packed & 31u), f32((packed >> 5u) & 31u), f32((packed >> 10u) & 31u), f32((packed >> 15u) & 31u));
}

fn q5_4(x: u32) -> vec4<f32> {
  return (q5_codes4(x) - 16.0) * scale(x / 32u);
}

// Same Q4 load as gemm.wgsl. This module is not concatenated with that one.
fn q4_codes4(x: u32) -> vec4<f32> {
  let block = x / 32u;
  let local = x % 32u;
  let word = w[block * 4u + local / 8u];
  let bits = (word >> ((local % 8u) * 4u)) & 0xFFFFu;
  return vec4(f32(bits & 15u), f32((bits >> 4u) & 15u), f32((bits >> 8u) & 15u), f32((bits >> 12u) & 15u));
}

fn q4_4(x: u32) -> vec4<f32> {
  return (q4_codes4(x) - 8.0) * scale(x / 32u);
}

fn dm(block: u32) -> vec2<f32> {
  return unpack2x16float(scales[block]);
}

fn q41_4(x: u32) -> vec4<f32> {
  let s = dm(x / 32u);
  return q4_codes4(x) * s.x + s.y;
}

fn q51_4(x: u32) -> vec4<f32> {
  let s = dm(x / 32u);
  return q5_codes4(x) * s.x + s.y;
}

fn src_stride() -> u32 {
  return select(p.k, p.src_k, p.src_k != 0u);
}

fn lane(v: vec4<f32>, x: u32) -> f32 {
  switch x & 3u {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return v.z; }
    default: { return v.w; }
  }
}

fn q4_one(e: u32) -> f32 { return lane(q4_4(e & ~3u), e); }
fn q5_one(e: u32) -> f32 { return lane(q5_4(e & ~3u), e); }
fn q41_one(e: u32) -> f32 { return lane(q41_4(e & ~3u), e); }
fn q51_one(e: u32) -> f32 { return lane(q51_4(e & ~3u), e); }

fn q4_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) { return q4_4(row * p.k + col); }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q4_one(e), col < src),
    select(0.0, q4_one(e + 1u), col + 1u < src),
    select(0.0, q4_one(e + 2u), col + 2u < src),
    select(0.0, q4_one(e + 3u), col + 3u < src),
  );
}

fn q5_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) { return q5_4(row * p.k + col); }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q5_one(e), col < src),
    select(0.0, q5_one(e + 1u), col + 1u < src),
    select(0.0, q5_one(e + 2u), col + 2u < src),
    select(0.0, q5_one(e + 3u), col + 3u < src),
  );
}

fn q41_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) { return q41_4(row * p.k + col); }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q41_one(e), col < src),
    select(0.0, q41_one(e + 1u), col + 1u < src),
    select(0.0, q41_one(e + 2u), col + 2u < src),
    select(0.0, q41_one(e + 3u), col + 3u < src),
  );
}

fn q51_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) { return q51_4(row * p.k + col); }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q51_one(e), col < src),
    select(0.0, q51_one(e + 1u), col + 1u < src),
    select(0.0, q51_one(e + 2u), col + 2u < src),
    select(0.0, q51_one(e + 3u), col + 3u < src),
  );
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
      case 3u: { v = q5_at(n0 + r, k0 + kk); }
      case 4u: { v = q4_at(n0 + r, k0 + kk); }
      case 5u: { v = q41_at(n0 + r, k0 + kk); }
      case 6u: { v = q51_at(n0 + r, k0 + kk); }
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
    if (m < p.m && n < p.n) {
      let v = ct[i] + bias[n];
      let o = m * (p.n + p.dual) + n;
      switch p.mode {
        case 1u: { c[o] = v / (1.0 + exp(-v)); }
        case 2u: { c[o] = max(v, 0.0); }
        case 3u: { c[o] += p.alpha * v; }
        case 4u: { c[o] = gelu(v); }
        case 5u: { c[o] += gelu(v); }
        case 6u: { c[o] = tanh(clamp(v, -10.0, 10.0)); }
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
