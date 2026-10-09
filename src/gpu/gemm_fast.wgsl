// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// gemm.wgsl on 8×8 F32 cooperative matrices: 128 threads (four 32-lane subgroups) compute a 32×64
// tile of c, each subgroup a 16×32 quadrant. K advances 16 per step. A dispatch with m == 0 is a
// probe: it writes the subgroup size and count to c[0..2].

enable wgpu_cooperative_matrix;

override WTYPE: u32;

struct Params {
  m: u32,
  n: u32,
  k: u32,
  mode: u32,
  alpha: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<u32>;
// Q8_0 block scales as f16 pairs.
@group(0) @binding(4) var<storage, read> scales: array<u32>;

fn scale(block: u32) -> f32 {
  let s = unpack2x16float(scales[block / 2u]);
  return select(s.x, s.y, (block & 1u) == 1u);
}

alias AM = coop_mat8x8<f32, A>;
alias BM = coop_mat8x8<f32, B>;
alias CM = coop_mat8x8<f32, C>;

// During the K loop: a as 8×8 row-major fragments [k / 8][row / 8] at 0, and Wᵀ as fragments
// [k / 8][col / 8] (k as rows) at 512. After it: the c tile as row-major [32][64].
var<workgroup> tile: array<f32, 2048>;

// Stages a[m0 + i / 16][k0 + i % 16 ..+4].
fn load_a(m0: u32, k0: u32, i: u32) {
  var v = vec4<f32>(0.0);
  if (m0 + i / 16u < p.m) {
    v = a[((m0 + i / 16u) * p.k + k0 + i % 16u) / 4u];
  }
  let o = (i % 16u / 8u) * 256u + (i / 128u) * 64u + (i / 16u % 8u) * 8u + i % 8u;
  tile[o] = v.x;
  tile[o + 1u] = v.y;
  tile[o + 2u] = v.z;
  tile[o + 3u] = v.w;
}

// Stages W[n0 + i / 16][k0 + i % 16 ..+4], dequantized.
fn load_w(n0: u32, k0: u32, i: u32) {
  let x = (n0 + i / 16u) * p.k + k0 + i % 16u;
  var v: vec4<f32>;
  switch WTYPE {
    case 0u: { v = bitcast<vec4<f32>>(vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u])); }
    case 1u: { v = vec4(unpack2x16float(w[x / 2u]), unpack2x16float(w[x / 2u + 1u])); }
    default: { v = vec4<f32>(unpack4xI8(w[x / 4u])) * scale(x / 32u); }
  }
  let o = 512u + (i % 16u / 8u) * 512u + (i / 128u) * 64u + (i % 8u) * 8u + i / 16u % 8u;
  tile[o] = v.x;
  tile[o + 8u] = v.y;
  tile[o + 16u] = v.z;
  tile[o + 24u] = v.w;
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
  let m0 = wg.y * 32u;
  // This subgroup's quadrant: rows sr..sr+16, columns sc..sc+32 of the tile.
  let sr = (sg / 2u) * 16u;
  let sc = (sg % 2u) * 32u;
  var c00: CM;
  var c01: CM;
  var c02: CM;
  var c03: CM;
  var c10: CM;
  var c11: CM;
  var c12: CM;
  var c13: CM;
  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    load_a(m0, k0, lane * 4u);
    load_w(n0, k0, lane * 4u);
    load_w(n0, k0, lane * 4u + 512u);
    workgroupBarrier();
    for (var f = 0u; f < 2u; f++) {
      let a0 = coopLoadT<AM>(&tile[f * 256u + sr * 8u], 8u);
      let a1 = coopLoadT<AM>(&tile[f * 256u + sr * 8u + 64u], 8u);
      let b = 512u + f * 512u + sc * 8u;
      let b0 = coopLoadT<BM>(&tile[b], 8u);
      let b1 = coopLoadT<BM>(&tile[b + 64u], 8u);
      let b2 = coopLoadT<BM>(&tile[b + 128u], 8u);
      let b3 = coopLoadT<BM>(&tile[b + 192u], 8u);
      c00 = coopMultiplyAdd(a0, b0, c00);
      c01 = coopMultiplyAdd(a0, b1, c01);
      c02 = coopMultiplyAdd(a0, b2, c02);
      c03 = coopMultiplyAdd(a0, b3, c03);
      c10 = coopMultiplyAdd(a1, b0, c10);
      c11 = coopMultiplyAdd(a1, b1, c11);
      c12 = coopMultiplyAdd(a1, b2, c12);
      c13 = coopMultiplyAdd(a1, b3, c13);
    }
    workgroupBarrier();
  }
  coopStoreT(c00, &tile[sr * 64u + sc], 64u);
  coopStoreT(c01, &tile[sr * 64u + sc + 8u], 64u);
  coopStoreT(c02, &tile[sr * 64u + sc + 16u], 64u);
  coopStoreT(c03, &tile[sr * 64u + sc + 24u], 64u);
  coopStoreT(c10, &tile[(sr + 8u) * 64u + sc], 64u);
  coopStoreT(c11, &tile[(sr + 8u) * 64u + sc + 8u], 64u);
  coopStoreT(c12, &tile[(sr + 8u) * 64u + sc + 16u], 64u);
  coopStoreT(c13, &tile[(sr + 8u) * 64u + sc + 24u], 64u);
  workgroupBarrier();
  for (var i = lane; i < 2048u; i += 128u) {
    let m = m0 + i / 64u;
    if (m < p.m) {
      let n = n0 + i % 64u;
      let v = tile[i] + bias[n];
      let o = m * p.n + n;
      switch p.mode {
        case 1u: { c[o] = v / (1.0 + exp(-v)); }
        case 2u: { c[o] = max(v, 0.0); }
        case 3u: { c[o] += p.alpha * v; }
        default: { c[o] = v; }
      }
    }
  }
}
