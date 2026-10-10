// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// attention.wgsl's attention without relative positions (same Params and layouts) as a flash kernel
// on 8×8 F32 cooperative matrices, heads of head_dim 32..64 (a multiple of 8). Each 32-lane
// subgroup owns 16 queries of one head and walks the keys in blocks of 32 on its own, loading q, k
// and v fragments straight from memory; its output stays in registers. The softmax runs against a
// reference maximum per row that moves (rescaling the output) only when a score passes it by more
// than 8, so a probability stays below e⁸. Rows of qk and v past t up to the next multiple of 32 are
// read (and masked), so must be finite: the arenas keep them so. A dispatch with t == 0 is a probe:
// it writes the subgroup size and count to the first two floats of o.

enable wgpu_cooperative_matrix;

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  stride: u32,
  v_stride: u32,
  v_offset: u32,
}

var<immediate> p: Params;

alias AM = coop_mat8x8<f32, A>;
alias BM = coop_mat8x8<f32, B>;
alias CM = coop_mat8x8<f32, C>;

@group(0) @binding(0) var<storage, read> qk: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;

// Per subgroup the scores, then probabilities, [16 queries][32 keys].
var<workgroup> sc: array<f32, 2048>;
// Per subgroup an 8×8 fragment, then per query the output's factor and the probabilities' sum.
var<workgroup> xs: array<f32, 384>;

// Rows r..r + 8 of a subgroup's output fragment `f`, each times its factor.
fn rescale(f: CM, x: u32, sl: u32, r: u32) -> CM {
  coopStoreT(f, &xs[x], 8u);
  subgroupBarrier();
  for (var e = sl; e < 64u; e += 32u) {
    xs[x + e] *= xs[x + 64u + r + e / 8u];
  }
  subgroupBarrier();
  return coopLoadT<CM>(&xs[x], 8u);
}

// Writes rows r..r + 8 of a subgroup's output fragment `f` over the sums to o[at..], rows below t.
fn finish(f: CM, x: u32, sl: u32, r: u32, at: u32, q0: u32) {
  coopStoreT(f, &xs[x], 8u);
  subgroupBarrier();
  for (var e = sl; e < 64u; e += 32u) {
    if (q0 + r + e / 8u < p.t) {
      o[at + (r + e / 8u) * p.ch + e % 8u] = xs[x + e] / xs[x + 80u + r + e / 8u];
    }
  }
  subgroupBarrier();
}

// `f` + the probabilities of 32 keys times their v rows from v[at..].
fn pv(f: CM, p0: AM, p1: AM, p2: AM, p3: AM, at: u32) -> CM {
  let vs = p.v_stride;
  let f1 = coopMultiplyAdd(p0, coopLoadT<BM>(&v[at], vs), f);
  let f2 = coopMultiplyAdd(p1, coopLoadT<BM>(&v[at + 8u * vs], vs), f1);
  let f3 = coopMultiplyAdd(p2, coopLoadT<BM>(&v[at + 16u * vs], vs), f2);
  return coopMultiplyAdd(p3, coopLoadT<BM>(&v[at + 24u * vs], vs), f3);
}

// Scores at sc[b + c] for the columns c of `i`, those from `lim` on masked.
fn load4(b: u32, i: vec4<u32>, lim: u32) -> vec4<f32> {
  let s = vec4(sc[b + i.x], sc[b + i.y], sc[b + i.z], sc[b + i.w]);
  return select(vec4(-3.0e38), s, i < vec4(lim));
}

fn store4(b: u32, i: vec4<u32>, e: vec4<f32>) {
  sc[b + i.x] = e.x;
  sc[b + i.y] = e.y;
  sc[b + i.z] = e.z;
  sc[b + i.w] = e.w;
}

@compute @workgroup_size(128)
fn flash(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (sg == 0u && sl == 0u) {
      o[0] = f32(width);
      o[1] = f32(count);
    }
    return;
  }
  let hd = p.head_dim;
  let qs = p.stride;
  let q0 = (wg.x * 4u + sg) * 16u;
  let qa = q0 * qs + wg.y * hd;
  let sb = sg * 512u;
  let x = sg * 96u;
  // A lane takes half h of row r, its keys in the order i = (n + r) % 16 for n = 0..16: no two
  // lanes on one bank.
  let r = sl % 16u;
  let h = sl / 16u * 16u;
  let row = sb + r * 32u + h;
  let i0 = (vec4(0u, 1u, 2u, 3u) + r) % 16u;
  let i1 = (i0 + 4u) % 16u;
  let i2 = (i0 + 8u) % 16u;
  let i3 = (i0 + 12u) % 16u;
  var o00: CM;
  var o01: CM;
  var o02: CM;
  var o03: CM;
  var o04: CM;
  var o05: CM;
  var o06: CM;
  var o07: CM;
  var o10: CM;
  var o11: CM;
  var o12: CM;
  var o13: CM;
  var o14: CM;
  var o15: CM;
  var o16: CM;
  var o17: CM;
  var m = -3.0e38;
  var l = 0.0;
  for (var j0 = 0u; j0 < p.t; j0 += 32u) {
    // Sᵀ = K·Qᵀ: k rows load as they are, only q transposed.
    var s00: CM;
    var s01: CM;
    var s10: CM;
    var s11: CM;
    var s20: CM;
    var s21: CM;
    var s30: CM;
    var s31: CM;
    let kb = j0 * qs + p.ch + wg.y * hd;
    for (var d = 0u; d < hd; d += 8u) {
      let b0 = coopLoad<BM>(&qk[qa + d], qs);
      let b1 = coopLoad<BM>(&qk[qa + 8u * qs + d], qs);
      let k0 = coopLoadT<AM>(&qk[kb + d], qs);
      let k1 = coopLoadT<AM>(&qk[kb + 8u * qs + d], qs);
      let k2 = coopLoadT<AM>(&qk[kb + 16u * qs + d], qs);
      let k3 = coopLoadT<AM>(&qk[kb + 24u * qs + d], qs);
      s00 = coopMultiplyAdd(k0, b0, s00);
      s01 = coopMultiplyAdd(k0, b1, s01);
      s10 = coopMultiplyAdd(k1, b0, s10);
      s11 = coopMultiplyAdd(k1, b1, s11);
      s20 = coopMultiplyAdd(k2, b0, s20);
      s21 = coopMultiplyAdd(k2, b1, s21);
      s30 = coopMultiplyAdd(k3, b0, s30);
      s31 = coopMultiplyAdd(k3, b1, s31);
    }
    // Column-major stores of Sᵀ give S [query][key].
    coopStore(s00, &sc[sb], 32u);
    coopStore(s01, &sc[sb + 256u], 32u);
    coopStore(s10, &sc[sb + 8u], 32u);
    coopStore(s11, &sc[sb + 264u], 32u);
    coopStore(s20, &sc[sb + 16u], 32u);
    coopStore(s21, &sc[sb + 272u], 32u);
    coopStore(s30, &sc[sb + 24u], 32u);
    coopStore(s31, &sc[sb + 280u], 32u);
    subgroupBarrier();
    let lim = u32(max(i32(p.t) - i32(j0 + h), 0));
    let x0 = load4(row, i0, lim);
    let x1 = load4(row, i1, lim);
    let x2 = load4(row, i2, lim);
    let x3 = load4(row, i3, lim);
    let mx = max(max(x0, x1), max(x2, x3));
    let top = max(max(mx.x, mx.y), max(mx.z, mx.w));
    let bm = max(top, subgroupShuffleXor(top, 16u)) * p.scale;
    var alpha = 1.0;
    if (bm > m + 8.0) {
      alpha = exp(m - bm);
      m = bm;
    }
    let e0 = exp(x0 * p.scale - m);
    let e1 = exp(x1 * p.scale - m);
    let e2 = exp(x2 * p.scale - m);
    let e3 = exp(x3 * p.scale - m);
    store4(row, i0, e0);
    store4(row, i1, e1);
    store4(row, i2, e2);
    store4(row, i3, e3);
    let es = e0 + e1 + e2 + e3;
    let sum = es.x + es.y + es.z + es.w;
    l = l * alpha + subgroupShuffleXor(sum, 16u) + sum;
    if (sl < 16u) {
      xs[x + 64u + sl] = alpha;
      xs[x + 80u + sl] = l;
    }
    subgroupBarrier();
    if (subgroupAny(alpha != 1.0)) {
      o00 = rescale(o00, x, sl, 0u);
      o01 = rescale(o01, x, sl, 0u);
      o02 = rescale(o02, x, sl, 0u);
      o03 = rescale(o03, x, sl, 0u);
      o04 = rescale(o04, x, sl, 0u);
      o05 = rescale(o05, x, sl, 0u);
      o06 = rescale(o06, x, sl, 0u);
      o07 = rescale(o07, x, sl, 0u);
      o10 = rescale(o10, x, sl, 8u);
      o11 = rescale(o11, x, sl, 8u);
      o12 = rescale(o12, x, sl, 8u);
      o13 = rescale(o13, x, sl, 8u);
      o14 = rescale(o14, x, sl, 8u);
      o15 = rescale(o15, x, sl, 8u);
      o16 = rescale(o16, x, sl, 8u);
      o17 = rescale(o17, x, sl, 8u);
    }
    let pa0 = coopLoadT<AM>(&sc[sb], 32u);
    let pa1 = coopLoadT<AM>(&sc[sb + 8u], 32u);
    let pa2 = coopLoadT<AM>(&sc[sb + 16u], 32u);
    let pa3 = coopLoadT<AM>(&sc[sb + 24u], 32u);
    let pb0 = coopLoadT<AM>(&sc[sb + 256u], 32u);
    let pb1 = coopLoadT<AM>(&sc[sb + 264u], 32u);
    let pb2 = coopLoadT<AM>(&sc[sb + 272u], 32u);
    let pb3 = coopLoadT<AM>(&sc[sb + 280u], 32u);
    let vb = j0 * p.v_stride + p.v_offset + wg.y * hd;
    o00 = pv(o00, pa0, pa1, pa2, pa3, vb);
    o10 = pv(o10, pb0, pb1, pb2, pb3, vb);
    o01 = pv(o01, pa0, pa1, pa2, pa3, vb + 8u);
    o11 = pv(o11, pb0, pb1, pb2, pb3, vb + 8u);
    o02 = pv(o02, pa0, pa1, pa2, pa3, vb + 16u);
    o12 = pv(o12, pb0, pb1, pb2, pb3, vb + 16u);
    o03 = pv(o03, pa0, pa1, pa2, pa3, vb + 24u);
    o13 = pv(o13, pb0, pb1, pb2, pb3, vb + 24u);
    if (hd > 32u) {
      o04 = pv(o04, pa0, pa1, pa2, pa3, vb + 32u);
      o14 = pv(o14, pb0, pb1, pb2, pb3, vb + 32u);
    }
    if (hd > 40u) {
      o05 = pv(o05, pa0, pa1, pa2, pa3, vb + 40u);
      o15 = pv(o15, pb0, pb1, pb2, pb3, vb + 40u);
    }
    if (hd > 48u) {
      o06 = pv(o06, pa0, pa1, pa2, pa3, vb + 48u);
      o16 = pv(o16, pb0, pb1, pb2, pb3, vb + 48u);
    }
    if (hd > 56u) {
      o07 = pv(o07, pa0, pa1, pa2, pa3, vb + 56u);
      o17 = pv(o17, pb0, pb1, pb2, pb3, vb + 56u);
    }
    subgroupBarrier();
  }
  let ob = q0 * p.ch + wg.y * hd;
  finish(o00, x, sl, 0u, ob, q0);
  finish(o01, x, sl, 0u, ob + 8u, q0);
  finish(o02, x, sl, 0u, ob + 16u, q0);
  finish(o03, x, sl, 0u, ob + 24u, q0);
  finish(o10, x, sl, 8u, ob, q0);
  finish(o11, x, sl, 8u, ob + 8u, q0);
  finish(o12, x, sl, 8u, ob + 16u, q0);
  finish(o13, x, sl, 8u, ob + 24u, q0);
  if (hd > 32u) {
    finish(o04, x, sl, 0u, ob + 32u, q0);
    finish(o14, x, sl, 8u, ob + 32u, q0);
  }
  if (hd > 40u) {
    finish(o05, x, sl, 0u, ob + 40u, q0);
    finish(o15, x, sl, 8u, ob + 40u, q0);
  }
  if (hd > 48u) {
    finish(o06, x, sl, 0u, ob + 48u, q0);
    finish(o16, x, sl, 8u, ob + 48u, q0);
  }
  if (hd > 56u) {
    finish(o07, x, sl, 0u, ob + 56u, q0);
    finish(o17, x, sl, 8u, ob + 56u, q0);
  }
}
