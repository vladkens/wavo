// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// attention.wgsl's flash attention (same Params and layouts, no relative positions) on 16×16
// cooperative matrices with f16 operands and F32 accumulators, heads of head_dim ≤ 64, a multiple of
// 4, as ggml's flash_attn_cm1.comp does on Vulkan. A workgroup of four 32-lane subgroups takes 64
// queries of one head. Per block of 64 keys it stages k and v as f16 in shared memory; each
// subgroup scores its 16 queries (q fragments in registers), runs their softmax and adds the
// probabilities (rounded to f16) times v onto its output fragments, also in registers, rescaled
// through shared memory when a row's maximum grows. (Letting the maximum lag by up to 8, as
// flash.wgsl does, kept the speed but moved a GigaAM e2e-ctc token by a frame.) A dispatch with
// t == 0 is a probe: it writes the subgroup size and count to the first two floats of o.

enable f16;
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

alias AM = coop_mat16x16<f16, A>;
alias BM = coop_mat16x16<f16, B>;
alias CM = coop_mat16x16<f32, C>;

// Row strides of the f16 tiles and of the f32 scores, padded against bank conflicts.
const KS: u32 = 72u;
const SS: u32 = 68u;

@group(0) @binding(0) var<storage, read> qk: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;

// k (first the workgroup's q) and v of a block as f16 [64 keys][KS], zeros past head_dim and t.
var<workgroup> kt: array<f16, 4608>;
var<workgroup> vt: array<f16, 4608>;
// Per subgroup the scores [16][SS], then an output fragment while it is rescaled or written.
var<workgroup> sc: array<f32, 4352>;
// Per subgroup the probabilities [16][KS].
var<workgroup> pt: array<f16, 4608>;
// Per subgroup and row the output's factor, then the probabilities' sum.
var<workgroup> rows: array<f32, 128>;

fn put_k(i: u32, x: vec4<f32>) {
  kt[i] = f16(x.x);
  kt[i + 1u] = f16(x.y);
  kt[i + 2u] = f16(x.z);
  kt[i + 3u] = f16(x.w);
}

fn put_v(i: u32, x: vec4<f32>) {
  vt[i] = f16(x.x);
  vt[i + 1u] = f16(x.y);
  vt[i + 2u] = f16(x.z);
  vt[i + 3u] = f16(x.w);
}

// `f` + the probabilities of 64 keys times columns c..c + 16 of v.
fn pv(f: CM, p0: AM, p1: AM, p2: AM, p3: AM, c: u32) -> CM {
  let f1 = coopMultiplyAdd(p0, coopLoadT<BM>(&vt[c], KS), f);
  let f2 = coopMultiplyAdd(p1, coopLoadT<BM>(&vt[16u * KS + c], KS), f1);
  let f3 = coopMultiplyAdd(p2, coopLoadT<BM>(&vt[32u * KS + c], KS), f2);
  return coopMultiplyAdd(p3, coopLoadT<BM>(&vt[48u * KS + c], KS), f3);
}

// `f` with each row times its factor in `rows`, through the subgroup's scores at `at`.
fn rescale(f: CM, at: u32, x: u32, sl: u32) -> CM {
  coopStoreT(f, &sc[at], 16u);
  subgroupBarrier();
  for (var e = sl; e < 256u; e += 32u) {
    sc[at + e] *= rows[x + e / 16u];
  }
  subgroupBarrier();
  return coopLoadT<CM>(&sc[at], 16u);
}

// Writes columns c..c + 16 of the output fragment `f` over the sums, rows below t.
fn finish(f: CM, at: u32, x: u32, sl: u32, q0: u32, head: u32, c: u32) {
  coopStoreT(f, &sc[at], 16u);
  subgroupBarrier();
  for (var e = sl; e < 256u; e += 32u) {
    let r = e / 16u;
    if (q0 + r < p.t && c + e % 16u < p.head_dim) {
      o[(q0 + r) * p.ch + head + c + e % 16u] = sc[at + e] / rows[x + 16u + r];
    }
  }
  subgroupBarrier();
}

@compute @workgroup_size(128)
fn flash(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (lid == 0u) {
      o[0] = f32(width);
      o[1] = f32(count);
    }
    return;
  }
  let hd = p.head_dim;
  let h4 = hd / 4u;
  let head = wg.y * hd;
  // Element e of a 64-row tile: row e / 16, column quad e % 16.
  for (var e = lid; e < 1024u; e += 128u) {
    let r = wg.x * 64u + e / 16u;
    var x = vec4(0.0);
    if (r < p.t && e % 16u < h4) {
      x = qk[(r * p.stride + head) / 4u + e % 16u];
    }
    put_k(e / 16u * KS + e % 16u * 4u, x);
  }
  workgroupBarrier();
  let qa = sg * 16u * KS;
  let qf0 = coopLoadT<AM>(&kt[qa], KS);
  let qf1 = coopLoadT<AM>(&kt[qa + 16u], KS);
  let qf2 = coopLoadT<AM>(&kt[qa + 32u], KS);
  let qf3 = coopLoadT<AM>(&kt[qa + 48u], KS);
  workgroupBarrier();
  let q0 = wg.x * 64u + sg * 16u;
  let sb = sg * 16u * SS;
  let pb = sg * 16u * KS;
  let x = sg * 32u;
  // A lane takes half h of row r's keys.
  let r = sl % 16u;
  let h = sl / 16u * 32u;
  var o0: CM;
  var o1: CM;
  var o2: CM;
  var o3: CM;
  // The row's maximum and sum, the same in both lanes of the row.
  var m = -3.0e38;
  var l = 0.0;
  for (var j0 = 0u; j0 < p.t; j0 += 64u) {
    for (var e = lid; e < 1024u; e += 128u) {
      let j = j0 + e / 16u;
      var kx = vec4(0.0);
      var vx = vec4(0.0);
      if (j < p.t && e % 16u < h4) {
        kx = qk[(j * p.stride + p.ch + head) / 4u + e % 16u];
        vx = v[(j * p.v_stride + p.v_offset + head) / 4u + e % 16u];
      }
      put_k(e / 16u * KS + e % 16u * 4u, kx);
      put_v(e / 16u * KS + e % 16u * 4u, vx);
    }
    workgroupBarrier();
    // Column-major loads of k rows give kᵀ.
    var s0: CM;
    var s1: CM;
    var s2: CM;
    var s3: CM;
    s0 = coopMultiplyAdd(qf0, coopLoad<BM>(&kt[0], KS), s0);
    s1 = coopMultiplyAdd(qf0, coopLoad<BM>(&kt[16u * KS], KS), s1);
    s2 = coopMultiplyAdd(qf0, coopLoad<BM>(&kt[32u * KS], KS), s2);
    s3 = coopMultiplyAdd(qf0, coopLoad<BM>(&kt[48u * KS], KS), s3);
    if (hd > 16u) {
      s0 = coopMultiplyAdd(qf1, coopLoad<BM>(&kt[16u], KS), s0);
      s1 = coopMultiplyAdd(qf1, coopLoad<BM>(&kt[16u * KS + 16u], KS), s1);
      s2 = coopMultiplyAdd(qf1, coopLoad<BM>(&kt[32u * KS + 16u], KS), s2);
      s3 = coopMultiplyAdd(qf1, coopLoad<BM>(&kt[48u * KS + 16u], KS), s3);
    }
    if (hd > 32u) {
      s0 = coopMultiplyAdd(qf2, coopLoad<BM>(&kt[32u], KS), s0);
      s1 = coopMultiplyAdd(qf2, coopLoad<BM>(&kt[16u * KS + 32u], KS), s1);
      s2 = coopMultiplyAdd(qf2, coopLoad<BM>(&kt[32u * KS + 32u], KS), s2);
      s3 = coopMultiplyAdd(qf2, coopLoad<BM>(&kt[48u * KS + 32u], KS), s3);
    }
    if (hd > 48u) {
      s0 = coopMultiplyAdd(qf3, coopLoad<BM>(&kt[48u], KS), s0);
      s1 = coopMultiplyAdd(qf3, coopLoad<BM>(&kt[16u * KS + 48u], KS), s1);
      s2 = coopMultiplyAdd(qf3, coopLoad<BM>(&kt[32u * KS + 48u], KS), s2);
      s3 = coopMultiplyAdd(qf3, coopLoad<BM>(&kt[48u * KS + 48u], KS), s3);
    }
    coopStoreT(s0, &sc[sb], SS);
    coopStoreT(s1, &sc[sb + 16u], SS);
    coopStoreT(s2, &sc[sb + 32u], SS);
    coopStoreT(s3, &sc[sb + 48u], SS);
    subgroupBarrier();
    let row = sb + r * SS + h;
    let lim = i32(p.t) - i32(j0 + h);
    var top = -3.0e38;
    for (var i = 0u; i < 32u; i++) {
      if (i32(i) < lim) {
        top = max(top, sc[row + i]);
      }
    }
    top = max(top, subgroupShuffleXor(top, 16u)) * p.scale;
    let alpha = exp(m - max(m, top));
    m = max(m, top);
    var sum = 0.0;
    for (var i = 0u; i < 32u; i++) {
      var e = f16(0.0);
      if (i32(i) < lim) {
        e = f16(exp(sc[row + i] * p.scale - m));
      }
      pt[pb + r * KS + h + i] = e;
      sum += f32(e);
    }
    l = l * alpha + sum + subgroupShuffleXor(sum, 16u);
    if (sl < 16u) {
      rows[x + sl] = alpha;
    }
    subgroupBarrier();
    if (subgroupAny(alpha != 1.0)) {
      o0 = rescale(o0, sb, x, sl);
      if (hd > 16u) {
        o1 = rescale(o1, sb, x, sl);
      }
      if (hd > 32u) {
        o2 = rescale(o2, sb, x, sl);
      }
      if (hd > 48u) {
        o3 = rescale(o3, sb, x, sl);
      }
    }
    let p0 = coopLoadT<AM>(&pt[pb], KS);
    let p1 = coopLoadT<AM>(&pt[pb + 16u], KS);
    let p2 = coopLoadT<AM>(&pt[pb + 32u], KS);
    let p3 = coopLoadT<AM>(&pt[pb + 48u], KS);
    o0 = pv(o0, p0, p1, p2, p3, 0u);
    if (hd > 16u) {
      o1 = pv(o1, p0, p1, p2, p3, 16u);
    }
    if (hd > 32u) {
      o2 = pv(o2, p0, p1, p2, p3, 32u);
    }
    if (hd > 48u) {
      o3 = pv(o3, p0, p1, p2, p3, 48u);
    }
    workgroupBarrier();
  }
  if (sl < 16u) {
    rows[x + 16u + sl] = l;
  }
  subgroupBarrier();
  finish(o0, sb, x, sl, q0, head, 0u);
  if (hd > 16u) {
    finish(o1, sb, x, sl, q0, head, 16u);
  }
  if (hd > 32u) {
    finish(o2, sb, x, sl, q0, head, 32u);
  }
  if (hd > 48u) {
    finish(o3, sb, x, sl, q0, head, 48u);
  }
}
