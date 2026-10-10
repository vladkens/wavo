// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// attention.wgsl's attention with relative positions (same Params and layouts) as a flash kernel
// on 8×8 F32 cooperative matrices, heads of head_dim ≤ 128, a multiple of 8. A workgroup is one
// 32-lane subgroup that owns 16 queries of one head and walks all keys in blocks of 64 with an online
// softmax, loading q, k and v fragments straight from memory; only its scores and output
// accumulator live in shared memory. Per block it first scores its queries (q + v) against the 80
// positions the block needs (key c of row r needs position c + 15 − r of them) and stores those
// scores skewed by one per row so each lands on its key; the key scores then accumulate onto them.
// Rows of qk past t up to the next multiple of 64, and of pos from 16 before to 80 after the 2t − 1
// used ones, are read (and masked), so must be finite: the arenas keep them so. A dispatch with
// t == 0 is a probe: it writes the subgroup size and count to the first two floats of o.

enable wgpu_cooperative_matrix;

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  stride: u32,
  v_stride: u32,
  v_offset: u32,
  relative: u32,
}

var<immediate> p: Params;

// The scores' row stride and lead.
const STRIDE: u32 = 80u;
const LEAD: u32 = 16u;

alias AM = coop_mat8x8<f32, A>;
alias BM = coop_mat8x8<f32, B>;
alias CM = coop_mat8x8<f32, C>;

@group(0) @binding(0) var<storage, read> att_qk: array<f32>;
@group(0) @binding(1) var<storage, read> att_v: array<f32>;
@group(0) @binding(2) var<storage, read> att_pos: array<f32>;
@group(0) @binding(3) var<storage, read_write> att_o: array<f32>;

// The scores, then probabilities, of row r and key c at LEAD + STRIDE·r + c; columns 64..79 and
// the 16 lead floats take the positions that fall outside the block.
var<workgroup> sc: array<f32, 1296>;
// The [16][head_dim] output accumulator.
var<workgroup> acc: array<f32, 2048>;

@compute @workgroup_size(32)
fn attention(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (sl == 0u) {
      att_o[0] = f32(width);
      att_o[1] = f32(count);
    }
    return;
  }
  let hd = p.head_dim;
  let qs = p.stride;
  let q0 = wg.x * 16u;
  let qa = q0 * qs + wg.y * hd;
  let ka = p.ch + wg.y * hd;
  let va = p.v_offset + wg.y * hd;
  let sb = LEAD;
  for (var i = sl; i < 16u * hd; i += 32u) {
    acc[i] = 0.0;
  }
  // Running max and sum of each row, the same in every lane.
  var m: array<f32, 16>;
  var l: array<f32, 16>;
  for (var r = 0u; r < 16u; r++) {
    m[r] = -3.0e38;
  }
  for (var j0 = 0u; j0 < p.t; j0 += 64u) {
    subgroupBarrier();
    let qv = qa + 3u * p.ch;
    // Position f of this block is logical row j0 − q0 + t − 16 + f.
    let pb = (j0 + p.t - q0) * p.ch + wg.y * hd;
    for (var f0 = 0u; f0 < 80u; f0 += 40u) {
      var s00: CM;
      var s01: CM;
      var s02: CM;
      var s03: CM;
      var s04: CM;
      var s10: CM;
      var s11: CM;
      var s12: CM;
      var s13: CM;
      var s14: CM;
      for (var d = 0u; d < hd; d += 8u) {
        let a0 = coopLoadT<AM>(&att_qk[qv + d], qs);
        let a1 = coopLoadT<AM>(&att_qk[qv + 8u * qs + d], qs);
        let b = pb + f0 * p.ch + d;
        let b0 = coopLoad<BM>(&att_pos[b], p.ch);
        let b1 = coopLoad<BM>(&att_pos[b + 8u * p.ch], p.ch);
        let b2 = coopLoad<BM>(&att_pos[b + 16u * p.ch], p.ch);
        let b3 = coopLoad<BM>(&att_pos[b + 24u * p.ch], p.ch);
        let b4 = coopLoad<BM>(&att_pos[b + 32u * p.ch], p.ch);
        s00 = coopMultiplyAdd(a0, b0, s00);
        s01 = coopMultiplyAdd(a0, b1, s01);
        s02 = coopMultiplyAdd(a0, b2, s02);
        s03 = coopMultiplyAdd(a0, b3, s03);
        s04 = coopMultiplyAdd(a0, b4, s04);
        s10 = coopMultiplyAdd(a1, b0, s10);
        s11 = coopMultiplyAdd(a1, b1, s11);
        s12 = coopMultiplyAdd(a1, b2, s12);
        s13 = coopMultiplyAdd(a1, b3, s13);
        s14 = coopMultiplyAdd(a1, b4, s14);
      }
      // Position f of row r goes to key f − 15 + r: a row stride of 81 from 15 floats back.
      let o = sb - 15u + f0;
      coopStoreT(s00, &sc[o], 81u);
      coopStoreT(s01, &sc[o + 8u], 81u);
      coopStoreT(s02, &sc[o + 16u], 81u);
      coopStoreT(s03, &sc[o + 24u], 81u);
      coopStoreT(s04, &sc[o + 32u], 81u);
      coopStoreT(s10, &sc[o + 648u], 81u);
      coopStoreT(s11, &sc[o + 656u], 81u);
      coopStoreT(s12, &sc[o + 664u], 81u);
      coopStoreT(s13, &sc[o + 672u], 81u);
      coopStoreT(s14, &sc[o + 680u], 81u);
    }
    subgroupBarrier();
    // Scores for two halves of 32 keys.
    for (var half = 0u; half < 64u; half += 32u) {
      let kb = (j0 + half) * qs + ka;
      let o = sb + half;
      var s00: CM;
      var s01: CM;
      var s02: CM;
      var s03: CM;
      var s10: CM;
      var s11: CM;
      var s12: CM;
      var s13: CM;
      s00 = coopLoadT<CM>(&sc[o], 80u);
      s01 = coopLoadT<CM>(&sc[o + 8u], 80u);
      s02 = coopLoadT<CM>(&sc[o + 16u], 80u);
      s03 = coopLoadT<CM>(&sc[o + 24u], 80u);
      s10 = coopLoadT<CM>(&sc[o + 640u], 80u);
      s11 = coopLoadT<CM>(&sc[o + 648u], 80u);
      s12 = coopLoadT<CM>(&sc[o + 656u], 80u);
      s13 = coopLoadT<CM>(&sc[o + 664u], 80u);
      for (var d = 0u; d < hd; d += 8u) {
        let a0 = coopLoadT<AM>(&att_qk[qa + d], qs);
        let a1 = coopLoadT<AM>(&att_qk[qa + 8u * qs + d], qs);
        // Column-major loads of k rows give kᵀ.
        let b0 = coopLoad<BM>(&att_qk[kb + d], qs);
        let b1 = coopLoad<BM>(&att_qk[kb + 8u * qs + d], qs);
        let b2 = coopLoad<BM>(&att_qk[kb + 16u * qs + d], qs);
        let b3 = coopLoad<BM>(&att_qk[kb + 24u * qs + d], qs);
        s00 = coopMultiplyAdd(a0, b0, s00);
        s01 = coopMultiplyAdd(a0, b1, s01);
        s02 = coopMultiplyAdd(a0, b2, s02);
        s03 = coopMultiplyAdd(a0, b3, s03);
        s10 = coopMultiplyAdd(a1, b0, s10);
        s11 = coopMultiplyAdd(a1, b1, s11);
        s12 = coopMultiplyAdd(a1, b2, s12);
        s13 = coopMultiplyAdd(a1, b3, s13);
      }
      let o8 = o + 8u * STRIDE;
      coopStoreT(s00, &sc[o], STRIDE);
      coopStoreT(s01, &sc[o + 8u], STRIDE);
      coopStoreT(s02, &sc[o + 16u], STRIDE);
      coopStoreT(s03, &sc[o + 24u], STRIDE);
      coopStoreT(s10, &sc[o8], STRIDE);
      coopStoreT(s11, &sc[o8 + 8u], STRIDE);
      coopStoreT(s12, &sc[o8 + 16u], STRIDE);
      coopStoreT(s13, &sc[o8 + 24u], STRIDE);
    }
    subgroupBarrier();
    // Online softmax, one row at a time with lanes on keys sl and sl + 32, then rescale that row.
    let valid0 = j0 + sl < p.t;
    let valid1 = j0 + 32u + sl < p.t;
    for (var r = 0u; r < 16u; r++) {
      let i = sb + r * STRIDE + sl;
      var s0 = -3.0e38;
      var s1 = -3.0e38;
      if (valid0) {
        s0 = sc[i] * p.scale;
      }
      if (valid1) {
        s1 = sc[i + 32u] * p.scale;
      }
      let m_new = max(m[r], subgroupMax(max(s0, s1)));
      var e0 = 0.0;
      var e1 = 0.0;
      if (valid0) {
        e0 = exp(s0 - m_new);
      }
      if (valid1) {
        e1 = exp(s1 - m_new);
      }
      sc[i] = e0;
      sc[i + 32u] = e1;
      // The same in all lanes. Once the row max settles, the output needs no rescale.
      if (m_new != m[r]) {
        let alpha = exp(m[r] - m_new);
        l[r] *= alpha;
        for (var c = sl; c < hd; c += 32u) {
          acc[r * hd + c] *= alpha;
        }
      }
      l[r] += subgroupAdd(e0 + e1);
      m[r] = m_new;
    }
    subgroupBarrier();
    let s8 = sb + 8u * STRIDE;
    let p00 = coopLoadT<AM>(&sc[sb], STRIDE);
    let p01 = coopLoadT<AM>(&sc[sb + 8u], STRIDE);
    let p02 = coopLoadT<AM>(&sc[sb + 16u], STRIDE);
    let p03 = coopLoadT<AM>(&sc[sb + 24u], STRIDE);
    let p04 = coopLoadT<AM>(&sc[sb + 32u], STRIDE);
    let p05 = coopLoadT<AM>(&sc[sb + 40u], STRIDE);
    let p06 = coopLoadT<AM>(&sc[sb + 48u], STRIDE);
    let p07 = coopLoadT<AM>(&sc[sb + 56u], STRIDE);
    let p10 = coopLoadT<AM>(&sc[s8], STRIDE);
    let p11 = coopLoadT<AM>(&sc[s8 + 8u], STRIDE);
    let p12 = coopLoadT<AM>(&sc[s8 + 16u], STRIDE);
    let p13 = coopLoadT<AM>(&sc[s8 + 24u], STRIDE);
    let p14 = coopLoadT<AM>(&sc[s8 + 32u], STRIDE);
    let p15 = coopLoadT<AM>(&sc[s8 + 40u], STRIDE);
    let p16 = coopLoadT<AM>(&sc[s8 + 48u], STRIDE);
    let p17 = coopLoadT<AM>(&sc[s8 + 56u], STRIDE);
    let vs = p.v_stride;
    let vb = j0 * vs + va;
    for (var c = 0u; c < hd; c += 8u) {
      var o0 = coopLoadT<CM>(&acc[c], hd);
      var o1 = coopLoadT<CM>(&acc[8u * hd + c], hd);
      let v0 = coopLoadT<BM>(&att_v[vb + c], vs);
      let v1 = coopLoadT<BM>(&att_v[vb + 8u * vs + c], vs);
      let v2 = coopLoadT<BM>(&att_v[vb + 16u * vs + c], vs);
      let v3 = coopLoadT<BM>(&att_v[vb + 24u * vs + c], vs);
      let v4 = coopLoadT<BM>(&att_v[vb + 32u * vs + c], vs);
      let v5 = coopLoadT<BM>(&att_v[vb + 40u * vs + c], vs);
      let v6 = coopLoadT<BM>(&att_v[vb + 48u * vs + c], vs);
      let v7 = coopLoadT<BM>(&att_v[vb + 56u * vs + c], vs);
      o0 = coopMultiplyAdd(p00, v0, o0);
      o0 = coopMultiplyAdd(p01, v1, o0);
      o0 = coopMultiplyAdd(p02, v2, o0);
      o0 = coopMultiplyAdd(p03, v3, o0);
      o0 = coopMultiplyAdd(p04, v4, o0);
      o0 = coopMultiplyAdd(p05, v5, o0);
      o0 = coopMultiplyAdd(p06, v6, o0);
      o0 = coopMultiplyAdd(p07, v7, o0);
      o1 = coopMultiplyAdd(p10, v0, o1);
      o1 = coopMultiplyAdd(p11, v1, o1);
      o1 = coopMultiplyAdd(p12, v2, o1);
      o1 = coopMultiplyAdd(p13, v3, o1);
      o1 = coopMultiplyAdd(p14, v4, o1);
      o1 = coopMultiplyAdd(p15, v5, o1);
      o1 = coopMultiplyAdd(p16, v6, o1);
      o1 = coopMultiplyAdd(p17, v7, o1);
      coopStoreT(o0, &acc[c], hd);
      coopStoreT(o1, &acc[8u * hd + c], hd);
    }
  }
  subgroupBarrier();
  for (var r = 0u; r < 16u; r++) {
    if (q0 + r < p.t) {
      for (var c = sl; c < hd; c += 32u) {
        att_o[(q0 + r) * p.ch + wg.y * hd + c] = acc[r * hd + c] / l[r];
      }
    }
  }
}
