// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Fast versions of ops.wgsl kernels, with the same bindings and Params: one 32-lane subgroup per
// row for the row-wise kernels, 8×8 F32 cooperative matrices for attention. Workgroups have 128
// threads (four subgroups). A dispatch with t == 0 is a probe: it writes the subgroup size and
// count to the first two floats of the output.

enable wgpu_cooperative_matrix;

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
}

var<immediate> p: Params;

alias AM = coop_mat8x8<f32, A>;
alias BM = coop_mat8x8<f32, B>;
alias CM = coop_mat8x8<f32, C>;

// LayerNorm, four rows per workgroup. layer_norm_rope recomputes the normalized partner element
// instead of exchanging it.

@group(0) @binding(0) var<storage, read> norm_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_g: array<f32>;
@group(0) @binding(2) var<storage, read> norm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_y: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_yr: array<f32>;

// Mean and 1 / std of row t.
fn row_stats(t: u32, sl: u32) -> vec2<f32> {
  var s = 0.0;
  for (var i = sl; i < p.ch; i += 32u) {
    s += norm_x[t * p.ch + i];
  }
  let mean = subgroupAdd(s) / f32(p.ch);
  var q = 0.0;
  for (var i = sl; i < p.ch; i += 32u) {
    let d = norm_x[t * p.ch + i] - mean;
    q += d * d;
  }
  return vec2(mean, inverseSqrt(subgroupAdd(q) / f32(p.ch) + 1e-5));
}

fn normalized(t: u32, i: u32, st: vec2<f32>) -> f32 {
  return (norm_x[t * p.ch + i] - st.x) * st.y * norm_g[i] + norm_b[i];
}

@compute @workgroup_size(128)
fn layer_norm(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (lane == 0u) {
      norm_y[0] = f32(width);
      norm_y[1] = f32(count);
    }
    return;
  }
  // Rows past the end compute on the last row and skip the stores.
  let r = wg.x * 4u + sg;
  let t = min(r, p.t - 1u);
  let st = row_stats(t, sl);
  if (r < p.t) {
    for (var i = sl; i < p.ch; i += 32u) {
      norm_y[t * p.ch + i] = normalized(t, i, st);
    }
  }
}

@compute @workgroup_size(128)
fn layer_norm_rope(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (lane == 0u) {
      norm_y[0] = f32(width);
      norm_y[1] = f32(count);
    }
    return;
  }
  let r = wg.x * 4u + sg;
  let t = min(r, p.t - 1u);
  let st = row_stats(t, sl);
  if (r < p.t) {
    let half = p.head_dim / 2u;
    for (var i = sl; i < p.ch; i += 32u) {
      let v = normalized(t, i, st);
      norm_y[t * p.ch + i] = v;
      let d = i % p.head_dim;
      let f = d % half;
      let sn = rope[t * p.head_dim + half + f];
      var vr = v * rope[t * p.head_dim + f];
      if (d < half) {
        vr -= normalized(t, i + half, st) * sn;
      } else {
        vr += normalized(t, i - half, st) * sn;
      }
      norm_yr[t * p.ch + i] = vr;
    }
  }
}

// Conformer conv module middle (see ops.wgsl), four frames per workgroup.

@group(0) @binding(0) var<storage, read> conv_h: array<f32>;
@group(0) @binding(1) var<storage, read> conv_w: array<f32>;
@group(0) @binding(2) var<storage, read> conv_b: array<f32>;
@group(0) @binding(3) var<storage, read> conv_g: array<f32>;
@group(0) @binding(4) var<storage, read> conv_beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_y: array<f32>;

// Each lane keeps its own channels of its subgroup's row.
var<workgroup> rows: array<f32, 4096>;

@compute @workgroup_size(128)
fn conv_glu(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (lane == 0u) {
      conv_y[0] = f32(width);
      conv_y[1] = f32(count);
    }
    return;
  }
  let r = wg.x * 4u + sg;
  let t = min(r, p.t - 1u);
  let row = sg * 1024u;
  var s = 0.0;
  for (var i = sl; i < p.ch; i += 32u) {
    var acc = conv_b[i];
    for (var k = 0u; k < 5u; k++) {
      let src = t + k;
      if (src >= 2u && src - 2u < p.t) {
        let base = (src - 2u) * 2u * p.ch;
        acc += conv_w[i * 5u + k] * conv_h[base + i] / (1.0 + exp(-conv_h[base + p.ch + i]));
      }
    }
    rows[row + i] = acc;
    s += acc;
  }
  let mean = subgroupAdd(s) / f32(p.ch);
  var q = 0.0;
  for (var i = sl; i < p.ch; i += 32u) {
    let d = rows[row + i] - mean;
    q += d * d;
  }
  let rstd = inverseSqrt(subgroupAdd(q) / f32(p.ch) + 1e-5);
  if (r < p.t) {
    for (var i = sl; i < p.ch; i += 32u) {
      let y = (rows[row + i] - mean) * rstd * conv_g[i] + conv_beta[i];
      conv_y[t * p.ch + i] = y / (1.0 + exp(-y));
    }
  }
}

// Flash attention with online softmax: qk is [t][q | k] (2·ch wide), v and o are [t][ch], heads of
// head_dim ≤ 64 and a multiple of 8. Each subgroup owns 16 queries of one head and walks all keys
// in blocks of 64 on its own, loading q, k and v fragments straight from memory; only its scores
// and output accumulator live in shared memory. Rows of qk and v past t up to the next multiple of
// 64 are read (and masked) so must hold finite values: the arena's spare rows do.

@group(0) @binding(0) var<storage, read> att_qk: array<f32>;
@group(0) @binding(1) var<storage, read> att_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> att_o: array<f32>;

// Per subgroup [16][64] scores, then probabilities.
var<workgroup> sp: array<f32, 4096>;
// Per subgroup [16][head_dim] output accumulator.
var<workgroup> acc: array<f32, 4096>;

@compute @workgroup_size(128)
fn attention(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (lane == 0u) {
      att_o[0] = f32(width);
      att_o[1] = f32(count);
    }
    return;
  }
  let hd = p.head_dim;
  let qs = 2u * p.ch;
  let q0 = wg.x * 64u + sg * 16u;
  let qa = q0 * qs + wg.y * hd;
  let ka = p.ch + wg.y * hd;
  let va = wg.y * hd;
  let sb = sg * 1024u;
  let ab = sg * 1024u;
  for (var i = sl; i < 16u * hd; i += 32u) {
    acc[ab + i] = 0.0;
  }
  // Running max and sum of each row, the same in every lane.
  var m: array<f32, 16>;
  var l: array<f32, 16>;
  for (var r = 0u; r < 16u; r++) {
    m[r] = -3.0e38;
  }
  for (var j0 = 0u; j0 < p.t; j0 += 64u) {
    subgroupBarrier();
    // Scores for two halves of 32 keys each.
    for (var half = 0u; half < 64u; half += 32u) {
      let kb = (j0 + half) * qs + ka;
      var s00: CM;
      var s01: CM;
      var s02: CM;
      var s03: CM;
      var s10: CM;
      var s11: CM;
      var s12: CM;
      var s13: CM;
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
      let o = sb + half;
      coopStoreT(s00, &sp[o], 64u);
      coopStoreT(s01, &sp[o + 8u], 64u);
      coopStoreT(s02, &sp[o + 16u], 64u);
      coopStoreT(s03, &sp[o + 24u], 64u);
      coopStoreT(s10, &sp[o + 512u], 64u);
      coopStoreT(s11, &sp[o + 520u], 64u);
      coopStoreT(s12, &sp[o + 528u], 64u);
      coopStoreT(s13, &sp[o + 536u], 64u);
    }
    subgroupBarrier();
    // Online softmax, one row at a time with lanes on keys sl and sl + 32, then rescale that row.
    let valid0 = j0 + sl < p.t;
    let valid1 = j0 + 32u + sl < p.t;
    for (var r = 0u; r < 16u; r++) {
      let i = sb + r * 64u + sl;
      var s0 = -3.0e38;
      var s1 = -3.0e38;
      if (valid0) {
        s0 = sp[i] * p.scale;
      }
      if (valid1) {
        s1 = sp[i + 32u] * p.scale;
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
      sp[i] = e0;
      sp[i + 32u] = e1;
      // The same in all lanes. Once the row max settles, the output needs no rescale.
      if (m_new != m[r]) {
        let alpha = exp(m[r] - m_new);
        l[r] *= alpha;
        for (var c = sl; c < hd; c += 32u) {
          acc[ab + r * hd + c] *= alpha;
        }
      }
      l[r] += subgroupAdd(e0 + e1);
      m[r] = m_new;
    }
    subgroupBarrier();
    let p00 = coopLoadT<AM>(&sp[sb], 64u);
    let p01 = coopLoadT<AM>(&sp[sb + 8u], 64u);
    let p02 = coopLoadT<AM>(&sp[sb + 16u], 64u);
    let p03 = coopLoadT<AM>(&sp[sb + 24u], 64u);
    let p04 = coopLoadT<AM>(&sp[sb + 32u], 64u);
    let p05 = coopLoadT<AM>(&sp[sb + 40u], 64u);
    let p06 = coopLoadT<AM>(&sp[sb + 48u], 64u);
    let p07 = coopLoadT<AM>(&sp[sb + 56u], 64u);
    let p10 = coopLoadT<AM>(&sp[sb + 512u], 64u);
    let p11 = coopLoadT<AM>(&sp[sb + 520u], 64u);
    let p12 = coopLoadT<AM>(&sp[sb + 528u], 64u);
    let p13 = coopLoadT<AM>(&sp[sb + 536u], 64u);
    let p14 = coopLoadT<AM>(&sp[sb + 544u], 64u);
    let p15 = coopLoadT<AM>(&sp[sb + 552u], 64u);
    let p16 = coopLoadT<AM>(&sp[sb + 560u], 64u);
    let p17 = coopLoadT<AM>(&sp[sb + 568u], 64u);
    let vb = j0 * p.ch + va;
    for (var c = 0u; c < hd; c += 8u) {
      var o0 = coopLoadT<CM>(&acc[ab + c], hd);
      var o1 = coopLoadT<CM>(&acc[ab + 8u * hd + c], hd);
      let v0 = coopLoadT<BM>(&att_v[vb + c], p.ch);
      let v1 = coopLoadT<BM>(&att_v[vb + 8u * p.ch + c], p.ch);
      let v2 = coopLoadT<BM>(&att_v[vb + 16u * p.ch + c], p.ch);
      let v3 = coopLoadT<BM>(&att_v[vb + 24u * p.ch + c], p.ch);
      let v4 = coopLoadT<BM>(&att_v[vb + 32u * p.ch + c], p.ch);
      let v5 = coopLoadT<BM>(&att_v[vb + 40u * p.ch + c], p.ch);
      let v6 = coopLoadT<BM>(&att_v[vb + 48u * p.ch + c], p.ch);
      let v7 = coopLoadT<BM>(&att_v[vb + 56u * p.ch + c], p.ch);
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
      coopStoreT(o0, &acc[ab + c], hd);
      coopStoreT(o1, &acc[ab + 8u * hd + c], hd);
    }
  }
  subgroupBarrier();
  for (var r = 0u; r < 16u; r++) {
    if (q0 + r < p.t) {
      for (var c = sl; c < hd; c += 32u) {
        att_o[(q0 + r) * p.ch + va + c] = acc[ab + r * hd + c] / l[r];
      }
    }
  }
}
