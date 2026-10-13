// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Portable multi-head attention, heads of head_dim ≤ 128 (ch = heads · head_dim). The projections
// are rows of `stride` floats with q at 0 and k at ch: [q | k], or [q + u | k | v | q + v] for
// relative positions. scores writes scale · q·k into s [heads][t][t], softmax (in ops.wgsl)
// normalizes each row in place, values computes o [t][ch] = s·v. For relative positions, scores
// adds (q + v)_i·P[j − i + t − 1] before the scale, P being the projected positions [2t − 1][ch]
// from row 16 on (row 16 + r is position t − 1 − r). flash and relative compute the same without
// the scores in memory, for the head sizes `Gpu::portable_flash` names.

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  stride: u32,
  // v [t][..] rows: their stride and the offset of v in them.
  v_stride: u32,
  v_offset: u32,
  relative: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> sc_qk: array<f32>;
@group(0) @binding(1) var<storage, read> sc_pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> sc_s: array<f32>;

// [16][head_dim + 1] tiles of query rows and key rows.
var<workgroup> qs: array<f32, 2064>;
var<workgroup> ks: array<f32, 2064>;

// One workgroup per 16×16 scores of one head.
@compute @workgroup_size(16, 16)
fn scores(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let hd = p.head_dim;
  let h = wg.z;
  let i0 = wg.y * 16u;
  let j0 = wg.x * 16u;
  for (var e = l.y * 16u + l.x; e < 16u * hd; e += 256u) {
    let r = e / hd;
    let c = e % hd;
    var q = 0.0;
    var k = 0.0;
    if (i0 + r < p.t) {
      q = sc_qk[(i0 + r) * p.stride + h * hd + c];
    }
    if (j0 + r < p.t) {
      k = sc_qk[(j0 + r) * p.stride + p.ch + h * hd + c];
    }
    qs[r * (hd + 1u) + c] = q;
    ks[r * (hd + 1u) + c] = k;
  }
  workgroupBarrier();
  var acc = 0.0;
  for (var c = 0u; c < hd; c++) {
    acc += qs[l.y * (hd + 1u) + c] * ks[l.x * (hd + 1u) + c];
  }
  let i = i0 + l.y;
  let j = j0 + l.x;
  if (i < p.t && j < p.t) {
    if (p.relative != 0u) {
      let qv = i * p.stride + 3u * p.ch + h * hd;
      let pr = (j + p.t + 15u - i) * p.ch + h * hd;
      var pos = 0.0;
      for (var c = 0u; c < hd; c++) {
        pos += sc_qk[qv + c] * sc_pos[pr + c];
      }
      acc += pos;
    }
    sc_s[(h * p.t + i) * p.t + j] = acc * p.scale;
  }
}

@group(0) @binding(0) var<storage, read> va_s: array<f32>;
@group(0) @binding(1) var<storage, read> va_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> va_o: array<f32>;

// [16 queries][16 keys] probabilities (padded) and [16 keys][head_dim] values.
var<workgroup> ps: array<f32, 272>;
var<workgroup> vs: array<f32, 2048>;

// One workgroup per 16 queries (wg.x) of one head (wg.y); thread (x, y) owns query y and
// columns x + 16u.
@compute @workgroup_size(16, 16)
fn values(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let hd = p.head_dim;
  let h = wg.y;
  let i = wg.x * 16u + l.y;
  var acc: array<f32, 8>;
  for (var j0 = 0u; j0 < p.t; j0 += 16u) {
    var pv = 0.0;
    if (i < p.t && j0 + l.x < p.t) {
      pv = va_s[(h * p.t + i) * p.t + j0 + l.x];
    }
    ps[l.y * 17u + l.x] = pv;
    // The same trip count for all threads: FXC (DX12) rejects the barrier below after a loop whose
    // length varies.
    for (var e0 = 0u; e0 < 16u * hd; e0 += 256u) {
      let e = e0 + l.y * 16u + l.x;
      if (e < 16u * hd) {
        var x = 0.0;
        if (j0 + e / hd < p.t) {
          x = va_v[(j0 + e / hd) * p.v_stride + p.v_offset + h * hd + e % hd];
        }
        vs[e] = x;
      }
    }
    workgroupBarrier();
    for (var jj = 0u; jj < 16u; jj++) {
      let pj = ps[l.y * 17u + jj];
      for (var u = 0u; u < 8u; u++) {
        let c = l.x + 16u * u;
        if (c < hd) {
          acc[u] += pj * vs[jj * hd + c];
        }
      }
    }
    workgroupBarrier();
  }
  if (i < p.t) {
    for (var u = 0u; u < 8u; u++) {
      let c = l.x + 16u * u;
      if (c < hd) {
        va_o[i * p.ch + h * hd + c] = acc[u];
      }
    }
  }
}

@group(0) @binding(0) var<storage, read> fl_qk: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> fl_v: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> fl_o: array<vec4<f32>>;

// q as [column][query] pairs, k as [column][key] quads, then v as [key][column] quads; zeros past
// head_dim and t.
var<workgroup> fq: array<vec2<f32>, 1024>;
var<workgroup> fk: array<vec4<f32>, 512>;
// Probabilities as [key][query] pairs.
var<workgroup> fp: array<vec2<f32>, 512>;
// Per thread column x, a value per query: [x][query].
var<workgroup> fx: array<f32, 256>;

// Without relative positions, head_dim ≤ 64 and a multiple of 4: attention without the scores in
// memory. A workgroup takes 32 queries of one head and walks the keys in blocks of 32 with an online
// softmax. Thread (x, y) scores queries 2y and 2y + 1 against keys 4x..4x + 3, then sums the
// values of columns 4x..4x + 3 and 32 + 4x..32 + 4x + 3 for those queries.
@compute @workgroup_size(8, 16)
fn flash(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) l: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let h4 = p.head_dim / 4u;
  let q0 = wg.x * 32u;
  let head = wg.y * p.head_dim;
  // Pairs of q rows by column quads, transposed to [column][pair].
  for (var e = lid; e < 256u; e += 128u) {
    let r = 2u * (e % 16u);
    let c = e / 16u;
    var x = vec4(0.0);
    var y = vec4(0.0);
    if (c < h4) {
      if (q0 + r < p.t) {
        x = fl_qk[((q0 + r) * p.stride + head) / 4u + c];
      }
      if (q0 + r + 1u < p.t) {
        y = fl_qk[((q0 + r + 1u) * p.stride + head) / 4u + c];
      }
    }
    fq[4u * c * 16u + r / 2u] = vec2(x.x, y.x);
    fq[(4u * c + 1u) * 16u + r / 2u] = vec2(x.y, y.y);
    fq[(4u * c + 2u) * 16u + r / 2u] = vec2(x.z, y.z);
    fq[(4u * c + 3u) * 16u + r / 2u] = vec2(x.w, y.w);
  }
  let r0 = 2u * l.y;
  let keys = 4u * l.x + vec4(0u, 1u, 2u, 3u);
  var m = vec2(-3.0e38);
  var sum = vec2(0.0);
  var oa0 = vec4(0.0);
  var ob0 = vec4(0.0);
  var oa1 = vec4(0.0);
  var ob1 = vec4(0.0);
  for (var j0 = 0u; j0 < p.t; j0 += 32u) {
    // Thread lid transposes keys 4·(lid % 8).. by column quad lid / 8 into [column][key quad].
    let r = j0 + 4u * (lid % 8u);
    let c = lid / 8u;
    var k: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
      k[i] = vec4(0.0);
      if (r + i < p.t && c < h4) {
        k[i] = fl_qk[((r + i) * p.stride + p.ch + head) / 4u + c];
      }
    }
    fk[4u * c * 8u + lid % 8u] = vec4(k[0].x, k[1].x, k[2].x, k[3].x);
    fk[(4u * c + 1u) * 8u + lid % 8u] = vec4(k[0].y, k[1].y, k[2].y, k[3].y);
    fk[(4u * c + 2u) * 8u + lid % 8u] = vec4(k[0].z, k[1].z, k[2].z, k[3].z);
    fk[(4u * c + 3u) * 8u + lid % 8u] = vec4(k[0].w, k[1].w, k[2].w, k[3].w);
    workgroupBarrier();
    var s0 = vec4(0.0);
    var s1 = vec4(0.0);
    for (var c = 0u; c < p.head_dim; c++) {
      let q = fq[c * 16u + l.y];
      let k = fk[c * 8u + l.x];
      s0 += q.x * k;
      s1 += q.y * k;
    }
    let valid = keys < vec4(p.t - j0);
    let x0 = select(vec4(-3.0e38), s0 * p.scale, valid);
    let x1 = select(vec4(-3.0e38), s1 * p.scale, valid);
    fx[l.x * 32u + r0] = max(max(x0.x, x0.y), max(x0.z, x0.w));
    fx[l.x * 32u + r0 + 1u] = max(max(x1.x, x1.y), max(x1.z, x1.w));
    workgroupBarrier();
    var top = m;
    for (var i = 0u; i < 8u; i++) {
      top = max(top, vec2(fx[i * 32u + r0], fx[i * 32u + r0 + 1u]));
    }
    let alpha = exp(m - top);
    m = top;
    let e0 = exp(x0 - top.x);
    let e1 = exp(x1 - top.y);
    sum = sum * alpha + vec2(dot(e0, vec4(1.0)), dot(e1, vec4(1.0)));
    oa0 *= alpha.x;
    ob0 *= alpha.x;
    oa1 *= alpha.y;
    ob1 *= alpha.y;
    fp[4u * l.x * 16u + l.y] = vec2(e0.x, e1.x);
    fp[(4u * l.x + 1u) * 16u + l.y] = vec2(e0.y, e1.y);
    fp[(4u * l.x + 2u) * 16u + l.y] = vec2(e0.z, e1.z);
    fp[(4u * l.x + 3u) * 16u + l.y] = vec2(e0.w, e1.w);
    for (var e = lid; e < 512u; e += 128u) {
      let r = e / 16u;
      let c = e % 16u;
      var x = vec4(0.0);
      if (j0 + r < p.t && c < h4) {
        x = fl_v[((j0 + r) * p.v_stride + p.v_offset + head) / 4u + c];
      }
      fk[e] = x;
    }
    workgroupBarrier();
    for (var j = 0u; j < 32u; j++) {
      let pj = fp[j * 16u + l.y];
      let va = fk[j * 16u + l.x];
      let vb = fk[j * 16u + 8u + l.x];
      oa0 += pj.x * va;
      ob0 += pj.x * vb;
      oa1 += pj.y * va;
      ob1 += pj.y * vb;
    }
    workgroupBarrier();
  }
  // Each thread summed only its own keys' probabilities.
  fx[l.x * 32u + r0] = sum.x;
  fx[l.x * 32u + r0 + 1u] = sum.y;
  workgroupBarrier();
  var total = vec2(0.0);
  for (var i = 0u; i < 8u; i++) {
    total += vec2(fx[i * 32u + r0], fx[i * 32u + r0 + 1u]);
  }
  let o = ((q0 + r0) * p.ch + head) / 4u;
  let ch4 = p.ch / 4u;
  if (q0 + r0 < p.t) {
    if (l.x < h4) {
      fl_o[o + l.x] = oa0 / total.x;
    }
    if (8u + l.x < h4) {
      fl_o[o + 8u + l.x] = ob0 / total.x;
    }
  }
  if (q0 + r0 + 1u < p.t) {
    if (l.x < h4) {
      fl_o[o + ch4 + l.x] = oa1 / total.y;
    }
    if (8u + l.x < h4) {
      fl_o[o + ch4 + 8u + l.x] = ob1 / total.y;
    }
  }
}

@group(0) @binding(0) var<storage, read> rl_qk: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> rl_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rl_o: array<vec4<f32>>;

// A chunk of 32 columns of (q + u, q + v) as [column][query pair] quads (q + u of both queries,
// then q + v of both); later half a block of v, 64 columns as [key][column quad].
var<workgroup> rq: array<vec4<f32>, 512>;
// The chunk's k as [column][key pair].
var<workgroup> rk: array<vec2<f32>, 512>;
// The chunk's 64 rows of positions as [column][row pair].
var<workgroup> rr: array<vec2<f32>, 1024>;

// Stages columns 64·half.. of v for keys j0.. into rq.
fn stage_v(half: u32, j0: u32, head4: u32, lid: u32) {
  for (var e = lid; e < 512u; e += 128u) {
    let r = e / 16u;
    let c = 16u * half + e % 16u;
    var x = vec4(0.0);
    if (j0 + r < p.t && c < p.head_dim / 4u) {
      x = rl_qk[(j0 + r) * p.stride / 4u + 2u * (p.ch / 4u) + head4 + c];
    }
    rq[e] = x;
  }
  workgroupBarrier();
}

// With relative positions, head_dim ≤ 128 and a multiple of 4: the flash kernel's layout (32
// queries of one head, keys in blocks of 32, thread (x, y) on queries 2y, 2y + 1 and keys 4x..4x +
// 3, columns 4x.. and 32 + 4x.. of each 64-column half), with the scores summed over chunks of 32
// columns. Query i and key j need position row j − i + t + 15, so a block needs 63 rows from j0 −
// i0 + t − 16 on (local row j − j0 − (i − i0) + 31), and a thread rows 4x − 2y + 30 to + 35.
@compute @workgroup_size(8, 16)
fn relative(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) l: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let h4 = p.head_dim / 4u;
  let ch4 = p.ch / 4u;
  let s4 = p.stride / 4u;
  let q0 = wg.x * 32u;
  let head4 = wg.y * h4;
  let r0 = 2u * l.y;
  let pair = (4u * l.x + 30u - r0) / 2u;
  let keys = 4u * l.x + vec4(0u, 1u, 2u, 3u);
  var m = vec2(-3.0e38);
  var sum = vec2(0.0);
  var oa0 = vec4(0.0);
  var ob0 = vec4(0.0);
  var oa1 = vec4(0.0);
  var ob1 = vec4(0.0);
  var oc0 = vec4(0.0);
  var od0 = vec4(0.0);
  var oc1 = vec4(0.0);
  var od1 = vec4(0.0);
  for (var j0 = 0u; j0 < p.t; j0 += 32u) {
    let base = i32(j0 + p.t) - i32(q0) - 16;
    var a0 = vec4(0.0);
    var a1 = vec4(0.0);
    var b0 = vec4(0.0);
    var b1 = vec4(0.0);
    for (var c0 = 0u; c0 < h4; c0 += 8u) {
      {
        // Query pair lid % 16, column quad lid / 16.
        let r = q0 + 2u * (lid % 16u);
        let c = c0 + lid / 16u;
        var u0 = vec4(0.0);
        var u1 = vec4(0.0);
        var v0 = vec4(0.0);
        var v1 = vec4(0.0);
        if (c < h4) {
          if (r < p.t) {
            u0 = rl_qk[r * s4 + head4 + c];
            v0 = rl_qk[r * s4 + 3u * ch4 + head4 + c];
          }
          if (r + 1u < p.t) {
            u1 = rl_qk[(r + 1u) * s4 + head4 + c];
            v1 = rl_qk[(r + 1u) * s4 + 3u * ch4 + head4 + c];
          }
        }
        let o = 4u * (lid / 16u) * 16u + lid % 16u;
        rq[o] = vec4(u0.x, u1.x, v0.x, v1.x);
        rq[o + 16u] = vec4(u0.y, u1.y, v0.y, v1.y);
        rq[o + 32u] = vec4(u0.z, u1.z, v0.z, v1.z);
        rq[o + 48u] = vec4(u0.w, u1.w, v0.w, v1.w);
      }
      {
        // Key pair lid % 16, column quad lid / 16.
        let r = j0 + 2u * (lid % 16u);
        let c = c0 + lid / 16u;
        var k0 = vec4(0.0);
        var k1 = vec4(0.0);
        if (c < h4) {
          if (r < p.t) {
            k0 = rl_qk[r * s4 + ch4 + head4 + c];
          }
          if (r + 1u < p.t) {
            k1 = rl_qk[(r + 1u) * s4 + ch4 + head4 + c];
          }
        }
        let o = 4u * (lid / 16u) * 16u + lid % 16u;
        rk[o] = vec2(k0.x, k1.x);
        rk[o + 16u] = vec2(k0.y, k1.y);
        rk[o + 32u] = vec2(k0.z, k1.z);
        rk[o + 48u] = vec2(k0.w, k1.w);
      }
      for (var e = lid; e < 256u; e += 128u) {
        // Row pair e % 32, column quad e / 32.
        let r = base + 2 * i32(e % 32u);
        let c = c0 + e / 32u;
        var x = vec4(0.0);
        var y = vec4(0.0);
        if (c < h4) {
          if (r >= 0 && u32(r) < 2u * p.t + 15u) {
            x = rl_pos[u32(r) * ch4 + head4 + c];
          }
          if (r + 1 >= 0 && u32(r + 1) < 2u * p.t + 15u) {
            y = rl_pos[u32(r + 1) * ch4 + head4 + c];
          }
        }
        let o = 4u * (e / 32u) * 32u + e % 32u;
        rr[o] = vec2(x.x, y.x);
        rr[o + 32u] = vec2(x.y, y.y);
        rr[o + 64u] = vec2(x.z, y.z);
        rr[o + 96u] = vec2(x.w, y.w);
      }
      workgroupBarrier();
      for (var c = 0u; c < 32u; c++) {
        let q = rq[c * 16u + l.y];
        let k = vec4(rk[c * 16u + 2u * l.x], rk[c * 16u + 2u * l.x + 1u]);
        let pa = rr[c * 32u + pair];
        let pb = rr[c * 32u + pair + 1u];
        let pc = rr[c * 32u + pair + 2u];
        a0 += q.x * k;
        a1 += q.y * k;
        b0 += q.z * vec4(pa.y, pb.x, pb.y, pc.x);
        b1 += q.w * vec4(pa.x, pa.y, pb.x, pb.y);
      }
      workgroupBarrier();
    }
    let valid = keys < vec4(p.t - j0);
    let x0 = select(vec4(-3.0e38), (a0 + b0) * p.scale, valid);
    let x1 = select(vec4(-3.0e38), (a1 + b1) * p.scale, valid);
    fx[l.x * 32u + r0] = max(max(x0.x, x0.y), max(x0.z, x0.w));
    fx[l.x * 32u + r0 + 1u] = max(max(x1.x, x1.y), max(x1.z, x1.w));
    workgroupBarrier();
    var top = m;
    for (var i = 0u; i < 8u; i++) {
      top = max(top, vec2(fx[i * 32u + r0], fx[i * 32u + r0 + 1u]));
    }
    let alpha = exp(m - top);
    m = top;
    let e0 = exp(x0 - top.x);
    let e1 = exp(x1 - top.y);
    sum = sum * alpha + vec2(dot(e0, vec4(1.0)), dot(e1, vec4(1.0)));
    oa0 *= alpha.x;
    ob0 *= alpha.x;
    oc0 *= alpha.x;
    od0 *= alpha.x;
    oa1 *= alpha.y;
    ob1 *= alpha.y;
    oc1 *= alpha.y;
    od1 *= alpha.y;
    fp[4u * l.x * 16u + l.y] = vec2(e0.x, e1.x);
    fp[(4u * l.x + 1u) * 16u + l.y] = vec2(e0.y, e1.y);
    fp[(4u * l.x + 2u) * 16u + l.y] = vec2(e0.z, e1.z);
    fp[(4u * l.x + 3u) * 16u + l.y] = vec2(e0.w, e1.w);
    stage_v(0u, j0, head4, lid);
    for (var j = 0u; j < 32u; j++) {
      let pj = fp[j * 16u + l.y];
      let va = rq[j * 16u + l.x];
      let vb = rq[j * 16u + 8u + l.x];
      oa0 += pj.x * va;
      ob0 += pj.x * vb;
      oa1 += pj.y * va;
      ob1 += pj.y * vb;
    }
    workgroupBarrier();
    if (p.head_dim > 64u) {
      stage_v(1u, j0, head4, lid);
      for (var j = 0u; j < 32u; j++) {
        let pj = fp[j * 16u + l.y];
        let va = rq[j * 16u + l.x];
        let vb = rq[j * 16u + 8u + l.x];
        oc0 += pj.x * va;
        od0 += pj.x * vb;
        oc1 += pj.y * va;
        od1 += pj.y * vb;
      }
      workgroupBarrier();
    }
  }
  fx[l.x * 32u + r0] = sum.x;
  fx[l.x * 32u + r0 + 1u] = sum.y;
  workgroupBarrier();
  var total = vec2(0.0);
  for (var i = 0u; i < 8u; i++) {
    total += vec2(fx[i * 32u + r0], fx[i * 32u + r0 + 1u]);
  }
  let o = (q0 + r0) * ch4 + head4;
  var out = array(oa0 / total.x, ob0 / total.x, oc0 / total.x, od0 / total.x);
  var out1 = array(oa1 / total.y, ob1 / total.y, oc1 / total.y, od1 / total.y);
  for (var u = 0u; u < 4u; u++) {
    let c = 8u * u + l.x;
    if (c < h4) {
      if (q0 + r0 < p.t) {
        rl_o[o + c] = out[u];
      }
      if (q0 + r0 + 1u < p.t) {
        rl_o[o + ch4 + c] = out1[u];
      }
    }
  }
}
