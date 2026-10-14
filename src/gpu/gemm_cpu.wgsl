// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Appended to gemm.wgsl on CPU adapters: its GEMM before the 128-wide tiles, which crashed WARP on
// an ARM64 VM. A workgroup computes a 64×64 tile of c, each thread 4×4 outputs strided by 16,
// summed over k in the same order.

// [64][32] tiles of a and W; the padded row stride avoids shared-memory bank conflicts.
const S: u32 = 33u;
var<workgroup> at64: array<f32, 2112>;
var<workgroup> wt64: array<f32, 2112>;

fn store16(m: u32, n0: u32, acc: vec4<f32>) {
  if (m < p.m) {
    epilogue(m, n0, acc.x);
    epilogue(m, n0 + 16u, acc.y);
    epilogue(m, n0 + 32u, acc.z);
    epilogue(m, n0 + 48u, acc.w);
  }
}

@compute @workgroup_size(256)
fn gemm64(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
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
      at64[r * S + e % 32u] = v;
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
      wt64[o] = to_half(v.x);
      wt64[o + 1u] = to_half(v.y);
      wt64[o + 2u] = to_half(v.z);
      wt64[o + 3u] = to_half(v.w);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let b = vec4(wt64[tx * S + kk], wt64[(tx + 16u) * S + kk], wt64[(tx + 32u) * S + kk], wt64[(tx + 48u) * S + kk]);
      acc0 += at64[ty * S + kk] * b;
      acc1 += at64[(ty + 16u) * S + kk] * b;
      acc2 += at64[(ty + 32u) * S + kk] * b;
      acc3 += at64[(ty + 48u) * S + kk] * b;
    }
    workgroupBarrier();
  }
  store16(m0 + ty, n0 + tx, acc0);
  store16(m0 + ty + 16u, n0 + tx, acc1);
  store16(m0 + ty + 32u, n0 + tx, acc2);
  store16(m0 + ty + 48u, n0 + tx, acc3);
}
