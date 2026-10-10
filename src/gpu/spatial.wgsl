// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// 2D subsampling convs over [t][f][ch] images (time, frequency, channels): 3×3 kernels
// [time tap][frequency tap][ch] with stride 2 and padding 1 on both axes, so output pixel (t, f)
// reads input rows 2t − 1 + i and columns 2f − 1 + j. One workgroup per output pixel, channels
// across its threads. The convs run in chunks of output rows t0.. of the second one: a chunk
// buffer holds the rows 2·t0 − 1.. of the first one's output.

struct Params {
  // The input image.
  t: u32,
  f: u32,
  ch: u32,
  // The chunk's first output row of depthwise.
  t0: u32,
}

var<immediate> p: Params;

fn half(n: u32) -> u32 {
  return (n - 1u) / 2u + 1u;
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
// The first conv: one input channel to ch, then ReLU.
@group(0) @binding(4) var<storage, read> w0: array<f32>;
@group(0) @binding(5) var<storage, read> b0: array<f32>;

// Depthwise conv + bias from a chunk buffer x into the output rows t0 + wg.y of y.
@compute @workgroup_size(256)
fn depthwise(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  for (var c = lid; c < p.ch; c += 256u) {
    var acc = b[c];
    for (var i = 0u; i < 3u; i++) {
      // Input row plus the padding; it is row 2·wg.y + i of the chunk.
      let ti = 2u * (p.t0 + wg.y) + i;
      for (var j = 0u; j < 3u; j++) {
        let fi = 2u * wg.x + j;
        if (ti >= 1u && ti <= p.t && fi >= 1u && fi <= p.f) {
          acc += w[(i * 3u + j) * p.ch + c] * x[((2u * wg.y + i) * p.f + fi - 1u) * p.ch + c];
        }
      }
    }
    y[((p.t0 + wg.y) * half(p.f) + wg.x) * p.ch + c] = acc;
  }
}

// The first conv + ReLU over a one-channel image x [t][f], then the depthwise conv + bias over its
// output, which is computed per tap and never stored, into row wg.y of a chunk buffer y (output
// row 2·t0 − 1 + wg.y; −1 comes out as the bias). The 7×7 input pixels it reads go to shared
// memory first, zero outside the image (zero taps add exact zeros).
var<workgroup> x7: array<f32, 49>;

@compute @workgroup_size(256)
fn first_depthwise(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t1 = half(p.t);
  let f1 = half(p.f);
  // Output pixel plus 1 on both axes; it reads input rows 4·r − 7.. and columns 4·wg.x − 3...
  let r = 2u * p.t0 + wg.y;
  if (lid < 49u) {
    let tm = 4u * r + lid / 7u - 7u;
    let fm = 4u * wg.x + lid % 7u - 3u;
    var v = 0.0;
    if (tm < p.t && fm < p.f) {
      v = x[tm * p.f + fm];
    }
    x7[lid] = v;
  }
  workgroupBarrier();
  for (var c = lid; c < p.ch; c += 256u) {
    let a0 = w0[c];
    let a1 = w0[p.ch + c];
    let a2 = w0[2u * p.ch + c];
    let a3 = w0[3u * p.ch + c];
    let a4 = w0[4u * p.ch + c];
    let a5 = w0[5u * p.ch + c];
    let a6 = w0[6u * p.ch + c];
    let a7 = w0[7u * p.ch + c];
    let a8 = w0[8u * p.ch + c];
    var acc = b[c];
    for (var i = 0u; i < 3u; i++) {
      // First conv's row plus the padding; wraps for row −1, so the check fails.
      let ti = 2u * r + i - 2u;
      for (var j = 0u; j < 3u; j++) {
        let fi = 2u * wg.x + j;
        if (ti >= 1u && ti <= t1 && fi >= 1u && fi <= f1) {
          let o = 14u * i + 2u * j;
          var v = b0[c];
          v += a0 * x7[o];
          v += a1 * x7[o + 1u];
          v += a2 * x7[o + 2u];
          v += a3 * x7[o + 7u];
          v += a4 * x7[o + 8u];
          v += a5 * x7[o + 9u];
          v += a6 * x7[o + 14u];
          v += a7 * x7[o + 15u];
          v += a8 * x7[o + 16u];
          acc += w[(i * 3u + j) * p.ch + c] * max(v, 0.0);
        }
      }
    }
    y[(wg.y * half(f1) + wg.x) * p.ch + c] = acc;
  }
}

@group(0) @binding(0) var<storage, read> fl_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> fl_y: array<f32>;

// [t][f][ch] to [t][ch·f]: y[t][c·f + j] = x[t][j][c]. One workgroup per row t.
@compute @workgroup_size(256)
fn flatten(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let n = p.f * p.ch;
  for (var e = lid; e < n; e += 256u) {
    fl_y[wg.x * n + e] = fl_x[wg.x * n + (e % p.f) * p.ch + e / p.f];
  }
}
