// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// 2D subsampling convs over [t][f][ch] images (time, frequency, channels): 3×3 kernels
// [ch][time tap][frequency tap] with stride 2 and padding 1 on both axes, so output pixel (t, f)
// reads input rows 2t − 1 + i and columns 2f − 1 + j. One workgroup per output pixel, channels
// across its threads.

struct Params {
  // The input image.
  t: u32,
  f: u32,
  ch: u32,
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

// Depthwise conv + bias.
@compute @workgroup_size(256)
fn depthwise(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  for (var c = lid; c < p.ch; c += 256u) {
    var acc = b[c];
    for (var i = 0u; i < 3u; i++) {
      // Input row plus the padding.
      let ti = 2u * wg.y + i;
      for (var j = 0u; j < 3u; j++) {
        let fi = 2u * wg.x + j;
        if (ti >= 1u && ti <= p.t && fi >= 1u && fi <= p.f) {
          acc += w[c * 9u + i * 3u + j] * x[((ti - 1u) * p.f + fi - 1u) * p.ch + c];
        }
      }
    }
    y[(wg.y * half(p.f) + wg.x) * p.ch + c] = acc;
  }
}

// The first conv + ReLU over a one-channel image x [t][f], then the depthwise conv + bias over its
// output, which is computed per tap and never stored.
@compute @workgroup_size(256)
fn first_depthwise(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t1 = half(p.t);
  let f1 = half(p.f);
  for (var c = lid; c < p.ch; c += 256u) {
    var acc = b[c];
    for (var i = 0u; i < 3u; i++) {
      let ti = 2u * wg.y + i;
      for (var j = 0u; j < 3u; j++) {
        let fi = 2u * wg.x + j;
        if (ti >= 1u && ti <= t1 && fi >= 1u && fi <= f1) {
          var v = b0[c];
          for (var k = 0u; k < 3u; k++) {
            let tm = 2u * (ti - 1u) + k;
            for (var l = 0u; l < 3u; l++) {
              let fm = 2u * (fi - 1u) + l;
              if (tm >= 1u && tm <= p.t && fm >= 1u && fm <= p.f) {
                v += w0[c * 9u + k * 3u + l] * x[(tm - 1u) * p.f + fm - 1u];
              }
            }
          }
          acc += w[c * 9u + i * 3u + j] * max(v, 0.0);
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
