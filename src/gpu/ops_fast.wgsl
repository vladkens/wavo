// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Fast versions of ops.wgsl kernels, with the same bindings and Params: one 32-lane subgroup per
// row. Workgroups have 128 threads (four subgroups). A dispatch with t == 0 is a probe: it writes
// the subgroup size and count to the first two floats of the output.

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  // conv_glu: the depthwise kernel size, and 1 for LayerNorm or 0 for the affine x·g + b.
  kernel: u32,
  layer_norm: u32,
}

var<immediate> p: Params;

// LayerNorm, four rows per workgroup. The rotary output recomputes the normalized partner element
// instead of exchanging it.

@group(0) @binding(0) var<storage, read> norm_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_g: array<f32>;
@group(0) @binding(2) var<storage, read> norm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_y: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_yr: array<f32>;

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
  let st = vec2(mean, inverseSqrt(subgroupAdd(q) / f32(p.ch) + 1e-5));
  if (r >= p.t) {
    return;
  }
  for (var i = sl; i < p.ch; i += 32u) {
    norm_y[t * p.ch + i] = normalized(t, i, st);
  }
  if (p.head_dim == 0u) {
    return;
  }
  let half = p.head_dim / 2u;
  for (var i = sl; i < p.ch; i += 32u) {
    let d = i % p.head_dim;
    let f = d % half;
    let sn = rope[t * p.head_dim + half + f];
    var vr = normalized(t, i, st) * rope[t * p.head_dim + f];
    if (d < half) {
      vr -= normalized(t, i + half, st) * sn;
    } else {
      vr += normalized(t, i - half, st) * sn;
    }
    norm_yr[t * p.ch + i] = vr;
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
  let pad = p.kernel / 2u;
  var s = 0.0;
  for (var i = sl; i < p.ch; i += 32u) {
    var acc = conv_b[i];
    for (var k = 0u; k < p.kernel; k++) {
      let src = t + k;
      if (src >= pad && src - pad < p.t) {
        let base = (src - pad) * 2u * p.ch;
        acc += conv_w[i * p.kernel + k] * conv_h[base + i] / (1.0 + exp(-conv_h[base + p.ch + i]));
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
      var y = rows[row + i] * conv_g[i] + conv_beta[i];
      if (p.layer_norm != 0u) {
        y = (rows[row + i] - mean) * rstd * conv_g[i] + conv_beta[i];
      }
      conv_y[t * p.ch + i] = y / (1.0 + exp(-y));
    }
  }
}
