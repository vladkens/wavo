// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Subgroup versions of the decoder kernels' reductions: `Gpu::decoder` puts each in place of the
// function of the same name in gemm.wgsl, decode.wgsl and ops.wgsl on GPUs with the fast tier,
// whose subgroups are 32 consecutive invocations.

fn group_sums(s: vec4<f32>, lid: u32) -> vec4<f32> {
  return subgroupAdd(s);
}

fn reduce(v: f32, lid: u32, is_max: bool) -> f32 {
  let r = select(subgroupAdd(v), subgroupMax(v), is_max);
  if (lid % 32u == 0u) {
    red[lid / 32u] = r;
  }
  workgroupBarrier();
  var t = red[0];
  for (var i = 1u; i < 8u; i++) {
    if (is_max) {
      t = max(t, red[i]);
    } else {
      t += red[i];
    }
  }
  workgroupBarrier();
  return t;
}
