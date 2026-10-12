// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// gemm_fast.wgsl's GEMM on the Metal 4 tensor API (MetalPerformancePrimitives' matmul2d), which
// runs on the neural accelerators of Apple M5 and newer GPUs, after the tensor `kernel_mul_mm` in
// ggml's ggml-metal/kernels/mul_mm.metal. Operands round as in the 8×8 kernel: a as f16, which
// `cast` writes to device memory, W dequantized to half in threadgroup memory, f32 accumulators.
// 128 threads compute 128 rows × 64 columns of c. Buffers sit in wgpu-hal's Metal slots for a
// passthrough module: the immediates at buffer(0), then group 0's bindings in order.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
  uint m;
  uint n;
  uint k;
  uint mode;
  float alpha;
  uint dual;
};

constant constexpr int ROWS = 128;
constant constexpr int COLS = 64;
// The K step: one Q8_0 block.
constant constexpr int KS = 32;

// `n4` vectors of 4 floats of a to f16.
kernel void cast(constant uint &n4 [[buffer(0)]], device const float4 *a [[buffer(1)]],
                 device half4 *h [[buffer(2)]], uint i [[thread_position_in_grid]],
                 uint grid [[threads_per_grid]]) {
  for (; i < n4; i += grid) {
    h[i] = half4(a[i]);
  }
}

// GELU as gemm.wgsl's.
static float gelu(float x) {
  float z = abs(0.70710678118654752440f * x);
  float t = 1.0f / (1.0f + 0.3275911f * z);
  float e = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-z * z);
  return 0.5f * x * (1.0f + sign(x) * e);
}

// WTYPE 0: F32, 1: F16, 2: Q8_0 (int8 values and an f16 scale per 32).
template <int WTYPE>
kernel void gemm(constant Params &p [[buffer(0)]], device half *a [[buffer(1)]],
                 device const float *bias [[buffer(2)]], device float *c [[buffer(3)]],
                 device const char *w [[buffer(4)]], device const half *scales [[buffer(5)]],
                 uint3 tg [[threadgroup_position_in_grid]],
                 ushort lane [[thread_index_in_threadgroup]]) {
  threadgroup half sw[COLS * KS];
  const int k = p.k, m0 = tg.y * ROWS, n0 = tg.x * COLS;
  matmul2d<matmul2d_descriptor(ROWS, COLS, static_cast<int>(dynamic_extent), false, true, false,
                               matmul2d_descriptor::mode::multiply_accumulate),
           execution_simdgroups<4>>
      mm;
  // A K step of a's rows m0.., and the staged W.
  const int rows = p.m - m0;
  auto ta = tensor(a + m0 * k, dextents<int32_t, 2>(KS, rows), array<int, 2>({1, k}));
  auto tw = tensor(sw, dextents<int32_t, 2>(KS, COLS), array<int, 2>({1, KS}));
  auto ct = mm.get_destination_cooperative_tensor<decltype(ta), decltype(tw), float>();
  for (uint16_t i = 0; i < ct.get_capacity(); i++) {
    if (ct.is_valid_element(i)) {
      ct[i] = 0.0f;
    }
  }
  // Each thread stages 16 values of a W row.
  const int r = lane / 2, kc = lane % 2 * 16;
  threadgroup half4 *dst = (threadgroup half4 *)(sw + r * KS + kc);
  for (int k0 = 0; k0 < k; k0 += KS) {
    const int x = (n0 + r) * k + k0 + kc;
    for (short j = 0; j < 4; j++) {
      if (WTYPE == 0) {
        dst[j] = half4(((device const float4 *)w)[x / 4 + j]);
      } else if (WTYPE == 1) {
        dst[j] = ((device const half4 *)w)[x / 4 + j];
      } else {
        dst[j] = half4(float4(((device const char4 *)(w + x))[j]) * float(scales[x / 32]));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ta = tensor(a + m0 * k + k0, dextents<int32_t, 2>(KS, rows), array<int, 2>({1, k}));
    mm.run(ta, tw, ct);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // The epilogue in registers; c's rows are n + dual wide.
  for (uint16_t i = 0; i < ct.get_capacity(); i++) {
    auto at = ct.get_multidimensional_index(i);
    const uint n = n0 + at[0], m = m0 + at[1];
    if (!ct.is_valid_element(i) || m >= p.m || n >= p.n) {
      continue;
    }
    const float v = ct[i] + bias[n];
    device float &o = c[m * (p.n + p.dual) + n];
    switch (p.mode) {
      case 1: o = v / (1.0f + exp(-v)); break;
      case 2: o = max(v, 0.0f); break;
      case 3: o += p.alpha * v; break;
      case 4: o = gelu(v); break;
      case 5: o += gelu(v); break;
      case 6: o = tanh(clamp(v, -10.0f, 10.0f)); break;
      default: o = v;
    }
    if (n < p.dual) {
      (&o)[p.n] = ct[i] + bias[p.n + n];
    }
  }
}

template [[host_name("gemm_f32")]] kernel void gemm<0>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
template [[host_name("gemm_f16")]] kernel void gemm<1>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
template [[host_name("gemm_q8")]] kernel void gemm<2>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
