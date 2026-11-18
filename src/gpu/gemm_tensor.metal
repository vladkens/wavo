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
  uint offset;
  // Q4 and Q5 conv rows shorter than `k`. 0 means the stored row is already `k` wide.
  uint src_k;
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

// Four Q5 codes at index x (a multiple of 4). Same packing as gemm.wgsl's q5_word.
static float4 q5_codes4(device const uchar *bytes, int x) {
  const int block = x >> 5;
  const int group = (x & 31) >> 2;
  const int bit = group * 20;
  const int byte = block * 24 + (bit >> 3);
  const uint shift = uint(byte & 3) << 3;
  device const uint *words = (device const uint *)bytes;
  const int i = byte >> 2;
  uint word = words[i];
  if (shift != 0) {
    word = (word >> shift) | (words[i + 1] << (32 - shift));
  }
  word >>= (bit & 7);
  return float4(float(word & 31u), float((word >> 5) & 31u), float((word >> 10) & 31u),
                float((word >> 15) & 31u));
}

static float4 q5_4(device const uchar *bytes, device const half *scales, int x) {
  return (q5_codes4(bytes, x) - 16.0f) * float(scales[x >> 5]);
}

// Four Q4 codes at index x (a multiple of 4). Nibbles are in weight order, 16 bytes per 32.
static float4 q4_codes4(device const uchar *bytes, int x) {
  const int block = x >> 5;
  const int local = x & 31;
  device const uint *words = (device const uint *)bytes;
  const uint word = words[block * 4 + local / 8];
  const uint bits = (word >> ((local % 8) * 4)) & 0xFFFFu;
  return float4(float(bits & 15u), float((bits >> 4) & 15u), float((bits >> 8) & 15u),
                float((bits >> 12) & 15u));
}

static float4 q4_4(device const uchar *bytes, device const half *scales, int x) {
  return (q4_codes4(bytes, x) - 8.0f) * float(scales[x >> 5]);
}

// Q4_1 and Q5_1 store an f16 scale and an f16 minimum per block.
static float4 q41_4(device const uchar *bytes, device const half *scales, int x) {
  const int block = x >> 5;
  return q4_codes4(bytes, x) * float(scales[block * 2]) + float(scales[block * 2 + 1]);
}

static float4 q51_4(device const uchar *bytes, device const half *scales, int x) {
  const int block = x >> 5;
  return q5_codes4(bytes, x) * float(scales[block * 2]) + float(scales[block * 2 + 1]);
}

static float4 q4k_4(device const uchar *bytes, device const half *scales, int x) {
  const int block = x >> 8;
  const int local = x & 255;
  device const uint *words = (device const uint *)bytes;
  const uint word = words[block * 32 + local / 8];
  const uint bits = (word >> ((local % 8) * 4)) & 0xFFFFu;
  const float4 codes = float4(float(bits & 15u), float((bits >> 4) & 15u), float((bits >> 8) & 15u),
                              float((bits >> 12) & 15u));
  const int sub = local >> 5;
  device const uint *sc = (device const uint *)(scales + block * 10);
  const float d = float(scales[block * 10]);
  const float dmin = float(scales[block * 10 + 1]);
  const uint scale = (sc[1 + sub / 4] >> ((sub % 4) * 8)) & 255u;
  const uint min = (sc[3 + sub / 4] >> ((sub % 4) * 8)) & 255u;
  return codes * (d * float(scale)) - dmin * float(min);
}

// WTYPE 0: F32, 1: F16, 2: Q8_0 (int8 and an f16 scale per 32), 3: Q5_0, 4: Q4_0 (4-bit codes,
// 16 bytes per 32, and the same scale), 5: Q4_1, 6: Q5_1 (scale and minimum per block), 7: Q4_K.
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
      const int col = k0 + kc + 4 * j;
      if (p.src_k != 0 && WTYPE >= 3) {
        float4 v = 0;
        const int base = (n0 + r) * (int)p.src_k + col;
        for (int i = 0; i < 4; i++) {
          if (col + i < (int)p.src_k) {
            const int e = base + i;
            const int a = e & ~3;
            float4 q;
            if (WTYPE == 3) q = q5_4((device const uchar *)w, scales, a);
            else if (WTYPE == 4) q = q4_4((device const uchar *)w, scales, a);
            else if (WTYPE == 5) q = q41_4((device const uchar *)w, scales, a);
            else q = q51_4((device const uchar *)w, scales, a);
            v[i] = q[e & 3];
          }
        }
        dst[j] = half4(v);
      } else if (WTYPE == 0) {
        dst[j] = half4(((device const float4 *)w)[x / 4 + j]);
      } else if (WTYPE == 1) {
        dst[j] = ((device const half4 *)w)[x / 4 + j];
      } else if (WTYPE == 2) {
        dst[j] = half4(float4(((device const char4 *)(w + x))[j]) * float(scales[x / 32]));
      } else if (WTYPE == 3) {
        dst[j] = half4(q5_4((device const uchar *)w, scales, x + 4 * j));
      } else if (WTYPE == 4) {
        dst[j] = half4(q4_4((device const uchar *)w, scales, x + 4 * j));
      } else if (WTYPE == 5) {
        dst[j] = half4(q41_4((device const uchar *)w, scales, x + 4 * j));
      } else if (WTYPE == 6) {
        dst[j] = half4(q51_4((device const uchar *)w, scales, x + 4 * j));
      } else {
        dst[j] = half4(q4k_4((device const uchar *)w, scales, x + 4 * j));
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
template [[host_name("gemm_q5")]] kernel void gemm<3>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
template [[host_name("gemm_q4")]] kernel void gemm<4>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
template [[host_name("gemm_q4_1")]] kernel void gemm<5>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
template [[host_name("gemm_q5_1")]] kernel void gemm<6>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
template [[host_name("gemm_q4_k")]] kernel void gemm<7>(constant Params &, device half *,
  device const float *, device float *, device const char *, device const half *, uint3, ushort);
