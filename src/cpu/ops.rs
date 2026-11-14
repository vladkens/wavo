// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoders' row-wise kernels on the CPU, as `src/gpu/ops.wgsl` and `attention.wgsl` compute
//! them, over time-major `[t][ch]` f32 activations, in plain arithmetic the compiler vectorizes.
//! Rows go to threads in chunks.

use super::gemm::{NR, panels, product};
use super::par_each;
use crate::gpu::pos_rows;

/// e^x as Cephes' `expf` computes it (relative error ~1e-7) without branches or calls, so loops
/// of it vectorize: x clamped to the normal range, n = round(x·log₂e) by adding 1.5·2²³, the
/// remainder by a polynomial, 2ⁿ from the exponent bits.
#[inline]
pub fn exp(x: f32) -> f32 {
  const MAGIC: f32 = 12582912.0;
  let x = x.clamp(-87.0, 88.0);
  let t = x * std::f32::consts::LOG2_E + MAGIC;
  let n = t - MAGIC;
  let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
  let p = 1.987_569_1e-4 * r + 1.398_199_9e-3;
  let p = ((((p * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 0.166_666_65) * r + 0.5) * r * r;
  let scale = f32::from_bits(((t.to_bits() as i32 - 0x4B40_0000 + 127) << 23) as u32);
  (p + r + 1.0) * scale
}

#[inline]
pub fn silu(x: f32) -> f32 {
  x / (1.0 + exp(-x))
}

/// Rows per job for `rows` rows: about four jobs per thread.
fn chunk(rows: usize) -> usize {
  rows.div_ceil(4 * super::threads()).max(1)
}

/// Mean and 1/√(variance + 1e-5) of a row.
fn moments(x: &[f32]) -> (f32, f32) {
  let n = x.len() as f32;
  let mean = sum(x.iter().copied()) / n;
  let var = sum(x.iter().map(|v| (v - mean) * (v - mean))) / n;
  (mean, 1.0 / (var + 1e-5).sqrt())
}

/// A sum in 8 lanes, which vectorizes.
fn sum(v: impl Iterator<Item = f32>) -> f32 {
  let mut acc = [0.0; 8];
  for (i, v) in v.enumerate() {
    acc[i % 8] += v;
  }

  acc.iter().sum()
}

/// LayerNorm (eps 1e-5) of the rows of `x` into `y`, both `[rows][w.len()]`; with `rope` (a table
/// `[t][cos | sin]` of head_dim / 2 frequencies each, `yr`, head_dim) also the same with rotary
/// embedding (NEOX split-half rotation inside each head) into `yr`.
pub fn layer_norm(
  x: &[f32],
  (w, b): (&[f32], &[f32]),
  y: &mut [f32],
  rope: Option<(&[f32], &mut [f32], usize)>,
) {
  let cols = w.len();
  let rows = chunk(x.len() / cols);
  let norm = |x: &[f32], y: &mut [f32]| {
    for (x, y) in x.chunks_exact(cols).zip(y.chunks_exact_mut(cols)) {
      let (mean, rstd) = moments(x);
      for (((y, x), w), b) in y.iter_mut().zip(x).zip(w).zip(b) {
        *y = (x - mean) * rstd * w + b;
      }
    }
  };
  let Some((table, yr, hd)) = rope else {
    let jobs = x.chunks(rows * cols).zip(y.chunks_mut(rows * cols));
    return par_each(jobs, || (), |_, (x, y)| norm(x, y));
  };
  let half = hd / 2;
  let jobs = x.chunks(rows * cols).zip(y.chunks_mut(rows * cols)).zip(yr.chunks_mut(rows * cols));
  let jobs = jobs.zip(table.chunks(rows * hd));
  par_each(
    jobs,
    || (),
    |_, (((x, y), yr), table)| {
      norm(x, y);
      for ((y, yr), t) in y.chunks_exact(cols).zip(yr.chunks_exact_mut(cols)).zip(table.chunks(hd))
      {
        let (cos, sin) = t.split_at(half);
        for (y, yr) in y.chunks_exact(hd).zip(yr.chunks_exact_mut(hd)) {
          let (y0, y1) = y.split_at(half);
          let (r0, r1) = yr.split_at_mut(half);
          for i in 0..half {
            r0[i] = y0[i] * cos[i] - y1[i] * sin[i];
            r1[i] = y1[i] * cos[i] + y0[i] * sin[i];
          }
        }
      }
    },
  );
}

/// Columns for a conv1d with `[kernel, stride, pad]` over `x` `[t][ch]`: rows of `ch · kernel`,
/// `col[i][c · kernel + k] = x[stride · i + k − pad][c]` (a ggml conv weight `[kernel, ch, out]`).
pub fn im2col(x: &[f32], t: usize, ch: usize, [kernel, stride, pad]: [usize; 3]) -> Vec<f32> {
  let rows = (t + 2 * pad - kernel) / stride + 1;
  let mut col = vec![0.0; rows * ch * kernel];
  for (i, col) in col.chunks_exact_mut(ch * kernel).enumerate() {
    for k in 0..kernel {
      let s = stride * i + k;
      if s < pad || s - pad >= t {
        continue;
      }

      for (c, x) in x[(s - pad) * ch..][..ch].iter().enumerate() {
        col[c * kernel + k] = *x;
      }
    }
  }

  col
}

/// The Conformer conv module between the pointwise convs: GLU (first half · sigmoid(second half))
/// of `h` `[t][2·ch]` in place into its first halves, the depthwise conv (odd kernel, centred,
/// `dw` taps `[kernel][ch]` and bias), LayerNorm over channels (`layer_norm`) or a per-channel
/// affine `(g, b)`, SiLU, into `y` `[t][ch]`.
pub fn conv_glu(
  h: &mut [f32],
  (taps, bias): (&[f32], &[f32]),
  (g, beta): (&[f32], &[f32]),
  layer_norm: bool,
  y: &mut [f32],
) {
  let ch = bias.len();
  let (t, kernel) = (y.len() / ch, taps.len() / ch);
  let rows = chunk(t);
  par_each(
    h.chunks_mut(rows * 2 * ch),
    || (),
    |_, h| {
      for h in h.chunks_exact_mut(2 * ch) {
        let (x, gate) = h.split_at_mut(ch);
        for (x, gate) in x.iter_mut().zip(&*gate) {
          *x /= 1.0 + exp(-gate);
        }
      }
    },
  );
  let h = &*h;
  par_each(
    y.chunks_mut(rows * ch).enumerate(),
    || vec![0.0; ch],
    |acc, (j, y)| {
      for (i, y) in y.chunks_exact_mut(ch).enumerate() {
        let i = j * rows + i;
        acc.copy_from_slice(bias);
        for (k, taps) in taps.chunks_exact(ch).enumerate() {
          // Source frame plus the padding.
          let Some(src) = (i + k).checked_sub(kernel / 2).filter(|&s| s < t) else { continue };
          for ((a, w), x) in acc.iter_mut().zip(taps).zip(&h[src * 2 * ch..][..ch]) {
            *a += w * x;
          }
        }

        let (mean, rstd) = if layer_norm { moments(acc) } else { (0.0, 1.0) };
        for (((y, a), g), b) in y.iter_mut().zip(&*acc).zip(g).zip(beta) {
          *y = silu((a - mean) * rstd * g + b);
        }
      }
    },
  );
}

/// Query rows per step of `attention`: scores for 48 queries over 1500 keys take 280 KiB.
const QUERIES: usize = 48;

/// Multi-head attention with rotary positions already applied: `qk` `[t][q | k]` and `v` `[t][ch]`
/// into `o` `[t][ch]`, scores scaled by 1/√head_dim. One head per job; heads of a multiple of 16
/// dims.
pub fn attention(qk: &[f32], v: &[f32], o: &mut [f32], t: usize, heads: usize) {
  let ch = o.len() / t;
  let (hd, tp) = (ch / heads, t.next_multiple_of(NR));
  let scale = 1.0 / (hd as f32).sqrt();
  let mut outs = vec![Vec::new(); heads];
  par_each(
    outs.iter_mut().enumerate(),
    || vec![0.0; QUERIES * tp],
    |s, (h, out)| {
      let keys = panels(&qk[ch + h * hd..], 2 * ch, t, hd, hd, false);
      let values = panels(&v[h * hd..], ch, hd, tp, t, true);
      *out = vec![0.0; t * hd];
      for i in (0..t).step_by(QUERIES) {
        let m = QUERIES.min(t - i);
        product(&qk[i * 2 * ch + h * hd..], 2 * ch, m, &keys, hd, s, tp);
        softmax(s, m, t, scale);
        product(s, tp, m, &values, tp, &mut out[i * hd..], hd);
      }
    },
  );
  gather(&outs, o, hd);
}

/// Multi-head attention with Transformer-XL relative positions over `qkv` `[t][q | k | v]`, with
/// `pos` the projected positions `[pos_rows(t)][ch]` (row r is position t + 15 − r) and `uv` the
/// biases u and v `[heads][head_dim]`, into `o` `[t][ch]`: score (i, j) is
/// `((q + u)_i·k_j + (q + v)_i·pos[j − i + t + 15]) / √head_dim`. One head per job; heads of a
/// multiple of 16 dims.
pub fn relative_attention(
  qkv: &[f32],
  pos: &[f32],
  uv: (&[f32], &[f32]),
  o: &mut [f32],
  t: usize,
  heads: usize,
) {
  let ch = o.len() / t;
  let (hd, tp) = (ch / heads, t.next_multiple_of(NR));
  // A step's queries i..i + m read the position rows t + 16 − i − m..2t + 15 − i.
  let pw = (t + QUERIES + 2 * NR).next_multiple_of(NR);
  let scale = 1.0 / (hd as f32).sqrt();
  let mut outs = vec![Vec::new(); heads];
  par_each(
    outs.iter_mut().enumerate(),
    || (vec![0.0; QUERIES * tp], vec![0.0; QUERIES * pw]),
    |(s, sp), (h, out)| {
      let keys = panels(&qkv[ch + h * hd..], 3 * ch, t, hd, hd, false);
      let values = panels(&qkv[2 * ch + h * hd..], 3 * ch, hd, tp, t, true);
      let positions = panels(&pos[h * hd..], ch, pos_rows(t), hd, hd, false);
      let q = |b: &[f32]| -> Vec<f32> {
        let b = &b[h * hd..][..hd];
        let rows = qkv.chunks_exact(3 * ch).map(|r| &r[h * hd..][..hd]);
        rows.flat_map(|q| q.iter().zip(b).map(|(q, b)| q + b)).collect()
      };
      let (qu, qv) = (q(uv.0), q(uv.1));
      *out = vec![0.0; t * hd];
      for i in (0..t).step_by(QUERIES) {
        let m = QUERIES.min(t - i);
        product(&qu[i * hd..], hd, m, &keys, hd, s, tp);
        let (p0, p1) = ((t + 16 - i - m) / NR, (2 * t + 15 - i).div_ceil(NR));
        product(&qv[i * hd..], hd, m, &positions[p0 * NR * hd..p1 * NR * hd], hd, sp, pw);
        for (a, (s, sp)) in s.chunks_exact_mut(tp).zip(sp.chunks_exact(pw)).take(m).enumerate() {
          // Query i + a, key j: position row j − i − a + t + 15, from p0·NR on.
          let sp = &sp[t + 15 - i - a - p0 * NR..][..t];
          s.iter_mut().zip(sp).for_each(|(s, p)| *s += p);
        }

        softmax(s, m, t, scale);
        product(s, tp, m, &values, tp, &mut out[i * hd..], hd);
      }
    },
  );
  gather(&outs, o, hd);
}

/// Softmax of `scores` × `scale` over the first `t` of each of `m` rows; the rest of a row is
/// zeroed.
fn softmax(scores: &mut [f32], m: usize, t: usize, scale: f32) {
  let tp = t.next_multiple_of(NR);
  for row in scores.chunks_exact_mut(tp).take(m) {
    let (row, pad) = row.split_at_mut(t);
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) * scale;
    row.iter_mut().for_each(|x| *x = exp(*x * scale - max));
    let inv = 1.0 / sum(row.iter().copied());
    row.iter_mut().for_each(|x| *x *= inv);
    pad.fill(0.0);
  }
}

/// The heads' outputs `[t][head_dim]` into the rows `[t][heads · head_dim]` of `o`.
fn gather(outs: &[Vec<f32>], o: &mut [f32], hd: usize) {
  let ch = outs.len() * hd;
  for (h, out) in outs.iter().enumerate() {
    for (o, out) in o.chunks_exact_mut(ch).zip(out.chunks_exact(hd)) {
      o[h * hd..][..hd].copy_from_slice(out);
    }
  }
}
