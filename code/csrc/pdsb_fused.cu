// pdsb_fused.cu -- fused CUDA step of PdSB (simplex Potts dSB), torch extension.
//
// Dense graphs: H = W @ OH by cuBLAS (caller).  Sparse graphs (n > dense_max): sparse_h kernel below,
//   H[i,r,c] = sum_{j in N(i)} w_ij [col_cur[j,r] == c]   (CSR, FP32, fixed summation order -> exact for
//   integer weights), reading the colours directly, so the one-hot buffer is not needed (oh_nxt empty).
//
// Same structure as cdsb (clock_dsb/code/cdsb/src/engine.cu): one GEMM + one fused kernel + one per-replica
// reduce kernel per step, the whole chunk captured in a CUDA graph by the caller.
//
//   GEMM (caller, cuBLAS FP16 tensor cores):  H = W @ OH_cur,  OH = one-hot colours [n][R][K]
//            -> integer counts, exact in FP16 for integer weights with row abs-sum <= 2048
//   step kernel, per (vertex i, replica r):
//     * energy partial: E_r += H[i,r,c_i]          (sum_i H[i,r,c_i] = 2 x same-colour weight, exact)
//     * commit: if flag[r] (state of step t-1 improved), best_col[i,r] = col of step t-1, which lives in the
//       buffer the new colour is about to overwrite (ping-pong)
//     * force f = -c0 sum_c H[i,r,c] e_c ;  p += dt(-(a0-a_t) x + f) ; x += dt a0 p
//     * inelastic wall on the simplex hull (barycentric coords, Michelot projection, outward momentum removed;
//       points less than kWallTol outside count as inside, which makes the wall insensitive to rounding)
//     * new colour argmax_c <x, e_c>, written as uint8 and as the next one-hot (GEMM input)
//     block reduction over its vertices -> part[blockIdx.y][r]   (no atomics, deterministic)
//   reduce kernel, per replica: E = sum_b part[b][r] in fixed order; flag = E < best_E; best_E = min; t += 1.
//
// Thread layout: block = 32 replicas (lanes) x kBlockY vertex groups, blockIdx.y = tile of kRows vertices.
// Consecutive lanes = consecutive replicas of one vertex -> contiguous loads of H / OH / x / p.
// K is a template parameter (3, 4, 5, 6, 8); D = K - 1.  Per-replica c0[r], dt[r]: a whole hyper-parameter
// grid runs as one batch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace {

constexpr int kBlockY = 8;
constexpr int kRows = 64;
constexpr float kWallTol = 1e-4f;   // = machines.WALL_TOL: points within 1e-4 of the hull count as inside

template <int K>
struct Verts {
  float v[K * (K - 1)];  // vertex c, coordinate d at v[c*(K-1)+d]
};

__device__ __forceinline__ float ldh(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ldh(const float* p) { return *p; }
__device__ __forceinline__ void sth(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ void sth(float* p, float v) { *p = v; }

template <class HT, int K>
__global__ void __launch_bounds__(32 * kBlockY)
step_kernel(const HT* __restrict__ H, float* __restrict__ x, float* __restrict__ p,
            const uint8_t* __restrict__ col_cur, uint8_t* __restrict__ col_nxt, HT* __restrict__ oh_nxt,
            uint8_t* __restrict__ best_col, const uint8_t* __restrict__ flag, double* __restrict__ part,
            const float* __restrict__ c0v, const float* __restrict__ dtv, const int* __restrict__ tstep, int T,
            float a0, int n, int R, Verts<K> V) {
  constexpr int D = K - 1;
  const int lane = threadIdx.x & 31;
  const int gy = threadIdx.x >> 5;
  const int r = blockIdx.x * 32 + lane;
  double esum = 0.0;
  if (r < R) {
    const float at = a0 * float(*tstep) / float(T);
    const float det = a0 - at;
    const float c0 = c0v[r], dt = dtv[r];
    const bool commit = flag[r] != 0;
    const int i_end = min(n, int(blockIdx.y + 1) * kRows);
    for (int i = blockIdx.y * kRows + gy; i < i_end; i += kBlockY) {
      const size_t e = size_t(i) * R + r;
      float h[K];
#pragma unroll
      for (int c = 0; c < K; ++c) h[c] = ldh(H + e * K + c);
      const int cc = col_cur[e];
      float hc = 0.f;
#pragma unroll
      for (int c = 0; c < K; ++c) hc = (c == cc) ? h[c] : hc;
      esum += double(hc);

      float xv[D], pv[D];
#pragma unroll
      for (int d = 0; d < D; ++d) {
        float f = 0.f;
#pragma unroll
        for (int c = 0; c < K; ++c) f -= h[c] * V.v[c * D + d];
        const float xd = x[e * D + d];
        const float pd = p[e * D + d] + dt * (-det * xd + c0 * f);
        pv[d] = pd;
        xv[d] = xd + dt * a0 * pd;
      }
      // wall: barycentric b_c = ((K-1)<x,e_c> + 1)/K, outside iff some b_c < 0
      float b[K];
      bool out = false;
#pragma unroll
      for (int c = 0; c < K; ++c) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < D; ++d) s += xv[d] * V.v[c * D + d];
        b[c] = (float(K - 1) * s + 1.f) / float(K);
        out |= b[c] < -kWallTol;
      }
      if (out) {
        bool act[K];
#pragma unroll
        for (int c = 0; c < K; ++c) act[c] = true;
        float theta = 0.f;
#pragma unroll
        for (int it = 0; it < K; ++it) {        // Michelot: exact after <= K rounds
          float sum = 0.f;
          int m = 0;
#pragma unroll
          for (int c = 0; c < K; ++c)
            if (act[c]) { sum += b[c]; ++m; }
          theta = (sum - 1.f) / float(max(m, 1));
#pragma unroll
          for (int c = 0; c < K; ++c) act[c] = b[c] > theta;
        }
        float xp[D];
#pragma unroll
        for (int d = 0; d < D; ++d) xp[d] = 0.f;
#pragma unroll
        for (int c = 0; c < K; ++c) {
          const float bc = fmaxf(b[c] - theta, 0.f);
#pragma unroll
          for (int d = 0; d < D; ++d) xp[d] += bc * V.v[c * D + d];
        }
        float nn[D], n2 = 0.f;
#pragma unroll
        for (int d = 0; d < D; ++d) { nn[d] = xv[d] - xp[d]; n2 += nn[d] * nn[d]; }
        const float inv = rsqrtf(fmaxf(n2, 1e-30f));
        float pn = 0.f;
#pragma unroll
        for (int d = 0; d < D; ++d) { nn[d] *= inv; pn += pv[d] * nn[d]; }
        pn = fmaxf(pn, 0.f);
#pragma unroll
        for (int d = 0; d < D; ++d) { pv[d] -= pn * nn[d]; xv[d] = xp[d]; }
      }
      // new colour = argmax_c <x, e_c>  (first maximum, as torch.argmax)
      int nc = 0;
      float bv = 0.f;
#pragma unroll
      for (int c = 0; c < K; ++c) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < D; ++d) s += xv[d] * V.v[c * D + d];
        if (c == 0 || s > bv) { bv = s; nc = c; }
      }
#pragma unroll
      for (int d = 0; d < D; ++d) { x[e * D + d] = xv[d]; p[e * D + d] = pv[d]; }
      if (commit) best_col[e] = col_nxt[e];     // colour of step t-1, about to be overwritten
      col_nxt[e] = uint8_t(nc);
      if (oh_nxt != nullptr) {
#pragma unroll
        for (int c = 0; c < K; ++c) sth(oh_nxt + e * K + c, c == nc ? 1.f : 0.f);
      }
    }
  }
  __shared__ double s_e[kBlockY][32];
  s_e[gy][lane] = esum;
  __syncthreads();
  if (gy == 0 && r < R) {
    double s = 0.0;
#pragma unroll
    for (int y = 0; y < kBlockY; ++y) s += s_e[y][lane];
    part[size_t(blockIdx.y) * R + r] = s;
  }
}

template <int K>
__global__ void __launch_bounds__(128)
sparse_h_kernel(const int64_t* __restrict__ indptr, const int* __restrict__ indices, const float* __restrict__ w,
                const uint8_t* __restrict__ col, float* __restrict__ H, int n, int R) {
  // block = 128 consecutive replicas of one vertex row; rows strided over gridDim.y
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    float acc[K];
#pragma unroll
    for (int c = 0; c < K; ++c) acc[c] = 0.f;
    const int64_t e1 = indptr[i + 1];
    for (int64_t e = indptr[i]; e < e1; ++e) {
      const float wj = w[e];
      const int cj = col[size_t(indices[e]) * R + r];
#pragma unroll
      for (int c = 0; c < K; ++c) acc[c] += (c == cj) ? wj : 0.f;
    }
    float* h = H + (size_t(i) * R + r) * K;
#pragma unroll
    for (int c = 0; c < K; ++c) h[c] = acc[c];
  }
}

__global__ void reduce_kernel(const double* __restrict__ part, int nby, int R, double* __restrict__ best_e,
                              uint8_t* __restrict__ flag, int* __restrict__ tstep) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < R) {
    double e = 0.0;
    for (int b = 0; b < nby; ++b) e += part[size_t(b) * R + r];
    const bool imp = e < best_e[r];
    if (imp) best_e[r] = e;
    flag[r] = imp ? 1 : 0;
  }
  if (r == 0) *tstep += 1;   // read by the next step kernel (stream order)
}

template <class HT, int K>
void launch_step(torch::Tensor H, torch::Tensor x, torch::Tensor p, torch::Tensor col_cur, torch::Tensor col_nxt,
                 torch::Tensor oh_nxt, torch::Tensor best_col, torch::Tensor flag, torch::Tensor part,
                 torch::Tensor c0, torch::Tensor dt, torch::Tensor tstep, int T, double a0, torch::Tensor V) {
  const int n = int(x.size(0)), R = int(x.size(1));
  Verts<K> vv;
  auto Vc = V.to(torch::kCPU, torch::kFloat32).contiguous();
  const float* vp = Vc.data_ptr<float>();
  for (int k = 0; k < K * (K - 1); ++k) vv.v[k] = vp[k];
  dim3 block(32 * kBlockY);
  dim3 grid((R + 31) / 32, (n + kRows - 1) / kRows);
  auto stream = at::cuda::getCurrentCUDAStream();
  step_kernel<HT, K><<<grid, block, 0, stream>>>(
      reinterpret_cast<const HT*>(H.data_ptr()), x.data_ptr<float>(), p.data_ptr<float>(),
      col_cur.data_ptr<uint8_t>(), col_nxt.data_ptr<uint8_t>(),
      oh_nxt.numel() ? reinterpret_cast<HT*>(oh_nxt.data_ptr()) : nullptr,
      best_col.data_ptr<uint8_t>(), flag.data_ptr<uint8_t>(), part.data_ptr<double>(), c0.data_ptr<float>(),
      dt.data_ptr<float>(), tstep.data_ptr<int>(), T, float(a0), n, R, vv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class HT>
void dispatch_k(int K, torch::Tensor H, torch::Tensor x, torch::Tensor p, torch::Tensor col_cur,
                torch::Tensor col_nxt, torch::Tensor oh_nxt, torch::Tensor best_col, torch::Tensor flag,
                torch::Tensor part, torch::Tensor c0, torch::Tensor dt, torch::Tensor tstep, int T, double a0,
                torch::Tensor V) {
  switch (K) {
    case 3: launch_step<HT, 3>(H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, T, a0, V); break;
    case 4: launch_step<HT, 4>(H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, T, a0, V); break;
    case 5: launch_step<HT, 5>(H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, T, a0, V); break;
    case 6: launch_step<HT, 6>(H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, T, a0, V); break;
    case 8: launch_step<HT, 8>(H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, T, a0, V); break;
    default: TORCH_CHECK(false, "fused PdSB: K must be 3, 4, 5, 6 or 8");
  }
}

}  // namespace

void step(torch::Tensor H, torch::Tensor x, torch::Tensor p, torch::Tensor col_cur, torch::Tensor col_nxt,
          torch::Tensor oh_nxt, torch::Tensor best_col, torch::Tensor flag, torch::Tensor part, torch::Tensor c0,
          torch::Tensor dt, torch::Tensor tstep, int64_t T, double a0, torch::Tensor V) {
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_contiguous() && p.is_contiguous() && H.is_contiguous() && oh_nxt.is_contiguous(), "contiguous");
  const int K = int(V.size(0));
  TORCH_CHECK(V.size(1) == K - 1 && H.size(-1) == K, "shape");
  if (H.scalar_type() == torch::kHalf)
    dispatch_k<__half>(K, H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, int(T), a0, V);
  else
    dispatch_k<float>(K, H, x, p, col_cur, col_nxt, oh_nxt, best_col, flag, part, c0, dt, tstep, int(T), a0, V);
}

void sparse_h(torch::Tensor indptr, torch::Tensor indices, torch::Tensor w, torch::Tensor col, torch::Tensor H) {
  const at::cuda::CUDAGuard guard(H.device());
  TORCH_CHECK(indptr.scalar_type() == torch::kInt64 && indices.scalar_type() == torch::kInt32 &&
              w.scalar_type() == torch::kFloat32 && col.scalar_type() == torch::kUInt8 &&
              H.scalar_type() == torch::kFloat32, "sparse_h: dtypes int64/int32/float32/uint8/float32");
  TORCH_CHECK(H.is_contiguous() && col.is_contiguous(), "sparse_h: contiguous");
  const int n = int(col.size(0)), R = int(col.size(1)), K = int(H.size(-1));
  dim3 block(128);
  dim3 grid((R + 127) / 128, std::min(n, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t* ip = indptr.data_ptr<int64_t>();
  const int* ix = indices.data_ptr<int>();
  const float* wp = w.data_ptr<float>();
  const uint8_t* cp = col.data_ptr<uint8_t>();
  float* hp = H.data_ptr<float>();
  switch (K) {
    case 3: sparse_h_kernel<3><<<grid, block, 0, stream>>>(ip, ix, wp, cp, hp, n, R); break;
    case 4: sparse_h_kernel<4><<<grid, block, 0, stream>>>(ip, ix, wp, cp, hp, n, R); break;
    case 5: sparse_h_kernel<5><<<grid, block, 0, stream>>>(ip, ix, wp, cp, hp, n, R); break;
    case 6: sparse_h_kernel<6><<<grid, block, 0, stream>>>(ip, ix, wp, cp, hp, n, R); break;
    case 8: sparse_h_kernel<8><<<grid, block, 0, stream>>>(ip, ix, wp, cp, hp, n, R); break;
    default: TORCH_CHECK(false, "sparse_h: K must be 3, 4, 5, 6 or 8");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce(torch::Tensor part, torch::Tensor best_e, torch::Tensor flag, torch::Tensor tstep) {
  const at::cuda::CUDAGuard guard(part.device());
  const int nby = int(part.size(0)), R = int(part.size(1));
  reduce_kernel<<<(R + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      part.data_ptr<double>(), nby, R, best_e.data_ptr<double>(), flag.data_ptr<uint8_t>(), tstep.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t rows_per_block() { return kRows; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step, "fused PdSB step");
  m.def("reduce", &reduce, "per-replica energy reduce + best commit flag");
  m.def("sparse_h", &sparse_h, "H = W @ onehot(col) for CSR W (exact, FP32)");
  m.def("rows_per_block", &rows_per_block);
}
