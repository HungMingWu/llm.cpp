#include "../common.h"
#define GGML_ASSERT(...)
#define GGML_ABORT(...)
#define GGML_UNUSED(x) (void)(x)

template <size_t splitD, size_t N>
__global__ void __launch_bounds__(splitD, 2)
ssm_scan_f32(const float* __restrict__ src0, const float* __restrict__ src1, const float* __restrict__ src2,
    const float* __restrict__ src3, const float* __restrict__ src4, const float* __restrict__ src5,
    const int src0_nb1, const int src0_nb2, const int src1_nb0, const int src1_nb1, const int src1_nb2,
    const int src1_nb3, const int src2_nb0, const int src2_nb1, const int src2_nb2, const int src3_nb1,
    const int src4_nb1, const int src4_nb2, const int src5_nb1, const int src5_nb2,
    float* __restrict__ dst, const int64_t L) {
    GGML_UNUSED(src1_nb0);
    GGML_UNUSED(src2_nb0);

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int bidx = blockIdx.x;  // split along B
    const int bidy = blockIdx.y;  // split along D
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int wtid = tid % 32;

    extern __shared__ float smem[];
    const int               stride_sA = N + 1;
    const int               stride_ss0 = N + 1;
    float* smem_A = smem;
    float* smem_s0 = smem_A + splitD * stride_sA;

    const float* s0_block = (const float*)((const char*)src0 + bidx * src0_nb2 + bidy * splitD * src0_nb1);
    const float* x_block = (const float*)((const char*)src1 + (bidx * src1_nb2) + bidy * splitD * sizeof(float));
    const float* dt_block = (const float*)((const char*)src2 + (bidx * src2_nb2) + bidy * splitD * sizeof(float));
    const float* A_block = (const float*)((const char*)src3 + bidy * splitD * src3_nb1);
    const float* B_block = (const float*)((const char*)src4 + (bidx * src4_nb2));
    const float* C_block = (const float*)((const char*)src5 + (bidx * src5_nb2));
    float* y_block = (float*)((char*)dst + (bidx * src1_nb2) + bidy * splitD * sizeof(float));
    float* s_block = (float*)((char*)dst + src1_nb3 + bidx * src0_nb2 + bidy * splitD * src0_nb1);

    const int stride_s0 = src0_nb1 / sizeof(float);
    const int stride_x = src1_nb1 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_A = src3_nb1 / sizeof(float);
    const int stride_B = src4_nb1 / sizeof(float);
    const int stride_C = src5_nb1 / sizeof(float);
    const int stride_s = stride_s0;
    const int stride_y = stride_x;

    // can N not be 16? for example 32?
    if (N == 16) {
#pragma unroll
        for (size_t i = 0; i < splitD / 4; i += 2) {
            float value = A_block[(wid * warp_size + i) * stride_A + wtid];
            // todo: bank conflict
            // I am always confused with how to use the swizzling method to solve
            // bank conflit. Hoping somebody can tell me.
            smem_A[(wid * warp_size + i) * stride_sA + wtid + ((wtid / 16) > 0 ? 1 : 0)] = value;
        }
#pragma unroll
        for (size_t i = 0; i < splitD / 4; i += 2) {
            float value = s0_block[(wid * warp_size + i) * stride_s0 + wtid];
            smem_s0[(wid * warp_size + i) * stride_ss0 + wtid + ((wtid / 16) > 0 ? 1 : 0)] = value;
        }
    }

    __syncthreads();

    for (int64_t i = 0; i < L; i++) {
        float dt_soft_plus = dt_block[i * stride_dt + tid];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = log1pf(exp(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + tid] * dt_soft_plus;
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < N; j++) {
            float state = (smem_s0[tid * stride_ss0 + j] * expf(dt_soft_plus * smem_A[tid * stride_sA + j])) +
                (B_block[i * stride_B + j] * x_dt);
            sumf += state * C_block[i * stride_C + j];
            if (i == L - 1) {
                s_block[tid * stride_s + j] = state;
            }
            else {
                smem_s0[tid * stride_ss0 + j] = state;
            }
        }
        __syncthreads();
        y_block[i * stride_y + tid] = sumf;
    }
}

void ssm_scan_f32_cuda(const float* src0, const float* src1, const float* src2, const float* src3,
    const float* src4, const float* src5, const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1, const int src1_nb2, const int src1_nb3,
    const int src2_nb0, const int src2_nb1, const int src2_nb2, const int src3_nb1,
    const int src4_nb1, const int src4_nb2, const int src5_nb1, const int src5_nb2,
    float* dst, const int64_t N, const int64_t D, const int64_t L, const int64_t B,
    cudaStream_t stream) {
    const int threads = 128;
    // todo: consider D cannot be divided,does this situation exist?
    GGML_ASSERT(D % threads == 0);
    const dim3 blocks(B, (D + threads - 1) / threads, 1);
    const int  smem_size = (threads * (N + 1) * 2) * sizeof(float);
    if (N == 16) {
        ssm_scan_f32<128, 16> << <blocks, threads, smem_size, stream >> > (
            src0, src1, src2, src3, src4, src5, src0_nb1, src0_nb2, src1_nb0, src1_nb1, src1_nb2, src1_nb3, src2_nb0,
            src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2, dst, L);
    }
    else {
        GGML_ABORT("doesn't support N!=16.");
    }
}