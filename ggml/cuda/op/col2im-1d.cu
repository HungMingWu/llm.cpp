#include "cuda_func.h"
#include "convert.cuh"
#include "common.cuh"

// col2im_1d: scatter-add GEMM columns to 1D signal (gather approach)
// columns: [K*OC, T_in]  ->  output: [T_out, OC]
// Supports F32, F16, BF16 data with F32 accumulator.

template <typename T>
static __global__ void col2im_1d_kernel(
    const T* __restrict__ col,
    T* __restrict__ dst,
    const int T_in, const uint3 T_out_fd,
    const int OC, const int K, const int K_OC,
    const int s0, const int p0, const int total) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total) return;

    // dst layout: [T_out, OC], ne[0]=T_out fastest
    const uint2 qr = fast_div_modulo((uint32_t)idx, T_out_fd);  // qr.x = idx / T_out, qr.y = idx % T_out
    const int oc = (int)qr.x;
    const int t_out = (int)qr.y;
    const int t_abs = t_out + p0;  // absolute position in uncropped signal

    // Gather: find all (t_in, k) where t_in*s + k == t_abs, 0 <= k < K
    int t_in_min = (t_abs - K + s0) / s0;  // ceil((t_abs - K + 1) / s)
    if (t_in_min < 0) t_in_min = 0;
    int t_in_max = t_abs / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
        const int k = t_abs - t_in * s0;
        // col layout: [K*OC, T_in], column index = oc * K + k
        sum += ggml_cuda_cast<float>(col[(oc * K + k) + t_in * K_OC]);
    }

    dst[idx] = ggml_cuda_cast<T>(sum);
}

void col2im_1d_cuda(const col2im_1d_context& ctx, cudaStream_t stream) {
    const int32_t s0 = ctx.s0;
    const int32_t OC = ctx.OC;
    const int32_t p0 = ctx.p0;

    const int K_OC = ctx.K_OC;
    const int T_in = ctx.T_in;
    const int K = K_OC / OC;
    const int T_out = ctx.T_out;

    const uint3 T_out_fd = init_fastdiv_values((uint32_t)T_out);

    const int total = T_out * OC;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    switch (ctx.src0_type) {
    case internal::GGML_TYPE_F32: {
        col2im_1d_kernel << <num_blocks, block_size, 0, stream >> > (
            (const float*)ctx.src0_d, (float*)ctx.dst_d,
            T_in, T_out_fd, OC, K, K_OC, s0, p0, total);
    } break;
    case internal::GGML_TYPE_F16: {
        col2im_1d_kernel << <num_blocks, block_size, 0, stream >> > (
            (const half*)ctx.src0_d, (half*)ctx.dst_d,
            T_in, T_out_fd, OC, K, K_OC, s0, p0, total);
    } break;
    case internal::GGML_TYPE_BF16: {
        col2im_1d_kernel << <num_blocks, block_size, 0, stream >> > (
            (const nv_bfloat16*)ctx.src0_d, (nv_bfloat16*)ctx.dst_d,
            T_in, T_out_fd, OC, K, K_OC, s0, p0, total);
    } break;
    default:
        GGML_ABORT("col2im_1d: unsupported type");
    }
}
