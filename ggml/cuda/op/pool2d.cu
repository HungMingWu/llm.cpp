#include <assert.h>
#include <float.h>
#include "common.cuh"
#include "cuda_func.h"

#define CUDA_POOL2D_BLOCK_SIZE 256

template <typename Ti, typename To>
static  __global__ void pool2d_nchw_kernel(
    const int ih, const int iw, const int oh, const int ow,
    const int kh, const int kw, const int sh, const int sw,
    const int ph, const int pw, const int parallel_elements,
    const Ti* src, To* dst, const internal::ggml_op_pool op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int I_HW = ih * iw;
    const int O_HW = oh * ow;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / ow;
    const int cur_ow = idx % O_HW % ow;
    const Ti* i_ptr = src + nc * I_HW;
    To* o_ptr = dst + nc * O_HW;
    const int start_h = cur_oh * sh - ph;
    const int bh = max(0, start_h);
    const int eh = min(ih, start_h + kh);
    const int start_w = cur_ow * sw - pw;
    const int bw = max(0, start_w);
    const int ew = min(iw, start_w + kw);
    const To scale = 1. / (kh * kw);
    To res = 0;

    switch (op) {
    case internal::GGML_OP_POOL_AVG: res = 0; break;
    case internal::GGML_OP_POOL_MAX: res = -FLT_MAX; break;
    default: assert(false);
    }

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
#if __CUDA_ARCH__ >= 350
            Ti cur = __ldg(i_ptr + i * iw + j);
#else
            Ti cur = i_ptr[i * iw + j];
#endif
            switch (op) {
            case internal::GGML_OP_POOL_AVG: res += cur * scale; break;
            case internal::GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
            default: assert(false);
            }
        }
    }
    o_ptr[cur_oh * ow + cur_ow] = res;
}

void pool2d_nchw_kernel_f32_f32_cuda(
    const int ih, const int iw, const int oh, const int ow,
    const int kh, const int kw, const int sh, const int sw,
    const int ph, const int pw, const int parallel_elements,
    const float* src, float* dst, internal::ggml_op_pool op,
    cudaStream_t stream) {

    const int num_blocks = (parallel_elements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool2d_nchw_kernel << <block_nums, CUDA_POOL2D_BLOCK_SIZE, 0, stream >> > (ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
}