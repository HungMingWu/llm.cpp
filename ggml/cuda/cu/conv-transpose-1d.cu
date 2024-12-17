#include <span>
#include "mdspan.hpp"
#include "../vendors/cuda.h"

namespace stdex = std::experimental;
#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

template <typename T1, typename T2, typename T3>
static  __global__ void conv_transpose_1d_kernel(
    const int s0,
    T1 src0,
    T2 src1,
    T3 dst) {
    size_t global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= dst.size()) {
        return;
    }

    size_t out_index = global_index / dst.extent(1);
    size_t idx = global_index % dst.extent(1);
    float accumulator = 0;
    for (size_t c = 0; c < src0.extent(0); c++) {
        for (size_t i = 0; i < src1.extent(1); i++) {
            if (idx < i * s0 || idx >= i * s0 + src0.extent(2)) {
                continue;
            }
            int weight_idx = idx - i * s0;

			float kernel_weight = src0(c, out_index, weight_idx);
            float input_value = src1(c, i);

            accumulator += kernel_weight * input_value;
        }
    }
    dst(out_index, idx) = accumulator;
}

void conv_transpose_1d_f32_f32_cuda(
    const int s0,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int src1_ne1,
    const int dst_ne0, const int dst_ne1,
    const float* src0, const float* src1, float* dst, size_t dst_size,
    cudaStream_t stream) {

    assert(src0_ne2 == src1_ne1);
    const int num_blocks = (dst_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    stdex::mdspan src0_span{ src0, src0_ne2, src0_ne1, src0_ne0 };
    stdex::mdspan src1_span{ src1, src1_ne1, src1_ne0 };
    stdex::mdspan dst_span{ dst, dst_ne1, dst_ne0 };
    conv_transpose_1d_kernel << <num_blocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream >> > (
        s0,
        src0_span,
        src1_span,
        dst_span);
}
