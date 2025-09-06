#include <span>
#include "../vendors/cuda.h"

static  __global__ void conv_transpose_1d_kernel(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2, const int /*src0_ne3*/,
    const int src1_ne0, const int /*src1_ne1*/, const int /*src1_ne2*/, const int /*src1_ne3*/,
    const int dst_ne0, const int /*dst_ne1*/, const int /*dst_ne2*/, const int /*dst_ne3*/,
    const float* src0, const float* src1, float* dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i * s0 && idx < i * s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i * s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value = src1[input_offset + i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

void conv_transpose_1d_f32_f32_cuda(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
    const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
    const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
    const float* src0, const float* src1, float* dst,
    cudaStream_t stream) {
    static constexpr size_t CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE = 256;
    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel << <num_blocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream >> > (
        s0, output_size,
        src0_ne0, src0_ne1, src0_ne2, src0_ne3,
        src1_ne0, src1_ne1, src1_ne2, src1_ne3,
        dst_ne0, dst_ne1, dst_ne2, dst_ne3,
        src0, src1, dst);
}
