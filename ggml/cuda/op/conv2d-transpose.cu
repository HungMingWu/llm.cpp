#include "cuda_func.h"

__global__ void conv2d_transpose_kernel(const float* __restrict__ input, const half* __restrict__ kernel,
    float* __restrict__ output, const int in_w, const int in_h, const int out_w,
    const int out_h, const int kernel_w, const int kernel_h, const int stride,
    const int c_in, const int c_out, const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_elements = out_w * out_h * c_out * batches;

    if (global_idx >= total_elements) {
        return;
    }

    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx = global_idx / (out_w * out_h * c_out);

    float accumulator = 0;
    // For each output idx, find the inputs that contribute to it by checking stride alignment and bounds

    for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_y = out_y_idx - kh;
            if (in_y < 0 || in_y % stride) continue;
            in_y /= stride;
            if (in_y >= in_h) continue;

            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_x = out_x_idx - kw;
                if (in_x < 0 || in_x % stride) continue;
                in_x /= stride;
                if (in_x >= in_w) continue;

                const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + (in_w)*in_y + in_x;
                const int kernel_idx =
                    (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + (kernel_w)*kh + kw;

                float input_val = input[input_idx];
                half  kern_val = kernel[kernel_idx];

                accumulator += input_val * (float)kern_val;
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + (out_w)*out_y_idx + out_x_idx] = accumulator;
}

void conv_2d_transpose_p0_cuda(conv2d_transpose_context* ctx, cudaStream_t stream)
{
    static constexpr size_t CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE = 256;

    const int total = (ctx->output_w * ctx->output_h * ctx->channels_out * ctx->batches);
    const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    conv2d_transpose_kernel << <blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, stream >> > (
        ctx->input_data, ctx->kernel_data, ctx->output_data,
        ctx->input_w, ctx->input_h, ctx->output_w, ctx->output_h,
        ctx->kernel_w, ctx->kernel_h, ctx->stride,
        ctx->channels_in, ctx->channels_out, ctx->batches);
}