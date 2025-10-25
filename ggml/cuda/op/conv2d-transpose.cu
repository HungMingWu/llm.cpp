#include "cuda_func.h"
#include "convert.cuh"
#include "helper.h"
#include "launch.cuh"

__global__ void conv2d_transpose_kernel(const float* __restrict__ input, const half* __restrict__ kernel,
    float* __restrict__ output, const int in_w, const int in_h, const int out_w,
    const int out_h, const int Kw, const int Kh, const int stride,
    const int c_in, const int c_out, const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_elements = out_w * out_h * c_out * batches;

    if (global_idx >= total_elements) {
        return;
    }
	std::experimental::mdspan dst_data(output, batches, c_out, out_h, out_w);
    std::experimental::mdspan input_data(input, batches, c_in, in_h, in_w);
    std::experimental::mdspan kernel_data(kernel, c_in, c_out, Kh, Kw);
    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx = global_idx / (out_w * out_h * c_out);

    float accumulator = 0;
    // For each output idx, find the inputs that contribute to it by checking stride alignment and bounds

    for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
        for (int kh = 0; kh < Kh; ++kh) {
            int in_y = out_y_idx - kh;
            if (in_y < 0 || in_y % stride) continue;
            in_y /= stride;
            if (in_y >= in_h) continue;

            for (int kw = 0; kw < Kw; ++kw) {
                int in_x = out_x_idx - kw;
                if (in_x < 0 || in_x % stride) continue;
                in_x /= stride;
                if (in_x >= in_w) continue;

                float input_val = input_data(n_idx, c_in_idx, in_y, in_x);
                half  kern_val = kernel_data(c_in_idx, c_idx, kh, kw);

                accumulator += input_val * ggml_cuda_cast<float>(kern_val);
            }
        }
    }

    dst_data(n_idx, c_idx, out_y_idx, out_x_idx) = accumulator;
}

void conv_2d_transpose_cuda(conv2d_transpose_context &ctx, cudaStream_t stream)
{
#if 0
    static constexpr size_t CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE = 256;

    const int total = (ctx->output_w * ctx->output_h * ctx->channels_out * ctx->batches);
    const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    conv2d_transpose_kernel << <blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, stream >> > (
        ctx->input_data, ctx->kernel_data, ctx->output_data,
        ctx->input_w /*in_w*/, ctx->input_h /*in_h*/, ctx->output_w /*out_w*/, ctx->output_h /*out_h*/,
        ctx->Kw, ctx->Kh, ctx->stride,
        ctx->channels_out /*c_out*/, ctx->batches);
#else
    std::experimental::mdspan dst_data(ctx.output_data, ctx.N, ctx.COut, ctx.HOut, ctx.WOut);
    std::experimental::mdspan input_data(ctx.input_data, ctx.N, ctx.CIn, ctx.HIn, ctx.WIn);
    std::experimental::mdspan kernel_data(ctx.kernel_data, ctx.CIn, ctx.COut, ctx.Kh, ctx.Kw);
    launch_functor(stream, std::make_tuple(ctx.N, ctx.COut, ctx.HOut, ctx.WOut),
        [=] __device__(int64_t n, int64_t cout, int64_t hout, int64_t wout) {
            float accumulator = 0;

            for (int64_t cin = 0; cin < ctx.CIn; cin++) {
                for (int64_t kh = 0; kh < ctx.Kh; ++kh) {
                    int64_t hin = hout - kh;
                    if (hin < 0 || hin % ctx.stride) continue;
                    hin /= ctx.stride;
                    if (hin >= ctx.HIn) continue;

                    for (int64_t kw = 0; kw < ctx.Kw; ++kw) {
                        int64_t win = wout - kw;
                        if (win < 0 || win % ctx.stride) continue;
                        win /= ctx.stride;
                        if (win >= ctx.WIn) continue;

                        accumulator += input_data(n, cin, hin, win) *
                            ggml_cuda_cast<float>(kernel_data(cin, cout, kh, kw));
                    }
                }
            }

            dst_data(n, cout, hout, wout) = accumulator;
        }
    );
#endif
}