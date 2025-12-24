#include "cuda_func.h"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

template <typename T>
void conv_2d_transpose_cuda(conv2d_transpose_context &ctx, cudaStream_t stream)
{
    std::mdspan dst_data(ctx.output_data, ctx.N, ctx.COut, ctx.HOut, ctx.WOut);
    std::mdspan input_data(ctx.input_data, ctx.N, ctx.CIn, ctx.HIn, ctx.WIn);
    std::mdspan kernel_data(static_cast<const T*>(ctx.kernel_data), ctx.CIn, ctx.COut, ctx.Kh, ctx.Kw);
    launch_functor(stream, std::make_tuple(ctx.N, ctx.COut, ctx.HOut, ctx.WOut),
        [=] __device__(int64_t n, int64_t cout, int64_t hout, int64_t wout) {
            float accumulator = 0;

            for (int64_t cin = 0; cin < ctx.CIn; cin++) {
                for (int64_t kh = 0; kh < ctx.Kh; ++kh) {
                    int64_t hin = hout + ctx.padding_h - kh * ctx.dilation_h;
                    if (hin < 0 || hin % ctx.stride_h) continue;
                    hin /= ctx.stride_h;
                    if (hin >= ctx.HIn) continue;

                    for (int64_t kw = 0; kw < ctx.Kw; ++kw) {
                        int64_t win = wout + ctx.padding_w - kw * ctx.dilation_w;
                        if (win < 0 || win % ctx.stride_w) continue;
                        win /= ctx.stride_w;
                        if (win >= ctx.WIn) continue;

                        accumulator += input_data(n, cin, hin, win) *
                            ggml_cuda_cast<float>(kernel_data(cin, cout, kh, kw));
                    }
                }
            }

            dst_data(n, cout, hout, wout) = accumulator;
        }
    );
}

void conv_2d_transpose_cuda(conv2d_transpose_context& ctx, cudaStream_t stream)
{
    if (ctx.kernel_type == internal::GGML_TYPE_F16) {
        conv_2d_transpose_cuda<half>(ctx, stream);
    }
    else {
        conv_2d_transpose_cuda<float>(ctx, stream);
    }
}