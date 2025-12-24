#include "cuda_func.h"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

__device__ __forceinline__ int64_t calculate_input_coord(int64_t out_coord,
    int64_t kern_coord,
    int64_t stride,
    int64_t dilation,
    int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

template <typename T>
void conv2d_cuda(const conv2d_context& ctx, cudaStream_t stream) {
    std::mdspan input_data(ctx.input_d, ctx.N, ctx.CIn, ctx.IH, ctx.IW);
    std::mdspan kernel_data(static_cast<const T*>(ctx.kernel_d), ctx.N, ctx.CIn, ctx.KH, ctx.KW);
    std::mdspan output_data(ctx.output_d, ctx.N, ctx.COut, ctx.OH, ctx.OW);
    launch_functor(stream, std::make_tuple(ctx.N, ctx.COut, ctx.OH, ctx.OW),
        [=] __device__(int64_t n, int64_t cout, int64_t oh, int64_t ow) {
            const int64_t kh_min = std::max(int64_t{ 0 }, (ctx.pad_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
            const int64_t kh_max = std::min(ctx.KH, (ctx.IH + ctx.pad_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
            const int64_t kw_min = std::max(int64_t{ 0 }, (ctx.pad_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);
            const int64_t kw_max = std::min(ctx.KW, (ctx.IW + ctx.pad_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);
            float accumulator = 0.0f;

            for (int64_t cin = 0; cin < ctx.CIn; ++cin) {
                for (int64_t kh = kh_min; kh < kh_max; kh++) {
                    const int64_t ih = calculate_input_coord(oh, kh, ctx.stride_h, ctx.dilation_h, ctx.pad_h);

                    for (int64_t kw = kw_min; kw < kw_max; kw++) {
                        const int64_t iw = calculate_input_coord(ow, kw, ctx.stride_w, ctx.dilation_w, ctx.pad_w);

                        accumulator += input_data(n, cin, ih, iw) *
                            ggml_cuda_cast<float>(kernel_data(cout, cin, kh, kw));
                    }
                }
            }

            output_data(n, cout, oh, ow) = accumulator;
        }
    );
}

void conv2d_cuda(const conv2d_context& ctx, cudaStream_t stream)
{
    if (ctx.kernel_type == internal::GGML_TYPE_F16) {
        conv2d_cuda<half>(ctx, stream);
    }
    else {
        conv2d_cuda<float>(ctx, stream);
    }
}