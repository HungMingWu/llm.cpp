#include "cuda_func.h"
#include "mdspan_helper.h"
#include "launch.cuh"

#define GGML_ABORT(...)

__device__ __forceinline__ int calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

void conv2d_dw_nchw(const conv2d_dw_context& ctx, cudaStream_t stream)
{
    // [N, C, H, W] layout
    std::mdspan input_data(ctx.x_d, ctx.batches, ctx.channels, ctx.in_h, ctx.in_w);
    std::mdspan kernel_data(ctx.w_d, ctx.channels, ctx.kernel_h, ctx.kernel_w);
    std::mdspan output_data(ctx.y_d, ctx.batches, ctx.channels, ctx.out_h, ctx.out_w);
    launch_functor(stream, std::make_tuple(ctx.batches, ctx.channels, ctx.out_h, ctx.out_w),
        [=] __device__(int64_t batch, int64_t channel, int64_t oh, int64_t ow) {
            const int64_t kh_min = std::max(int64_t{ 0 }, (ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
            const int64_t kh_max = std::min(ctx.kernel_h, (ctx.in_h + ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
            const int64_t kw_min = std::max(int64_t{ 0 }, (ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);
            const int64_t kw_max = std::min(ctx.kernel_w, (ctx.in_w + ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);

            float accumulator = 0;
            for (int64_t kh = kh_min; kh < kh_max; kh++) {
                int64_t ih = calculate_input_coord(oh, kh, ctx.stride_h, ctx.dilation_h, ctx.padding_h);

                for (int64_t kw = kw_min; kw < kw_max; kw++) {
                    int64_t iw = calculate_input_coord(ow, kw, ctx.stride_w, ctx.dilation_w, ctx.padding_w);

                    const float input_val = input_data(batch, channel, ih, iw);
                    const float kernel_val = kernel_data(channel, kh, kw);

                    accumulator += input_val * kernel_val;
                }
            }

            output_data(batch, channel, oh, ow) = accumulator;
        }
    );
}

void conv2d_dw_nhwc(const conv2d_dw_context& ctx, cudaStream_t stream)
{
    // [N, H, W, C] layout
    std::mdspan input_data(ctx.x_d, ctx.batches, ctx.in_h, ctx.in_w, ctx.channels);
    std::mdspan kernel_data(ctx.w_d, ctx.kernel_h, ctx.kernel_w, ctx.channels);
    std::mdspan output_data(ctx.y_d, ctx.batches, ctx.out_h, ctx.out_w, ctx.channels);
    launch_functor(stream, std::make_tuple(ctx.batches, ctx.out_h, ctx.out_w, ctx.channels),
        [=] __device__(int64_t batch, int64_t oh, int64_t ow, int64_t channel) {
            const int64_t kh_min = std::max(int64_t{ 0 }, (ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
            const int64_t kh_max = std::min(ctx.kernel_h, (ctx.in_h + ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
            const int64_t kw_min = std::max(int64_t{ 0 }, (ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);
            const int64_t kw_max = std::min(ctx.kernel_w, (ctx.in_w + ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);

            float accumulator = 0;
            for (int64_t kh = kh_min; kh < kh_max; kh++) {
                int64_t ih = calculate_input_coord(oh, kh, ctx.stride_h, ctx.dilation_h, ctx.padding_h);

                for (int64_t kw = kw_min; kw < kw_max; kw++) {
                    int64_t iw = calculate_input_coord(ow, kw, ctx.stride_w, ctx.dilation_w, ctx.padding_w);

                    const float input_val = input_data(batch, ih, iw, channel);
                    const float kernel_val = kernel_data(kh, kw, channel);

                    accumulator += input_val * kernel_val;
                }
            }

            output_data(batch, oh, ow, channel) = accumulator;
        }
    );
}

void conv2d_dw_cuda(const conv2d_dw_context&  ctx, cudaStream_t stream)
{
    if (ctx.input_is_contiguous) {
        conv2d_dw_nchw(ctx, stream);
    }
    else if (ctx.input_is_contiguous_channels) {
        conv2d_dw_nhwc(ctx, stream);
    }
    else {
        GGML_ABORT("Unsupported memory layout for conv_2d_dw");
    }
}