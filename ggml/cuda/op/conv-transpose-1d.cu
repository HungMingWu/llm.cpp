#include "cuda_func.h"
#include "mdspan_helper.h"
#include "launch.cuh"
#include "convert.cuh"

template <typename T>
void conv_transpose_1d_f32_cuda(const conv_transpose_1d_context& ctx, cudaStream_t stream) {
    assert(ctx.src0_ne[2] == ctx.src1_ne[1]);
    assert(ctx.dst_ne[1] == ctx.src0_ne[1]);
    const int64_t CIn = ctx.src0_ne[2];
    const int64_t COut = ctx.src0_ne[1];
    const int64_t K = ctx.src0_ne[0];
    const int64_t L = ctx.src1_ne[0];
    const int64_t LOut = ctx.dst_ne[0];
    assert(LOut == (L - 1) * ctx.stride - 2 * ctx.padding + ctx.dilation * (K - 1) + 1);
    std::mdspan dst_data(ctx.dst_d, ctx.dst_ne[1], ctx.dst_ne[0]);
    std::mdspan src0_data(static_cast<const T*>(ctx.src0_d), ctx.src0_ne[2], ctx.src0_ne[1], ctx.src0_ne[0]);
    std::mdspan src1_data(ctx.src1_d, ctx.src1_ne[1], ctx.src1_ne[0]);
    launch_functor(stream, std::make_tuple(COut, LOut),
        [=] __device__(int64_t cout, int64_t lout) {
            float accumulator = 0;

            for (int64_t cin = 0; cin < CIn; cin++) {
                for (int64_t k = 0; k < K; k++) {
                    int64_t lin = lout + ctx.padding - k * ctx.dilation;
                    if (lin % ctx.stride != 0) continue;
                    lin /= ctx.stride;
                    if (lin <  0 || lin >= L) continue;
                    accumulator += ggml_cuda_cast<float>(src0_data(cin, cout, k)) * src1_data(cin, lin);
                }
            }
            dst_data(cout, lout) = accumulator;
        }
    );
}

void conv_transpose_1d_f32_cuda(const conv_transpose_1d_context& ctx, cudaStream_t stream) {

    if (ctx.src0_type == internal::GGML_TYPE_F16) {
        conv_transpose_1d_f32_cuda<half>(ctx, stream);
    }
    else {
        conv_transpose_1d_f32_cuda<float>(ctx, stream);
    }
}
