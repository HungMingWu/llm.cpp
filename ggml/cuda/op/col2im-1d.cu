#include "cuda_func.h"
#include "convert.cuh"
#include "common.cuh"
#include "launch.cuh"
#include "mdspan_helper.h"

// col2im_1d: scatter-add GEMM columns to 1D signal (gather approach)
// columns: [T_om, OC, K]  ->  output: [OC, T_out]
// Supports F32, F16, BF16 data with F32 accumulator.

template <typename T>
void col2im_1d_cuda(const col2im_1d_context& ctx, cudaStream_t stream) {
    const int K = ctx.K_OC / ctx.OC;
    const auto col_data = std::mdspan((const T*)ctx.src0_d, ctx.T_in, ctx.OC, K);
    auto dst_data = std::mdspan((T*)ctx.dst_d, ctx.OC, ctx.T_out);
    launch_functor(stream, std::make_tuple(ctx.OC, ctx.T_out),
        [=] __device__(int64_t oc, int64_t t_out) {
            const int t_abs = t_out + ctx.p0;  // absolute position in uncropped signal

            // Gather: find all (t_in, k) where t_in*s + k == t_abs, 0 <= k < K
            int t_in_min = (t_abs - K + ctx.s0) / ctx.s0;  // ceil((t_abs - K + 1) / s)
            if (t_in_min < 0) t_in_min = 0;
            int t_in_max = t_abs / ctx.s0;
            if (t_in_max >= ctx.T_in) t_in_max = ctx.T_in - 1;

            float sum = 0.0f;
            for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
                const int k = t_abs - t_in * ctx.s0;
                // col layout: [ctx.T_in, ctx.OC, ctx.K], column index = [t_in, oc, k]
                sum += ggml_cuda_cast<float>(col_data(t_in, oc, k));
            }
            // dst layout: [ctx.OC, T_out]
            dst_data(oc, t_out) = ggml_cuda_cast<T>(sum);
        }
    );
}

void col2im_1d_cuda(const col2im_1d_context& ctx, cudaStream_t stream) {
    switch (ctx.src0_type) {
    case internal::GGML_TYPE_F32: {
        col2im_1d_cuda<float>(ctx, stream);
    } break;
    case internal::GGML_TYPE_F16: {
        col2im_1d_cuda<half>(ctx, stream);
    } break;
    case internal::GGML_TYPE_BF16: {
        col2im_1d_cuda<nv_bfloat16>(ctx, stream);
    } break;
    default:
        GGML_ABORT("col2im_1d: unsupported type");
    }
}
