#include <assert.h>
#include <float.h>
#include "common.cuh"
#include "cuda_func.h"
#include "helper.h"
#include "launch.cuh"

void pool2d_nchw_kernel_cuda(const pool2d_context& ctx, cudaStream_t stream) {
    std::experimental::mdspan src0_data(ctx.src0_d, ctx.N, ctx.OC, ctx.IH, ctx.IW);
    std::experimental::mdspan dst_data(ctx.dst_d, ctx.N, ctx.OC, ctx.OH, ctx.OW);

    launch_functor(stream, std::make_tuple(ctx.N, ctx.OC, ctx.OH, ctx.OW),
        [=] __device__(int64_t n, int64_t c, int64_t oh, int64_t ow) {
            const int64_t start_h = oh * ctx.SH - ctx.PH;
            const int64_t bh = max(int64_t{ 0 }, start_h);
            const int64_t eh = min(ctx.IH, start_h + ctx.KH);
            const int64_t start_w = ow * ctx.SW - ctx.PW;
            const int64_t bw = max(int64_t{ 0 }, start_w);
            const int64_t ew = min(ctx.IW, start_w + ctx.KW);
            const float scale = 1. / (ctx.KH * ctx.KW);
            float res = 0;

            switch (ctx.op) {
                case internal::GGML_OP_POOL_AVG: res = 0; break;
                case internal::GGML_OP_POOL_MAX: res = -FLT_MAX; break;
                default: assert(false);
            }

            for (int64_t i = bh; i < eh; i += 1) {
                for (int64_t j = bw; j < ew; j += 1) {
#if __CUDA_ARCH__ >= 350
                    float cur = __ldg(&src0_data(n, c, i, j));
#else
                    float cur = src0_data(n, c, i, j);
#endif
                    switch (ctx.op) {
                        case internal::GGML_OP_POOL_AVG: res += cur * scale; break;
                        case internal::GGML_OP_POOL_MAX: res = max(res, cur); break;
                        default: assert(false);
                    }
                }
            }
            dst_data(n, c, oh, ow) = res;
        }
    );
}