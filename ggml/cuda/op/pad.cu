#include "helper.h"
#include "cuda_func.h"
#include "launch.cuh"

void pad_f32_cuda(const pad_context &ctx, cudaStream_t stream) {
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i13, int64_t i12, int64_t i11, int64_t i10) {
            if ((i10 >= ctx.lp0 && i10 < ctx.dst_ne[0] - ctx.rp0) &&
                (i11 >= ctx.lp1 && i11 < ctx.dst_ne[1] - ctx.rp1) &&
                (i12 >= ctx.lp2 && i12 < ctx.dst_ne[2] - ctx.rp2) &&
                (i13 >= ctx.lp3 && i13 < ctx.dst_ne[3] - ctx.rp3)) {
                dst_data(i13, i12, i11, i10) = src0_data(i13 - ctx.lp3, i12 - ctx.lp2, i11 - ctx.lp1, i10 - ctx.lp0);
            }
            else {
                dst_data(i13, i12, i11, i10) = 0.0f;
            }
        }
    );
}