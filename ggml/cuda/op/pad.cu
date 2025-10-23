#include "helper.h"
#include "cuda_func.h"
#include "launch.cuh"

void pad_f32_cuda(const pad_context &ctx, cudaStream_t stream) {
    std::array<int64_t, 4> dst_ne = { ctx.ne0, ctx.ne1, ctx.ne2, ctx.ne3 };
    std::array<size_t, 4> dst_nb = { ctx.nb0, ctx.nb1, ctx.nb2, ctx.nb3 };
    auto dst_data = make_strided_mdspan(ctx.dst_d, dst_ne, dst_nb);
    std::array<int64_t, 4> src0_ne = { ctx.ne00, ctx.ne01, ctx.ne02, ctx.ne03 };
    std::array<size_t, 4> src0_nb = { ctx.nb00, ctx.nb01, ctx.nb02, ctx.nb03 };
    auto src0_data = make_strided_mdspan(ctx.src0_d, src0_ne, src0_nb);
    launch_functor(stream, std::make_tuple(ctx.ne3, ctx.ne2, ctx.ne1, ctx.ne0),
        [=] __device__(int64_t i13, int64_t i12, int64_t i11, int64_t i10) {
            if ((i10 >= ctx.lp0 && i10 < ctx.ne0 - ctx.rp0) &&
                (i11 >= ctx.lp1 && i11 < ctx.ne1 - ctx.rp1) &&
                (i12 >= ctx.lp2 && i12 < ctx.ne2 - ctx.rp2) &&
                (i13 >= ctx.lp3 && i13 < ctx.ne3 - ctx.rp3)) {
                dst_data(i13, i12, i11, i10) = src0_data(i13 - ctx.lp3, i12 - ctx.lp2, i11 - ctx.lp1, i10 - ctx.lp0);
            }
            else {
                dst_data(i13, i12, i11, i10) = 0.0f;
            }
        }
    );
}