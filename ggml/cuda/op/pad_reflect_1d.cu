#include "cuda_func.h"
#include "common.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

void pad_reflect_1d_cuda(const pad_reflect_1d_context& ctx, cudaStream_t stream)
{
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);

    launch_functor(stream, std::make_tuple(ctx.src0_ne[3], ctx.src0_ne[2], ctx.src0_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            const int64_t rel_i0 = i0 - ctx.p0;  // relative i0 in src0
            int64_t src_idx;

            if (rel_i0 < 0) {
                // Left padding - reflect
                src_idx = -rel_i0;
            }
            else if (rel_i0 < ctx.src0_ne[0]) {
                // Middle - copy
                src_idx = rel_i0;
            }
            else {
                // Right padding - reflect
                src_idx = 2 * ctx.src0_ne[0] - 2 - rel_i0;
            }
            dst_data(i3, i2, i1, i0) = src0_data(i3, i2, i1, src_idx);
        }
    );
}