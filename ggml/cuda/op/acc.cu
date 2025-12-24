#include "cuda_func.h"
#include "mdspan_helper.h"
#include "launch.cuh"

void acc_f32_cuda(const acc_context& ctx, cudaStream_t stream) {
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            dst_data(i3, i2, i1, i0) = src0_data(i3, i2, i1, i0);
            int64_t src_i3 = i3 - ctx.offset[3];
            int64_t src_i2 = i2 - ctx.offset[2];
            int64_t src_i1 = i1 - ctx.offset[1];
            int64_t src_i0 = i0 - ctx.offset[0];
            if (src_i3 >= 0 && src_i3 < src1_data.extent(0) &&
                src_i2 >= 0 && src_i2 < src1_data.extent(1) &&
                src_i1 >= 0 && src_i1 < src1_data.extent(2) &&
                src_i0 >= 0 && src_i0 < src1_data.extent(3))
                dst_data(i3, i2, i1, i0) += src1_data(src_i3, src_i2, src_i1, src_i0);
        }
    );
}