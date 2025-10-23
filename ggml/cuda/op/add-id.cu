#include <algorithm>
#include "cuda_func.h"
#include "helper.h"
#include "launch.cuh"

void add_id_cuda(const add_id_context &ctx, cudaStream_t stream)
{
    auto dst_data = make_strided_mdspan<3>(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan<3>(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<2>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto src2_data = make_strided_mdspan<2>(ctx.src2_d, ctx.src2_ne, ctx.src2_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__ (int64_t i2, int64_t i1, int64_t i0) {
            const int i11 = src2_data(i2, i1);
            dst_data(i2, i1, i0) = src0_data(i2, i1, i0) + src1_data(i11, i0);
		}
    );
}