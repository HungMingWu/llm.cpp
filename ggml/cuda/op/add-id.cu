#include <algorithm>
#include "cuda_func.h"
#include "helper.h"
#include "launch.cuh"

void add_id_cuda(const add_id_context* ctx, cudaStream_t stream)
{
    std::array<int64_t, 3> dst_ne = { ctx->ne0, ctx->ne1, ctx->ne2 };
    std::array<size_t, 3> dst_nb = { ctx->nb0, ctx->nb1, ctx->nb2 };
    auto dst_data = make_strided_mdspan<3>(ctx->dst_d, dst_ne, dst_nb);
    std::array<int64_t, 3> src0_ne = { ctx->ne00, ctx->ne01, ctx->ne02 };
    std::array<size_t, 3> src0_nb = { ctx->nb00, ctx->nb01, ctx->nb02 };
    auto src0_data = make_strided_mdspan<3>(ctx->src0_d, src0_ne, src0_nb);
    std::array<int64_t, 2> src1_ne = { ctx->ne10, ctx->ne11 };
    std::array<size_t, 2> src1_nb = { ctx->nb10, ctx->nb11 };
    auto src1_data = make_strided_mdspan<2>(ctx->src1_d, src1_ne, src1_nb);
    std::array<int64_t, 2> src2_ne = { ctx->ne20, ctx->ne21 };
	std::array<size_t, 2> src2_nb = { ctx->nb20, ctx->nb21 };
    auto src2_data = make_strided_mdspan<2>(ctx->src2_d, src2_ne, src2_nb);
    launch_functor(stream, std::make_tuple(ctx->ne2, ctx->ne1, ctx->ne0),
        [=] __device__ (int64_t i2, int64_t i1, int64_t i0) {
            const int i11 = src2_data(i2, i1);
            dst_data(i2, i1, i0) = src0_data(i2, i1, i0) + src1_data(i11, i0);
		}
    );
}