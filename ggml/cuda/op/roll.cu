#include "cuda_func.h"
#include "helper.h"
#include "launch.cuh"

static __forceinline__ __device__ int64_t wrap_index(const int64_t idx, const int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}

void roll_f32_cuda(const roll_context& ctx, cudaStream_t stream)
{
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);

    launch_functor(stream, std::make_tuple(ctx.src0_ne[3], ctx.src0_ne[2], ctx.src0_ne[1], ctx.src0_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            const int64_t d0 = wrap_index(i0 - ctx.s0, ctx.src0_ne[0]);
            const int64_t d1 = wrap_index(i1 - ctx.s1, ctx.src0_ne[1]);
            const int64_t d2 = wrap_index(i2 - ctx.s2, ctx.src0_ne[2]);
            const int64_t d3 = wrap_index(i3 - ctx.s3, ctx.src0_ne[3]);

            dst_data(i3, i2, i1, i0) = src0_data(d3, d2, d1, d0);
        }
    );
}