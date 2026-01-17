#include "cuda_func.h"
#include "reduce_rows.cuh"

void mean_fallback(const mean_context &ctx, cudaStream_t stream) {
    const dim3 block_nums(ctx.nrows, 1, 1);

    const int id = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // Heuristic for block size selection to optimize occupancy.
    // See discussion in: https://github.com/ggml-org/llama.cpp/pull/15132
    if ((ctx.nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/true> << <block_nums, block_dims, 0, stream >> > (ctx.src0_d, ctx.dst_d, ctx.ncols);
    }
    else {
        const dim3 block_dims(ctx.ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/true> << <block_nums, block_dims, 0, stream >> > (ctx.src0_d, ctx.dst_d, ctx.ncols);
    }
}