#include <float.h>
#include <algorithm>
#include "common.cuh"
#include "cuda_func.h"
#include "reduce.cuh"

struct Pair {
    float val = std::numeric_limits<float>::min();
    int index = -1;
    __device__ auto operator<=>(const Pair&) const = default;
};

static __global__ void argmax_f32(const float* __restrict__ x, int32_t* __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    Pair maxValue;
    const float* rowx = x + row * ncols;

    constexpr int max_warps = 1024 / WARP_SIZE;
    __shared__ Pair shared[max_warps];

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        Pair value{ rowx[col] , col };
        if (value > maxValue) {
            maxValue = value;
        }
    }

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    const int tid = block.thread_rank();
    const int tile_id = tid / tile.size();
    const int lane_id = tile.thread_rank();
    maxValue = reduceWithBlock<cooperative_groups::greater>(block, tile, maxValue, maxValue, shared);

    if (tile_id == 0 && lane_id == 0) {
        dst[row] = maxValue.index;
    }
}

void argmax_cuda(const argmax_context &ctx, cudaStream_t stream)
{
    const int64_t num_blocks = ctx.nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ctx.ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32 << <blocks_num, blocks_dim, 0, stream >> > (ctx.src0_d, ctx.dst_d, ctx.ne00);
}

