#include <float.h>
#include <algorithm>
#include "common.cuh"
#include "cuda_func.h"

static __global__ void argmax_f32(const float* __restrict__ x, int32_t* __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float* rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

void argmax_cuda(const argmax_context* ctx, cudaStream_t stream)
{
    const int64_t num_blocks = ctx->nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ctx->ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32 << <blocks_num, blocks_dim, 0, stream >> > (ctx->src0_d, ctx->dst_d, ctx->ne00);
}
