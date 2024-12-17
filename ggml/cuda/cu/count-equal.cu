#include "cuda_func.h"
#include "common.cuh"
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

template <typename T>
static __global__ void count_equal(const T* __restrict__ x, const T* __restrict__ y, int64_t* __restrict__ dst, const int64_t dk, const int64_t k) {
    const int64_t i0 = (int64_t)blockIdx.x * dk;
    const int64_t i1 = min(i0 + dk, k);

    int nequal = 0;

    for (int64_t i = i0 + threadIdx.x; i < i1; i += WARP_SIZE) {
        const T xi = x[i];
        const T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum(nequal);

    if (threadIdx.x != 0) {
        return;
    }

    atomicAdd((int*)dst, nequal);
}

void count_equal_cuda(const count_equal_context* ctx, cudaStream_t stream)
{
    static constexpr size_t CUDA_COUNT_EQUAL_CHUNK_SIZE = 128;

	const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
	const int64_t dne = GGML_PAD((ctx->ne + 4 * nsm - 1) / (4 * nsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);

    CUDA_CHECK(cudaMemsetAsync(ctx->dst_d, 0, ctx->dst_size, stream));

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(std::min<int64_t>((int64_t)4 * nsm, (ctx->ne + CUDA_COUNT_EQUAL_CHUNK_SIZE - 1) / CUDA_COUNT_EQUAL_CHUNK_SIZE), 1, 1);

    count_equal << <blocks_num, blocks_dim, 0, stream >> > (ctx->src0_d, ctx->src1_d, ctx->dst_d, dne, ctx->ne);
}