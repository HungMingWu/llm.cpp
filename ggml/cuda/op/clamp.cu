#include <assert.h>
#include "cuda_func.h"
#include "internal_ds.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)

static __device__ __forceinline__ float op_clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

template <class T>
static __global__ void op_clamp_kernel(const T* x, T* dst, const T min, const T max, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_clamp((float)x[i], (float)min, (float)max);
}

template <class T>
static void clamp_cuda(const T* x, T* dst, const T min, const T max, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_CLAMP_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    op_clamp_kernel << <num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream >> > (x, dst, min, max, k);
}

void clamp_cuda(const clamp_context* ctx) {
    GGML_ASSERT(ctx->src0_type == GGML_TYPE_F32 || ctx->src0_type == GGML_TYPE_F16);
    GGML_ASSERT(ctx->dst_type == GGML_TYPE_F32 || ctx->dst_type == GGML_TYPE_F16);
    GGML_ASSERT(ctx->src0_type == ctx->dst_type);

    if (ctx->src0_type == GGML_TYPE_F16) {
        clamp_cuda((const half*)ctx->src0_d, (half*)ctx->dst_d,
            (half)ctx->min, (half)ctx->max, ctx->nelements, ctx->stream);
    }
    else {
        clamp_cuda((const float*)ctx->src0_d, (float*)ctx->dst_d,
            (float)ctx->min, (float)ctx->max, ctx->nelements, ctx->stream);
    }
}