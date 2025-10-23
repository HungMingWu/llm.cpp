#include <assert.h>
#include <span>
#include "convert.cuh"
#include "cuda_func.h"
#include "internal_ds.h"
#include "launch.cuh"

static __device__ __forceinline__ float op_clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

template <class T>
void clamp_cuda(const T* x, T* dst, const T min, const T max, const size_t k, cudaStream_t stream) {
    std::span x_span{ x, k };
    std::span dst_span{ dst, k };
    launch_functor(stream, std::make_tuple(k),
        [=] __device__(int64_t i) {
            dst_span[i] = ggml_cuda_cast<T>(op_clamp((float)x_span[i], (float)min, (float)max));
        }
    );
}

void clamp_cuda(const clamp_context* ctx) {
    assert(ctx->src0_type == GGML_TYPE_F32 || ctx->src0_type == GGML_TYPE_F16);
    assert(ctx->dst_type == GGML_TYPE_F32 || ctx->dst_type == GGML_TYPE_F16);
    assert(ctx->src0_type == ctx->dst_type);

    if (ctx->src0_type == GGML_TYPE_F16) {
        clamp_cuda((const half*)ctx->src0_d, (half*)ctx->dst_d,
            (half)ctx->min, (half)ctx->max, ctx->nelements, ctx->stream);
    }
    else {
        clamp_cuda((const float*)ctx->src0_d, (float*)ctx->dst_d,
            (float)ctx->min, (float)ctx->max, ctx->nelements, ctx->stream);
    }
}