#include "cuda_func.h"
#include "common.cuh"

template <bool use_shared>
static __global__ void cross_entropy_loss_f32(
    const float* __restrict__ logits, const float* __restrict__ labels, float* __restrict__ dst, const int nclasses, const int k) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x) * nclasses;
    labels += int64_t(blockIdx.x) * nclasses;

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        max_logit = fmaxf(max_logit, val);

        if constexpr (use_shared) {
            tmp[i] = val;
        }
    }
    max_logit = warp_reduce_max(max_logit);

    // Calculate log(softmax(logits)) which is just logits - max:
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        sum += expf(logit_i - max_logit);
    }
    sum = warp_reduce_sum(sum);
    sum = logf(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        loss += (logit_i - max_logit - sum) * labels[i];
    }
    loss = -warp_reduce_sum(loss) / (float)k;

    if (threadIdx.x != 0) {
        return;
    }

    dst[blockIdx.x] = loss;
}

void cross_entropy_loss_cuda(const cross_entropy_context* ctx, cudaStream_t stream)
{
    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(ctx->nrows, 1, 1);
    const size_t nbytes_shared = ctx->ne00 * sizeof(float);

    ggml_cuda_pool_alloc<float> dst_tmp(ctx->pool, blocks_num.x);
    if (nbytes_shared <= ctx->smpbo) {
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };
        if (!shared_memory_limit_raised[ctx->id]) {
            CUDA_CHECK(cudaFuncSetAttribute(cross_entropy_loss_f32<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, ctx->smpbo));
            shared_memory_limit_raised[ctx->id] = true;
        }
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && !defined(GGML_USE_MUSA)
        cross_entropy_loss_f32<true> << <blocks_num, blocks_dim, nbytes_shared, stream >> > 
            (ctx->src0_d, ctx->src1_d, dst_tmp.ptr, ctx->ne00, ctx->nrows);
    }
    else {
        cross_entropy_loss_f32<false> << <blocks_num, blocks_dim, 0, stream >> > 
            (ctx->src0_d, ctx->src1_d, dst_tmp.ptr, ctx->ne00, ctx->nrows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Combine results from individual blocks:
    sum_f32_cuda(ctx->pool, dst_tmp.ptr, ctx->dst_d, blocks_num.x, stream);
}

template <bool use_shared>
static __global__ void cross_entropy_loss_back_f32(
    const float* __restrict__ grad, const float* __restrict__ logits, const float* __restrict__ labels,
    float* __restrict__ dst, const int nclasses) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x) * nclasses;
    labels += int64_t(blockIdx.x) * nclasses;
    dst += int64_t(blockIdx.x) * nclasses;

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        maxval = fmaxf(maxval, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf((use_shared ? tmp[i] : logits[i]) - maxval);
        sum += val;

        if constexpr (use_shared) {
            tmp[i] = val;
        }
        else {
            dst[i] = val;
        }
    }
    sum = warp_reduce_sum(sum);
    const float sm_scale = 1.0f / sum;

    const float d_by_nrows = *grad / gridDim.x;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = use_shared ? tmp[i] : dst[i];
        dst[i] = (val * sm_scale - labels[i]) * d_by_nrows;
    }
}

void cross_entropy_loss_back_cuda(const cross_entropy_back_context* ctx, cudaStream_t stream)
{
    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(ctx->nrows, 1, 1);
    const size_t nbytes_shared = ctx->ne00 * sizeof(float);

    if (nbytes_shared <= ctx->smpbo) {
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };
        if (!shared_memory_limit_raised[ctx->id]) {
            CUDA_CHECK(cudaFuncSetAttribute(cross_entropy_loss_back_f32<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, ctx->smpbo));
            shared_memory_limit_raised[ctx->id] = true;
        }
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && !defined(GGML_USE_MUSA)
        cross_entropy_loss_back_f32<true> << <blocks_num, blocks_dim, nbytes_shared, stream >> >
            (ctx->grad_d, ctx->src0f_d, ctx->src1f_d, ctx->dst_d, ctx->ne00);
    }
    else {
        cross_entropy_loss_back_f32<false> << <blocks_num, blocks_dim, 0, stream >> >
            (ctx->grad_d, ctx->src0f_d, ctx->src1f_d, ctx->dst_d, ctx->ne00);
    }
}