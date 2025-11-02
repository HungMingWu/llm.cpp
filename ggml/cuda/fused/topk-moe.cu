#include <assert.h>
#include "common.cuh"
#include "fused.h"
#define GGML_ASSERT(x) assert(x)

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float(&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i] = val;
            sum += val;
        }
        else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool with_norm, bool delayed_softmax = false>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(topk_moe_context ctx) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ctx.n_rows) {
        return;
    }

    const float *logits = ctx.logits_d + ctx.n_experts * row;
    float *weights = ctx.weights_d + ctx.n_expert_used * row;
    int32_t* ids = ctx.ids_d + ctx.n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if constexpr (!delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over ctx.n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < ctx.n_expert_used; k++) {
        float max_val = wt[0];
        int   max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val = val;
                max_expert = expert;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_expert / WARP_SIZE] = -INFINITY;

            ids[k] = max_expert;
            if constexpr (with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if constexpr (with_norm) {
        wt_sum = warp_reduce_sum(wt_sum);
        wt_sum = max(wt_sum, ctx.clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if constexpr (delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, ctx.n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < ctx.n_expert_used) {
            weights[idx] = output_weights[i];
        }
    }
}

template <bool with_norm, bool delayed_softmax = false>
static void launch_topk_moe_cuda(const topk_moe_context& ctx, cudaStream_t stream) {
    static_assert(!(with_norm && delayed_softmax), "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = 4;
    dim3         grid_dims((ctx.n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);

    switch (ctx.n_experts) {
    case 1:
        topk_moe_cuda<1, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 2:
        topk_moe_cuda<2, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 4:
        topk_moe_cuda<4, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 8:
        topk_moe_cuda<8, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 16:
        topk_moe_cuda<16, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 32:
        topk_moe_cuda<32, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 64:
        topk_moe_cuda<64, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 128:
        topk_moe_cuda<128, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 256:
        topk_moe_cuda<256, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 512:
        topk_moe_cuda<512, with_norm, delayed_softmax>
            << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    default:
        GGML_ASSERT(false && "fatal error");
        break;
    }
}

void topk_moe_cuda(const topk_moe_context& ctx, cudaStream_t stream)
{
    if (ctx.with_norm) {
        launch_topk_moe_cuda<true>(ctx, stream);
    }
    else {
        if (ctx.delayed_softmax) {
            launch_topk_moe_cuda<false, true>(ctx, stream);
        }
        else {
            launch_topk_moe_cuda<false, false>(ctx, stream);
        }
    }
}
