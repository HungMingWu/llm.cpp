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

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float(&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i] = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
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
template <int n_experts, bool has_bias>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(topk_moe_context ctx) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= ctx.n_rows) {
        return;
    }

    const topk_moe_config& config = ctx.config;
    const float* logits = ctx.logits + n_experts * row;
    float* weights = ctx.weights + ctx.n_expert_used * row;
    int32_t* ids = ctx.ids + ctx.n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
        else {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + ctx.bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
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

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val = wt[i];
                    max_val_s = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val = val;
                    max_val_s = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }
        else {
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

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum = warp_reduce_sum(wt_sum);
        wt_sum = max(wt_sum, ctx.clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, ctx.n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < ctx.n_expert_used) {
            weights[idx] = output_weights[i] * ctx.scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(const topk_moe_context& ctx, cudaStream_t stream) {
    GGML_ASSERT(!(ctx.config.with_norm && ctx.config.delayed_softmax) &&
        "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = 4;
    dim3         grid_dims((ctx.n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);

    switch (ctx.n_experts) {
    case 1:
        topk_moe_cuda<1, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 2:
        topk_moe_cuda<2, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 4:
        topk_moe_cuda<4, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 8:
        topk_moe_cuda<8, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 16:
        topk_moe_cuda<16, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 32:
        topk_moe_cuda<32, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 64:
        topk_moe_cuda<64, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 128:
        topk_moe_cuda<128, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 256:
        topk_moe_cuda<256, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 512:
        topk_moe_cuda<512, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    case 576:
        topk_moe_cuda<576, has_bias> << <grid_dims, block_dims, 0, stream >> > (ctx);
        break;
    default:
        GGML_ASSERT(false && "fatal error");
        break;
    }
}

void topk_moe_cuda(const topk_moe_context& ctx, cudaStream_t stream)
{
    if (ctx.has_bias) {
        launch_topk_moe_cuda<true>(ctx, stream);
    }
    else {
        launch_topk_moe_cuda<false>(ctx, stream);
    }
}
