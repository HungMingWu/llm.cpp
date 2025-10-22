#include <assert.h>
#include "common.cuh"
#define GGML_ASSERT(x) assert(x)

/*
    This kernel does the following:
    1. softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool with_norm>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(const float* logits,
    float* weights,
    int32_t* ids,
    const int     n_rows,
    const int     n_expert_used) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float logits_r[experts_per_thread];

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + threadIdx.x;
        logits_r[i / WARP_SIZE] = n_experts % WARP_SIZE == 0 || expert < n_experts ? logits[expert] : -INFINITY;
    }

    float max_val = logits_r[0];

#pragma unroll
    for (int i = 1; i < experts_per_thread; i++) {
        const float val = logits_r[i];
        max_val = max(val, max_val);
    }

    max_val = warp_reduce_max(max_val);

    float wt[experts_per_thread];
    float tmp = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const float val = logits_r[i];
        wt[i] = expf(val - max_val);
        tmp += wt[i];
    }

    tmp = warp_reduce_sum(tmp);

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = wt[i] * inv_sum;
    }

    //at this point, each thread holds a portion of softmax,
    //we do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

    for (int k = 0; k < n_expert_used; k++) {
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
        const float inv_sum = 1.0f / wt_sum;

        for (int i = threadIdx.x; i < n_expert_used; i += WARP_SIZE) {
            output_weights[i] *= inv_sum;
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i];
        }
    }
}

template <bool with_norm>
static void launch_topk_moe_cuda(cudaStream_t stream,
    const float* logits,
    float* weights,
    int32_t* ids,
    const int                   n_rows,
    const int                   n_expert,
    const int                   n_expert_used) {
    const int    rows_per_block = 4;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);

    switch (n_expert) {
    case 1:
        topk_moe_cuda<1, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 2:
        topk_moe_cuda<2, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 4:
        topk_moe_cuda<4, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 8:
        topk_moe_cuda<8, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 16:
        topk_moe_cuda<16, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 32:
        topk_moe_cuda<32, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 64:
        topk_moe_cuda<64, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 128:
        topk_moe_cuda<128, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 256:
        topk_moe_cuda<256, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    case 512:
        topk_moe_cuda<512, with_norm>
            << <grid_dims, block_dims, 0, stream >> > (logits, weights, ids, n_rows, n_expert_used);
        break;
    default:
        GGML_ASSERT(false && "fatal error");
        break;
    }
}

void launch_topk_moe_cuda(bool with_norm, const float* logits_d, float* weights_d, int32_t* ids_d,
    const int n_rows, const int n_experts, const int n_expert_used, cudaStream_t stream)
{
    if (with_norm) {
        launch_topk_moe_cuda<true>(stream, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used);
    }
    else {
        launch_topk_moe_cuda<false>(stream, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used);
    }
}
