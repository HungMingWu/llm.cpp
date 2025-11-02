#pragma once
#include "../vendors/cuda.h"

// moe-expert-reduce.cu
void moe_expert_reduce(
    const float* experts,
    const float* weights,
    float* dst,
    const int                   n_expert_used,
    const int                   n_cols,
    const int                   n_rows,
    cudaStream_t stream);

// softcap.cu
void softcap_f32_cuda(const float* x, float* dst, const float scale, const float softcap, const int k, cudaStream_t stream);

// topk-moe.cu
struct topk_moe_context {
    bool with_norm;
    const float* logits_d;
    float* weights_d;
    int32_t* ids_d;
    const int n_rows;
    const int n_experts;
    const int n_expert_used;
    const float clamp_val;
    const bool delayed_softmax;
};

void topk_moe_cuda(const topk_moe_context& ctx, cudaStream_t stream);