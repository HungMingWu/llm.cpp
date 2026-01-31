#pragma once
#include "../vendors/cuda.h"

// softcap.cu
void softcap_f32_cuda(const float* x, float* dst, const float scale, const float softcap, const int k, cudaStream_t stream);

// topk-moe.cu

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

struct topk_moe_context {
    const bool has_bias;
    const float* logits;
    float* weights;
    float* bias;
    int32_t* ids;
    const int64_t n_rows;
    const int64_t n_experts;
    const int64_t n_expert_used;
    const float clamp_val;
    const float scale_val;
    topk_moe_config config;
};

void topk_moe_cuda(const topk_moe_context& ctx, cudaStream_t stream);