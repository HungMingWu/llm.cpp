#pragma once
#include "cuda_func.h"

struct convert_context {
    internal::ggml_type src_type;
    int64_t src_ne[4];
    size_t src_nb[4];
};

void convert_to_cuda(const convert_context &ctx, const void* x, float* y, cudaStream_t stream);
void convert_to_cuda(const convert_context &ctx, const void* x, half* y, cudaStream_t stream);
void convert_to_cuda(const convert_context &ctx, const void* x, nv_bfloat16* y, cudaStream_t stream);

template<typename dst_t, typename src_t>
__host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    }
    else if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    }
    else if constexpr (std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    }
    else if constexpr (std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    }
    else if constexpr (std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
        if constexpr (ggml_use_hip_v) {
            return __float22bfloat162_rn(x);
        } else {
            return { x.x, x.y };
        }
    }
    else if constexpr (std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    }
    else {
        return float(x);
    }
}
