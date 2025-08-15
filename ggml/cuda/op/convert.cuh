#pragma once
#include "../vendors/cuda.h"

enum ggml_type : int;
void to_fp16_cuda(ggml_type type, const void* x, half* y, int64_t k, cudaStream_t stream);
void to_fp32_cuda(ggml_type type, const void* x, float* y, int64_t k, cudaStream_t stream);
void to_bf16_cuda(ggml_type type, const void* x, nv_bfloat16* y, int64_t k, cudaStream_t stream);

// TODO more general support for non-contiguous inputs

void convert_to_nc_cuda(ggml_type type, const void* x, float* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);
void convert_to_nc_cuda(ggml_type type, const void* x, half* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);
void convert_to_nc_cuda(ggml_type type, const void* x, nv_bfloat16* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

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
    else {
        return float(x);
    }
}
