#pragma once
#include "../vendors/cuda.h"

enum ggml_type : int;
void to_fp16_cuda(ggml_type type, const void* x, half* y, int64_t k, cudaStream_t stream);
void to_fp32_cuda(ggml_type type, const void* x, float* y, int64_t k, cudaStream_t stream);
void to_bf16_cuda(ggml_type type, const void* x, nv_bfloat16* y, int64_t k, cudaStream_t stream);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void* x, T* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);