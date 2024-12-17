#pragma once
#include "../vendors/cuda.h"

template<typename T>
using to_t_cuda_t = void (*)(const void* __restrict__ x, T* __restrict__ y, int64_t k, cudaStream_t stream);

using to_fp32_cuda_t = to_t_cuda_t<float> ;
using to_fp16_cuda_t = to_t_cuda_t<half>;

enum ggml_type : int;
to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);
