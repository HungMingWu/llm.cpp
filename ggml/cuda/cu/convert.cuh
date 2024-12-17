#pragma once
#include "../vendors/cuda.h"

enum ggml_type : int;
void to_fp16_cuda(ggml_type type, const void* x, half* y, int64_t k, cudaStream_t stream);
void to_fp32_cuda(ggml_type type, const void* x, float* y, int64_t k, cudaStream_t stream);
