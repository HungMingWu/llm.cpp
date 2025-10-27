#include "cuda_func.h"
#include "common.cuh"

void sum_f32_cuda(ggml_cuda_pool&, const float* x, float* dst, const int64_t ne, cudaStream_t stream) {
    // Use (inefficient) sum_rows implementation as a fallback.
    // For AMD there is rocPRIM which could be used as a drop-in replacement via hipcub but this would require C++11 -> C++14.
    sum_rows_f32_cuda(x, dst, ne, 1, stream);
}