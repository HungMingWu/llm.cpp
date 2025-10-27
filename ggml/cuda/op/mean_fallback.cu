#include "cuda_func.h"

void mean_cuda(ggml_cuda_pool&, const float* src0_d, float* dst_d, const int64_t ncols, const int64_t nrows, cudaStream_t stream) {
    mean_fallback(src0_d, dst_d, ncols, nrows, stream);
}