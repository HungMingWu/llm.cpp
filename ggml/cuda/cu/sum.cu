#include "cuda_func.h"
#include "common.cuh"

void sum_f32_cuda(ggml_cuda_pool& pool, const float* x, float* dst, const int64_t ne, cudaStream_t stream) {
#ifdef USE_CUB
    size_t tmp_size = 0;
    DeviceReduce::Sum(nullptr, tmp_size, x, dst, ne, stream);
    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
    DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);
#else
    // Use (inefficient) sum_rows implementation as a fallback.
    // For AMD there is rocPRIM which could be used as a drop-in replacement via hipcub but this would require C++11 -> C++14.
    sum_rows_f32_cuda(x, dst, ne, 1, stream);
    (void)(pool);
#endif // USE_CUB
}