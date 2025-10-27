#include <cub/cub.cuh>
#include "cuda_func.h"
#include "common.cuh"
using namespace cub;

void sum_f32_cuda(ggml_cuda_pool& pool, const float* x, float* dst, const int64_t ne, cudaStream_t stream) {
    size_t tmp_size = 0;
    DeviceReduce::Sum(nullptr, tmp_size, x, dst, ne, stream);
    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
    DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);
}