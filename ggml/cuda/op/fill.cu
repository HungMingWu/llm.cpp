#include "cuda_func.h"
#include "convert.cuh"
#define CUDA_FILL_BLOCK_SIZE 256
#define GGML_ABORT(...)

template <typename T>
static __global__ void fill_kernel(T* dst, const int64_t k, const T value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

void fill_cuda(internal::ggml_type dst_type, void* dst_d, const int64_t k, float value, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

    switch (dst_type) {
    case internal::GGML_TYPE_F32:
        fill_kernel << <num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream >> > ((float*)dst_d, k, value);
        break;
    case internal::GGML_TYPE_F16:
        fill_kernel << <num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream >> > ((half*)dst_d, k, ggml_cuda_cast<half>(value));
        break;
    default:
        GGML_ABORT("unsupported type");
    }
}
