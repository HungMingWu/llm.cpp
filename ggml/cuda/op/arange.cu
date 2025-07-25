#include <span>
#define CUDA_ARANGE_BLOCK_SIZE 256

static __global__ void arange_f32(std::span<float> dst, const float start, const float step) {
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= dst.size()) {
        return;
    }
    dst[nidx] = start + step * nidx;
}

void arange_f32_cuda(float *dst, size_t dst_size, const float start, const float step, cudaStream_t stream) {
    int num_blocks = (dst_size + CUDA_ARANGE_BLOCK_SIZE - 1) / CUDA_ARANGE_BLOCK_SIZE;
    arange_f32 << <num_blocks, CUDA_ARANGE_BLOCK_SIZE, 0, stream >> > (std::span{dst, dst_size}, start, step);
}
