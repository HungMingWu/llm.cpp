static __global__ void clamp_f32(const float* x, float* dst, const float min, const float max, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

void clamp_f32_cuda(const float* x, float* dst, const float min, const float max, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_CLAMP_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    clamp_f32 << <num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream >> > (x, dst, min, max, k);
}