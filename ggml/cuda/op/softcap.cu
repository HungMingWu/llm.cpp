static __global__ void softcap_f32(const float* x, float* dst, const float scale, const float softcap, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = tanhf(scale * x[i]) * softcap;
}

void softcap_f32_cuda(const float* x, float* dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_SOFTCAP_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    softcap_f32 << <num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream >> > (x, dst, scale, softcap, k);
}