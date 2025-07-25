static __global__ void scale_f32(const float* x, float* dst, const float scale, const float bias, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}

void scale_f32_cuda(const float* x, float* dst, const float scale, const float bias, const int k, cudaStream_t stream) {
	static constexpr size_t CUDA_SCALE_BLOCK_SIZE = 256;
	const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32 << <num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream >> > (x, dst, scale, bias, k);
}