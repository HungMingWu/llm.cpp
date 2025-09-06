static constexpr int64_t MAX_GRIDDIM_X = 0x7FFFFFFF;

static __global__ void scale_f32(const float* x, float* dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

void scale_f32_cuda(const float* x, float* dst, const float scale, 
    const float bias, const int64_t nelements, cudaStream_t stream) {
    static constexpr size_t CUDA_SCALE_BLOCK_SIZE = 256;
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32 << <std::min(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream >> > (x, dst, scale, bias, nelements);
}