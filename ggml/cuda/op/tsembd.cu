static __global__ void timestep_embedding_f32(const float* timesteps, float* dst, const int nb1, const int dim, const int max_period) {
    // blockIDx.y: idx of timesteps->ne[0]
    // blockIDx.x: idx of ((dim + 1) / 2) / BLOCK_SIZE
    int i = blockIdx.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    float* embed_data = (float*)((char*)dst + i * nb1);

    int half = dim / 2;
    if (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    if (j >= half) {
        return;
    }

    float timestep = timesteps[i];
    float freq = (float)expf(-logf(max_period) * j / half);
    float arg = timestep * freq;
    embed_data[j] = cosf(arg);
    embed_data[j + half] = sinf(arg);
}

void timestep_embedding_f32_cuda(const float* x, float* dst, const int ne00, const int nb1,
    const int dim, const int max_period, cudaStream_t stream)
{
    static constexpr size_t CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE = 256;
    int half_ceil = (dim + 1) / 2;
    int num_blocks = (half_ceil + CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne00, 1);
    timestep_embedding_f32 << <gridDim, CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE, 0, stream >> >
        (x, dst, nb1, dim, max_period);
}