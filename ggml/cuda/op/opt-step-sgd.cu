static __global__ void opt_step_sgd_f32(
    float* __restrict__ x, const float* __restrict__ g,
    const float* __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

void opt_step_sgd_f32_cuda(
    float* x, const float* g, const float* pars, const int64_t k, cudaStream_t stream) {

    static constexpr size_t CUDA_OPT_STEP_SGD_BLOCK_SIZE = 256;

    const dim3 block_dims(CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_SGD_BLOCK_SIZE - 1) / CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    opt_step_sgd_f32 << <block_nums, block_dims, 0, stream >> > (x, g, pars, k);
}