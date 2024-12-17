static __global__ void acc_f32(const float* x, const float* y, float* dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    }
    else {
        dst[i] = x[i];
    }
}

void acc_f32_cuda(const float* x, const float* y, float* dst, const int n_elements,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, const int offset, cudaStream_t stream)
{
    static constexpr size_t CUDA_ACC_BLOCK_SIZE = 256;
    int num_blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    acc_f32 << <num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream >> > (x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset);
}