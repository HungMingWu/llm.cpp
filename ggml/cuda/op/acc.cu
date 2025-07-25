static __global__ void acc_f32(const float* x, const float* y, float* dst, const int64_t ne,
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    const int64_t s11, const int64_t s12, const int64_t s13, const int64_t offset) {
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    const int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    const int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    const int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    const int64_t i10 = tmp;

    float val = x[i];
    if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += y[((i13 * ne12 + i12) * ne11 + i11) * ne10 + i10];
    }
    dst[i] = val;
}

void acc_f32_cuda(const float* x, const float* y, float* dst, const int64_t n_elements,
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset, cudaStream_t stream) {
    static constexpr size_t CUDA_ACC_BLOCK_SIZE = 256;
    const int num_blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    acc_f32 << <num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream >> > (x, y, dst, n_elements, ne10, ne11, ne12, ne13, s1, s2, s3, offset);
}