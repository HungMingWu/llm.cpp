
static __global__ void pad_reflect_1d_kernel_f32(
    const void* __restrict__ src0,
    void* __restrict__ dst,
    const int64_t ne0,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t ne02,
    const int64_t ne03,
    const int64_t nb00,
    const int64_t nb01,
    const int64_t nb02,
    const int64_t nb03,
    const int64_t nb0,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int p0,
    const int p1) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const char* src0_ptr = (const char*)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char* dst_ptr = (char*)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        float value;

        if (i0 < p0) {
            // Left padding - reflect
            value = *(const float*)(src0_ptr + (p0 - i0) * nb00);
        }
        else if (i0 < ne0 - p1) {
            // Middle - copy
            value = *(const float*)(src0_ptr + (i0 - p0) * nb00);
        }
        else {
            // Right padding - reflect
            int64_t src_idx = (ne0 - p1 - p0) - (p1 + 1 - (ne0 - i0)) - 1;
            value = *(const float*)(src0_ptr + src_idx * nb00);
        }

        *(float*)(dst_ptr + i0 * nb0) = value;
    }
}

void pad_reflect_1d_cuda(
    const void* src0, void* dst,
    const int64_t ne0,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3,
    const int p0, const int p1,cudaStream_t stream)
{
    static constexpr size_t CUDA_PAD_REFLECT_1D_BLOCK_SIZE = 256;

    const dim3 block_dims(CUDA_PAD_REFLECT_1D_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(ne01, ne02, ne03);

    pad_reflect_1d_kernel_f32 << <grid_dims, block_dims, 0, stream >> > (
        src0, dst,
        ne0, ne00, ne01, ne02, ne03,
        nb00, nb01, nb02, nb03,
        nb0, nb1, nb2, nb3,
        p0, p1
    );
}