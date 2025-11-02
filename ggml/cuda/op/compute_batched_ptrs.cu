#include "common.cuh"

static __global__ void k_compute_batched_ptrs(
    const void* src0_as_f16, const void* src1_as_f16, char* dst,
    const void** ptrs_src, void** ptrs_dst,
    int64_t ne12, int64_t ne13,
    int64_t ne23,
    size_t  nb02, size_t  nb03,
    size_t  nb12, size_t  nb13,
    size_t  nbd2, size_t  nbd3,
    int64_t r2, int64_t r3) {
    int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = (const char*)src0_as_f16 + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = (const char*)src1_as_f16 + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = (char*)dst + i12 * nbd2 + i13 * nbd3;
}

void k_compute_batched_ptrs_cuda(
    const void* src0_as_f16, const void* src1_as_f16, char* dst,
    const void** ptrs_src, void** ptrs_dst,
    int64_t ne12, int64_t ne13,
    int64_t ne23,
    size_t  nb02, size_t  nb03,
    size_t  nb12, size_t  nb13,
    size_t  nbd2, size_t  nbd3,
    int64_t r2, int64_t r3, cudaStream_t stream)
{
    const int threads_x = 16;
    const int threads_y = 16;
    dim3 block_dims(threads_x, threads_y);

    dim3 grid_dims(
        (ne13 + threads_x - 1) / threads_x,
        (ne12 + threads_y - 1) / threads_y
    );
    k_compute_batched_ptrs << <grid_dims, block_dims, 0, stream >> > (
        src0_as_f16, src1_as_f16, dst,
        ptrs_src, ptrs_dst,
        ne12, ne13,
        ne23,
        nb02, nb03,
        nb12,
        nb13,
        nbd2, nbd3,
        r2, r3);
    CUDA_CHECK(cudaGetLastError());
}