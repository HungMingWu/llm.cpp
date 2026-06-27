#include "cuda_func.h"

static __global__ void k_compute_out_prod_ptrs(
        const float * src0_d, const float * src1_d, float * dst_d,
        const float ** ptrs_a, const float ** ptrs_b, float ** ptrs_c,
        const int64_t ne2, const int64_t ne3,
        const int64_t dps2, const int64_t dps3,
        const size_t s02, const size_t s03,
        const size_t s12, const size_t s13,
        const size_t s2,  const size_t s3) {
    const int64_t i2 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i3 = blockIdx.y*blockDim.y + threadIdx.y;

    if (i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t idx = i3*ne2 + i2;

    ptrs_a[idx] = src0_d + (i3/dps3)*s03 + (i2/dps2)*s02;
    ptrs_b[idx] = src1_d +  i3      *s13 +  i2      *s12;
    ptrs_c[idx] = dst_d  +  i3      *s3  +  i2      *s2;
}

void k_compute_out_prod_ptrs(
        const float * src0_d, const float * src1_d, float * dst_d,
        const float ** ptrs_a, const float ** ptrs_b, float ** ptrs_c,
        const int64_t ne2, const int64_t ne3,
        const int64_t dps2, const int64_t dps3,
        const size_t s02, const size_t s03,
        const size_t s12, const size_t s13,
        const size_t s2,  const size_t s3, cudaStream_t stream)
{
    const dim3 block_dims(16, 16);
    const dim3 grid_dims((ne2 + block_dims.x - 1) / block_dims.x, (ne3 + block_dims.y - 1) / block_dims.y);
    k_compute_out_prod_ptrs << <grid_dims, block_dims, 0, stream >> > (
        src0_d, src1_d, dst_d,
        ptrs_a, ptrs_b, ptrs_c,
        ne2, ne3, dps2, dps3, s02, s03, s12, s13, s2, s3);
    CUDA_CHECK(cudaGetLastError());
}