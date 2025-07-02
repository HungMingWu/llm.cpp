#include "common.cuh"

void mean_cuda(const float* src0_d, float* dst_d, const int64_t ncols, const int64_t nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    reduce_rows_f32</*norm*/ true> << <block_nums, block_dims, 0, stream >> > (src0_d, dst_d, ncols);
}