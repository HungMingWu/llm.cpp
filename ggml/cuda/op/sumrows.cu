#include "cuda_func.h"
#include "common.cuh"

void sum_rows_f32_cuda(const float* x, float* dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    reduce_rows_f32</*norm*/false> << <block_nums, block_dims, 0, stream >> > (x, dst, ncols);
}