#include "cuda_func.h"
#include "reduce_rows.cuh"

void sum_rows_f32_cuda(const float* x, float* dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/false> << <block_nums, block_dims, 0, stream >> > (x, dst, ncols);
    }
    else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/false> << <block_nums, block_dims, 0, stream >> > (x, dst, ncols);
    }
}