#include "cuda_func.h"
#include "reduce_rows.cuh"

void mean_fallback(const float* src0_d, float* dst_d, const int64_t ncols, const int64_t nrows, cudaStream_t stream) {
    const dim3 block_nums(nrows, 1, 1);

    const int id = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/true> << <block_nums, block_dims, 0, stream >> > (src0_d, dst_d, ncols);
    }
    else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/true> << <block_nums, block_dims, 0, stream >> > (src0_d, dst_d, ncols);
    }
}