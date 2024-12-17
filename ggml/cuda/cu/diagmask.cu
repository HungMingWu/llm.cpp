#include <float.h>

static __global__ void diag_mask_inf_f32(const float* x, float* dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    const int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int i = row * ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

void diag_mask_inf_f32_cuda(const float* x, float* dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    static constexpr size_t CUDA_DIAG_MASK_INF_BLOCK_SIZE = 32;
    const dim3 block_dims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(nrows_x, block_num_x, 1);
    diag_mask_inf_f32 << <block_nums, block_dims, 0, stream >> > (x, dst, ncols_x, rows_per_channel, n_past);
}