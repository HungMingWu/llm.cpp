#include <algorithm>
#include "internal_ds.h"
#include "common.cuh"
#define GGML_ASSERT(...)
#define GGML_ABORT(...)

template <ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float* x, int* dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float* x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                    x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                        ) {
                        std::swap(dst_row[col], dst_row[ixj]);
                    }
                }
                else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                        ) {
                        std::swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda(
    const float* x, int* dst,
    const int ncols, const int nrows,
    ggml_sort_order order, cudaStream_t stream)
{
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC> << <block_nums, block_dims, shared_mem, stream >> > (x, dst, ncols, ncols_pad);
    }
    else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC> << <block_nums, block_dims, shared_mem, stream >> > (x, dst, ncols, ncols_pad);
    }
    else {
        GGML_ABORT("fatal error");
    }
}