#include <cub/cub.cuh>
#include "cuda_func.h"
#include "internal_ds.h"
using namespace cub;

static __global__ void init_indices(int* indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

static __global__ void init_offsets(int* offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}

static void argsort_f32_i32_cuda_cub(ggml_cuda_pool& pool,
    const float* x,
    int* dst,
    const int        ncols,
    const int        nrows,
    ggml_sort_order  order,
    cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<int>   offsets_alloc(pool, nrows + 1);

    int* temp_indices = temp_indices_alloc.get();
    float* temp_keys = temp_keys_alloc.get();
    int* d_offsets = offsets_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices << <grid_size, block_size, 0, stream >> > (temp_indices, ncols, nrows);

    const dim3 offset_grid((nrows + block_size - 1) / block_size);
    init_offsets << <offset_grid, block_size, 0, stream >> > (d_offsets, ncols, nrows);

    cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    size_t temp_storage_bytes = 0;

    if (order == GGML_SORT_ORDER_ASC) {
        DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
            temp_indices, dst,                                  // values (indices)
            ncols * nrows, nrows,                            // num items, num segments
            d_offsets, d_offsets + 1, 0, sizeof(float) * 8,  // all bits
            stream);
    }
    else {
        DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices,
            dst, ncols * nrows, nrows, d_offsets, d_offsets + 1, 0,
            sizeof(float) * 8, stream);
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void* d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst,
            ncols * nrows, nrows, d_offsets, d_offsets + 1, 0, sizeof(float) * 8,
            stream);
    }
    else {
        DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
            temp_indices, dst, ncols * nrows, nrows, d_offsets, d_offsets + 1,
            0, sizeof(float) * 8, stream);
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda(ggml_cuda_pool& pool,
    const float* x, int* dst,
    const int ncols, const int nrows,
    ggml_sort_order order, cudaStream_t stream)
{
    const int    ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, x, dst, ncols, nrows, order, stream);
    }
    else {
        argsort_f32_i32_cuda_bitonic(x, dst, ncols, nrows, order, stream);
    }
}