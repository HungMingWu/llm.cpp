#include <algorithm>
#include "common.cuh"
#include "cuda_func.h"

static __global__ void k_copy_src1_to_contiguous(const char* __restrict__ src1_original, char* __restrict__ src1_contiguous,
    int* __restrict__ cur_src1_row, mmid_row_mapping* __restrict__ row_mapping,
    const char* __restrict ids, int64_t i02, size_t ids_nb1, size_t ids_nb0,
    int64_t ne11, int64_t ne10,
    size_t nb11, size_t nb12) {
    int32_t iid1 = blockIdx.x;
    int32_t id = blockIdx.y;

    const int32_t row_id_i = *(const int32_t*)(ids + iid1 * ids_nb1 + id * ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    __shared__ int src1_row;
    if (threadIdx.x == 0) {
        src1_row = atomicAdd(cur_src1_row, 1);
        row_mapping[src1_row] = { id, iid1 };
    }
    __syncthreads();

    const float* src1_row_original = (const float*)(src1_original + i11 * nb11 + i12 * nb12);
    float* src1_row_contiguous = (float*)(src1_contiguous + src1_row * nb11);

    for (int i = threadIdx.x; i < ne10; i += blockDim.x) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

static __global__ void k_copy_dst_from_contiguous(char* __restrict__ dst_original, const char* __restrict__ dst_contiguous,
    const mmid_row_mapping* __restrict__ row_mapping,
    int64_t ne0,
    size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float* dst_row_contiguous = (const float*)(dst_contiguous + i * nb1);
    float* dst_row_original = (float*)(dst_original + i1 * nb1 + i2 * nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}


void k_copy_src1_to_contiguous_cuda(const copy_src1_to_contiguous_context* ctx, cudaStream_t stream)
{
    dim3 block_dims(std::min<uint32_t>(ctx->ne10, 768u));
    dim3 grid_dims(ctx->ids_ne1, ctx->n_ids);
    k_copy_src1_to_contiguous << <grid_dims, block_dims, 0, stream >> > (
        ctx->src1_original, ctx->src1_contiguous,
        ctx->dev_cur_src1_row, ctx->dev_row_mapping,
        ctx->ids_dev, ctx->i02, ctx->ids_nb1, ctx->ids_nb0,
        ctx->ne11, ctx->ne10,
        ctx->nb11, ctx->nb12);
    CUDA_CHECK(cudaGetLastError());
}

void k_copy_dst_from_contiguous_cuda(const k_copy_dst_from_contiguous_context* ctx, cudaStream_t stream)
{
    dim3 block_dims(std::min<uint32_t>(ctx->ne0, 768u));
    dim3 grid_dims(ctx->num_src1_rows);
    k_copy_dst_from_contiguous << <grid_dims, block_dims, 0, stream >> > (
        ctx->dst_original, ctx->dst_contiguous,
        ctx->dev_row_mapping,
        ctx->ne0,
        ctx->nb1, ctx->nb2);
    CUDA_CHECK(cudaGetLastError());
}