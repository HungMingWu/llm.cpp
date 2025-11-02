#include "common.cuh"

// This kernel is a fusion of the expert weight reduce, common in MoE models

template <int n_expert_used_template>
__global__ void moe_expert_reduce_cuda(const float* __restrict__ experts,
    const float* __restrict__ weights,
    float* __restrict__ dst,
    const int n_expert_used,
    const int n_cols) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_cols) {
        return;
    }

    experts += row * n_cols * n_expert_used;
    weights += row * n_expert_used;
    dst += row * n_cols;

    float acc = 0.f;
    if constexpr (n_expert_used_template == 0) {
        for (int expert = 0; expert < n_expert_used; ++expert) {
            ggml_cuda_mad(acc, experts[col], weights[expert]);
            experts += n_cols;
        }
        dst[col] = acc;
    }
    else {
#pragma unroll
        for (int i = 0; i < n_expert_used_template; ++i) {
            ggml_cuda_mad(acc, experts[col], weights[i]);
            experts += n_cols;
        }
        dst[col] = acc;
    }
}

void moe_expert_reduce(
    const float* experts,
    const float* weights,
    float* dst,
    const int                   n_expert_used,
    const int                   n_cols,
    const int                   n_rows,
    cudaStream_t stream) {
    const int block_size = 32;

    const int n_blocks_x = n_rows;
    const int n_blocks_y = (n_cols + block_size - 1) / block_size;

    dim3 block_dims(block_size);
    dim3 grid_dims(n_blocks_x, n_blocks_y);

    switch (n_expert_used) {
    case 1:
        moe_expert_reduce_cuda<1>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 2:
        moe_expert_reduce_cuda<2>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 4:
        moe_expert_reduce_cuda<4>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 6:
        moe_expert_reduce_cuda<6>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 8:
        moe_expert_reduce_cuda<8>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 16:
        moe_expert_reduce_cuda<16>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 32:
        moe_expert_reduce_cuda<32>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 64:
        moe_expert_reduce_cuda<64>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    case 128:
        moe_expert_reduce_cuda<128>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    default:
        moe_expert_reduce_cuda<0>
            << <grid_dims, block_dims, 0, stream >> > (experts, weights, dst, n_expert_used, n_cols);
        break;
    }
}