#pragma once
#include <algorithm>
#include <array>
#include <numeric>
#include "common.cuh"
#include "reduce.cuh"

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm>
static __global__ void reduce_rows_f32(const float* __restrict__ x, float* __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    static constexpr size_t num_unroll = 8;
    __shared__ float s_sum[32];
    std::array<float, num_unroll> temp;
    std::array<float, num_unroll> sum_temp{};
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[row * ncols + i];
            }
            else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        std::transform(temp.begin(), temp.end(), sum_temp.begin(), sum_temp.begin(), std::plus<float>());
    }
    float sum = std::reduce(sum_temp.begin(), sum_temp.end(), 0.0f, std::plus<float>());

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    // sum up partial sums
    sum = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, sum, s_sum);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}
