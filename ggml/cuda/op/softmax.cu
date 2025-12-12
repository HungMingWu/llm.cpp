#include <utility>
#include "cuda_func.h"
#include "common.cuh"
#include "convert.cuh"
#include "helper.h"
#include "reduce.cuh"

static constexpr size_t CUDA_SOFT_MAX_BLOCK_SIZE = 1024;
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

template <bool use_shared, typename src_t, typename mask_t, typename dst_t>
static __global__ void soft_max_f32(
    src_t x, mask_t mask, std::span<const float> sinks, dst_t dst, const soft_max_params p) {
    const int ncols = p.ncols;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    auto block = cooperative_groups::this_thread_block();
    const int tid = block.thread_rank();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.src1_ne[2];
    const int64_t i13 = i03 % p.src1_ne[3];

    const int block_size = blockDim.x;

    const int tile_id = tid / tile.size();
    const int lane_id = tile.thread_rank();

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_soft_max_f32[];
    float* buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float* vals = use_shared ? buf_iw + WARP_SIZE : &dst(i03, i02, i01, 0);

    float max_val = !sinks.empty() ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col = tid; col < ncols; col += block_size) {
        const float val = [&]() {
            float val = x(i03, i02, i01, col) * p.scale;
            if (!mask.empty()) {
                val += slope * ggml_cuda_cast<float>(mask(i13, i12, i11, col));
            }
            return val;
        }();

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = reduceWithBlock<cooperative_groups::greater>(block, tile, -INFINITY, max_val, buf_iw);

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col = tid; col < ncols; col += block_size) {
        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, buf_iw);

    if (!sinks.empty()) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col = tid; col < ncols; col += block_size) {
        dst(i03, i02, i01, col) = vals[col] * inv_sum;
    }
}

template <typename T>
static void soft_max_f32_cuda(const softmax_context& ctx, cudaStream_t stream) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = ctx.params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth, 1, 1);
    const dim3 block_nums(ctx.params.src0_ne[1], ctx.params.src0_ne[2], ctx.params.src0_ne[3]);
    const size_t nbytes_shared = (GGML_PAD1(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.params.dst_ne, ctx.params.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.params.src0_ne, ctx.params.src0_nb);
    auto mask = [&]() {
        using dst_t = decltype(make_strided_mdspan(static_cast<const T*>(ctx.src1_d), ctx.params.src1_ne, ctx.params.src1_nb));
        if (!ctx.src1_d) return dst_t{};
        else {
            return make_strided_mdspan(static_cast<const T*>(ctx.src1_d), ctx.params.src1_ne, ctx.params.src1_nb);
        }
    }();

    auto sinks = [&]() -> std::span<const float> {
        if (!ctx.src2_d) return {};
        else {
            assert(ctx.params.src2_ne[0] == ctx.params.src0_ne[2]);
            return { static_cast<const float*>(ctx.src2_d), static_cast<size_t>(ctx.params.src2_ne[0]) };
        }
    }();
    if (nbytes_shared <= smpbo) {
        soft_max_f32<true> << <block_nums, block_dims, nbytes_shared, stream >> > (src0_data,
            mask, sinks, dst_data, ctx.params);
    }
    else {
        const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);

        soft_max_f32<false> << <block_nums, block_dims, nbytes_shared_low, stream >> > (src0_data,
            mask, sinks, dst_data, ctx.params);
    }
}

void soft_max_f32_cuda(const softmax_context &ctx, cudaStream_t stream)
{
    if (ctx.use_f16) {
        soft_max_f32_cuda<half>(ctx, stream);
    }
    else {
        soft_max_f32_cuda<float>(ctx, stream);
    }
}

static __global__ void soft_max_back_f32(
    auto grad, auto dstf, auto dst, const int ncols, const float scale) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf(row, col) * grad(row, col);
    }

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    dgf_dot = cooperative_groups::reduce(tile, dgf_dot, cooperative_groups::plus<float>{});

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst(row, col) = scale * (grad(row, col) - dgf_dot) * dstf(row, col);
    }
}

void soft_max_back_f32_cuda(const softmax_back_context &ctx, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(ctx.nrows, 1, 1);

    std::experimental::mdspan grad(ctx.src0_d, ctx.nrows, ctx.ncols);
    std::experimental::mdspan dstf(ctx.src1_d, ctx.nrows, ctx.ncols);
    std::experimental::mdspan dst(ctx.dst_d, ctx.nrows, ctx.ncols);
    soft_max_back_f32 << <block_nums, block_dims, 0, stream >> > (grad, dstf, dst, ctx.ncols, ctx.scale);
}
