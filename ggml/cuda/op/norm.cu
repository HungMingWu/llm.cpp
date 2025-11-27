#include "common.cuh"
#include "reduce.cuh"
#include "cuda_func.h"
#include "helper.h"

static __global__ void norm_f32(
    auto x, auto dst, const int ncols, const float eps) {
    __shared__ float2 s_sum[32];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block.size()) {
        const float xi = x(sample, channel, row, col);
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = reduceWithBlock<cooperative_groups::plus>(block, tile, {}, mean_var, s_sum);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block.size()) {
        dst(sample, channel, row, col) = (x(sample, channel, row, col) - mean) * inv_std;
    }
}

void norm_f32_cuda(const norm_context& ctx, cudaStream_t stream)
{
    auto s_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
	auto [ncols, nrows, nchannels, nsamples] = ctx.src0_ne;
    const dim3 blocks_num(nrows, nchannels,  nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32 << <blocks_num, block_dims, 0, stream >> > 
            (s_data, dst_data, ncols, ctx.eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32 << <blocks_num, block_dims, 0, stream >> > 
            (s_data, dst_data, ncols, ctx.eps);
    }
}

template <typename dst_t, typename src_t, typename... Ts>
requires (std::is_same_v<Ts, src_t> && ...)
static __global__ void rms_norm_f32(dst_t dst, src_t x, const int ncols, const float eps, Ts... args) {
    static_assert(sizeof...(Ts) < 3, "rms_norm_f32 only supports up to 2 extra arguments");
    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;
    __shared__ float s_sum[32];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block.size()) {
        const float xi = x(sample, channel, row, col);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block.size()) {
        dst(sample, channel, row, col) = scale * x(sample, channel, row, col);
        if constexpr (sizeof...(Ts) > 0) {
            auto tuple = std::make_tuple(args...);
            auto mul_data = std::get<0>(tuple);
            dst(sample, channel, row, col) *=
                mul_data(sample % mul_data.extent(0), 
                    channel % mul_data.extent(1), row % mul_data.extent(2), col % mul_data.extent(3));
            if constexpr (sizeof...(Ts) > 1) {
                auto add_data = std::get<1>(tuple);
                dst(sample, channel, row, col) += 
                    add_data(sample % add_data.extent(0), channel % add_data.extent(1), row % add_data.extent(2), col % add_data.extent(3));
            }
        }
    }
}

static __global__ void rms_norm_back_f32(
    auto grad, auto xf, auto dst, const int ncols, const float eps) {
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    __shared__ float2 s_sum[32]; // xx and xg

    // x -> xx, sum for squares of x, equivalent to forward pass
    // y -> xg, sum for x * gradient, needed because RMS norm mixes inputs
    float2 sum_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block.size()) {
        sum_var.x += xf(row, col) * xf(row, col);
        sum_var.y += xf(row, col) * grad(row, col);
    }

    // sum up partial sums
    sum_var = reduceWithBlock<cooperative_groups::plus>(block, tile, {}, sum_var, s_sum);

    const float mean_eps = sum_var.x / ncols + eps;
    const float sum_eps = sum_var.x + ncols * eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x = -scale_grad * sum_var.y / sum_eps;

    for (int col = tid; col < ncols; col += block.size()) {
        dst(row, col) = scale_grad * grad(row, col) + scale_x * xf(row, col);
    }
}

void rms_norm_back_f32_cuda(const rms_norm_back_context& ctx, cudaStream_t stream) {
    std::experimental::mdspan grad(ctx.grad_d, ctx.nrows, ctx.ncols);
    std::experimental::mdspan xf(ctx.xf_d, ctx.nrows, ctx.ncols);
    std::experimental::mdspan dst(ctx.dst_d, ctx.nrows, ctx.ncols);
    if (ctx.ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32 << <ctx.nrows, block_dims, 0, stream >> > (grad, xf, dst, ctx.ncols, ctx.eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32 << <ctx.nrows, block_dims, 0, stream >> > (grad, xf, dst, ctx.ncols, ctx.eps);
    }
}

static __global__ void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    __shared__ float s_sum[32];
    const int start = blockIdx.x * group_size + threadIdx.x;
    const int end = min(blockIdx.x * group_size + group_size, ne_elements);

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block.size()) {
        tmp += x[j];
    }

    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block.size()) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, s_sum);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block.size()) {
        dst[j] *= scale;
    }
}

void group_norm_f32_cuda(
    const float* x, float* dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32 << <num_groups, block_dims, 0, stream >> > (x, dst, group_size, ne_elements, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32 << <num_groups, block_dims, 0, stream >> > (x, dst, group_size, ne_elements, eps);
    }
}

static __global__ void l2_norm_f32(
    auto x, auto dst, const int ncols,  const float eps) {
    __shared__ float s_sum[32];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block.size()) {
        const float xi = x(sample, channel, row, col);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, s_sum);

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block.size()) {
        dst(sample, channel, row, col) = scale * x(sample, channel, row, col);
    }
}

void l2_norm_f32_cuda(const norm_context& ctx, cudaStream_t stream)
{
    auto s_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);
    auto [ncols, nrows, nchannels, nsamples] = ctx.src0_ne;
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        l2_norm_f32 << <blocks_num, block_dims, 0, stream >> > (s_data, dst_data, ncols, ctx.eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        l2_norm_f32 << <blocks_num, block_dims, 0, stream >> > (s_data, dst_data, ncols, ctx.eps);
    }
}

void rms_norm_mul_f32_cuda(
    cudaStream_t  stream,
    const float eps,
    float* dst,
    std::array<int64_t, 4> dst_ne,
    std::array<size_t, 4> dst_nb,
    const float* x,
    std::array<int64_t, 4> x_ne,
    std::array<size_t, 4> x_nb,
    const float* mul,
    std::array<int64_t, 4> mul_ne,
    std::array<size_t, 4> mul_nb,
    const float* add,
    std::array<int64_t, 4> add_ne,
    std::array<size_t, 4> add_nb) {
    auto [ncols, nrows, nchannels, nsamples] = x_ne;
    const dim3 blocks_num(nrows, nchannels, nsamples);

    auto x_data = make_strided_mdspan(x, x_ne, x_nb);
    auto dst_data = make_strided_mdspan(dst, dst_ne, dst_nb);
    if (mul == nullptr) {
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32 << <blocks_num, block_dims, 0, stream >> > (dst_data, x_data, ncols, eps);
        }
        else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32 << <blocks_num, block_dims, 0, stream >> > (dst_data, x_data, ncols, eps);
        }
        return;
    }
    auto mul_data = make_strided_mdspan(mul, mul_ne, mul_nb);
    if (add == nullptr) {
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32 << <blocks_num, block_dims, 0, stream >> >(dst_data, x_data, ncols, eps, mul_data);
        }
        else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32 << <blocks_num, block_dims, 0, stream >> > (dst_data, x_data, ncols, eps, mul_data);
        }
    }
    else {
        auto add_data = make_strided_mdspan(add, add_ne, add_nb);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32 << <blocks_num, block_dims, 0, stream >> > (dst_data, x_data, ncols, eps, mul_data, add_data);
        }
        else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32 << <blocks_num, block_dims, 0, stream >> > (dst_data, x_data, ncols, eps, mul_data, add_data);
        }
    }
}