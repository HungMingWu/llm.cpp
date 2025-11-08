#include "common.cuh"
#include "reduce.cuh"

static __global__ void norm_f32(
    const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps) {
    __shared__ float2 s_sum[32];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block.size()) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = reduceWithBlock<cooperative_groups::plus>(block, tile, {}, mean_var, s_sum);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block.size()) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

void norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream)
{
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32 << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32 << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

template <bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float* x,
    float* dst,
    const int     ncols,
    const int64_t stride_row,
    const int64_t stride_channel,
    const int64_t stride_sample,
    const float   eps,
    const float* mul = nullptr,
    const int64_t mul_stride_row = 0,
    const int64_t mul_stride_channel = 0,
    const int64_t mul_stride_sample = 0,
    const uint3   mul_ncols_packed = make_uint3(0, 0, 0),
    const uint3   mul_nrows_packed = make_uint3(0, 0, 0),
    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
    const uint3   mul_nsamples_packed = make_uint3(0, 0, 0),
    const float* add = nullptr,
    const int64_t add_stride_row = 0,
    const int64_t add_stride_channel = 0,
    const int64_t add_stride_sample = 0,
    const uint3   add_ncols_packed = make_uint3(0, 0, 0),
    const uint3   add_nrows_packed = make_uint3(0, 0, 0),
    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
    const uint3   add_nsamples_packed = make_uint3(0, 0, 0)) {
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;
    __shared__ float s_sum[32];

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block.size()) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block.size()) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col] = scale * x[col] * mul[mul_col] + add[add_col];
        }
        else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col] = scale * x[col] * mul[mul_col];
        }
        else {
            dst[col] = scale * x[col];
        }
    }
}

void rms_norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        rms_norm_f32<false> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<false> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static __global__ void rms_norm_back_f32(
    const float* grad, const float* xf, float* dst, const int ncols, const float eps) {
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    __shared__ float2 s_sum[32]; // xx and xg
    grad += int64_t(row) * ncols;
    xf += int64_t(row) * ncols;
    dst += int64_t(row) * ncols;

    // x -> xx, sum for squares of x, equivalent to forward pass
    // y -> xg, sum for x * gradient, needed because RMS norm mixes inputs
    float2 sum_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block.size()) {
        const float xfi = xf[col];
        sum_var.x += xfi * xfi;
        sum_var.y += xfi * grad[col];
    }

    // sum up partial sums
    sum_var = reduceWithBlock<cooperative_groups::plus>(block, tile, {}, sum_var, s_sum);

    const float mean_eps = sum_var.x / ncols + eps;
    const float sum_eps = sum_var.x + ncols * eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x = -scale_grad * sum_var.y / sum_eps;

    for (int col = tid; col < ncols; col += block.size()) {
        dst[col] = scale_grad * grad[col] + scale_x * xf[col];
    }
}

void rms_norm_back_f32_cuda(const float* grad, const float* xf, float* dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32 << <nrows, block_dims, 0, stream >> > (grad, xf, dst, ncols, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32 << <nrows, block_dims, 0, stream >> > (grad, xf, dst, ncols, eps);
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
    const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps) {
    __shared__ float s_sum[32];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block.size()) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, s_sum);

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block.size()) {
        dst[col] = scale * x[col];
    }
}

void l2_norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps, cudaStream_t stream)
{
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        l2_norm_f32 << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        l2_norm_f32 << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void rms_norm_mul_f32_cuda(const float* x,
    const float* mul,
    const float* add,
    float* dst,
    const int      ncols,
    const int      nrows,
    const int      nchannels,
    const int      nsamples,
    const int64_t  stride_row,
    const int64_t  stride_channel,
    const int64_t  stride_sample,
    const int64_t  mul_stride_row,
    const int64_t  mul_stride_channel,
    const int64_t  mul_stride_sample,
    const uint32_t mul_ncols,
    const uint32_t mul_nrows,
    const uint32_t mul_nchannels,
    const uint32_t mul_nsamples,
    const int64_t  add_stride_row,
    const int64_t  add_stride_channel,
    const int64_t  add_stride_sample,
    const uint32_t add_ncols,
    const uint32_t add_nrows,
    const uint32_t add_nchannels,
    const uint32_t add_nsamples,
    const float    eps,
    cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32<true> << <blocks_num, block_dims, 0, stream >> > (
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        }
        else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32<true> << <blocks_num, block_dims, 0, stream >> > (
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        }
    }
    else {
        const uint3 mul_ncols_packed = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32<true, true> << <blocks_num, block_dims, 0, stream >> > (
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
        else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32<true, true> << <blocks_num, block_dims, 0, stream >> > (
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}