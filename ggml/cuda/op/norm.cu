#include "common.cuh"
template <int block_size>
static __global__ void norm_f32(
    const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float2 s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
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
        norm_f32<WARP_SIZE> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

template <int block_size, bool do_multiply = false>
static __global__ void rms_norm_f32(
    const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps, const float* mul = nullptr, const int64_t mul_stride_row = 0,
    const int64_t mul_stride_channel = 0, const int64_t mul_stride_sample = 0, const int mul_ncols = 0,
    const int mul_nrows = 0, const int mul_nchannels = 0, const int mul_nsamples = 0) {
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    if constexpr (do_multiply) {
        const int mul_row = row % mul_nrows;
        const int mul_channel = channel % mul_nchannels;
        const int mul_sample = sample % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply) {
            const int mul_col = col % mul_ncols;
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
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE, false> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024, false> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
    const float* grad, const float* xf, float* dst, const int ncols, const float eps) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row) * ncols;
    xf += int64_t(row) * ncols;
    dst += int64_t(row) * ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps = sum_xx + ncols * eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x = -scale_grad * sum_xg / sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad * grad[col] + scale_x * xf[col];
    }
}

void rms_norm_back_f32_cuda(const float* grad, const float* xf, float* dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE> << <nrows, block_dims, 0, stream >> > (grad, xf, dst, ncols, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024> << <nrows, block_dims, 0, stream >> > (grad, xf, dst, ncols, eps);
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start = blockIdx.x * group_size + threadIdx.x;
    const int end = min(blockIdx.x * group_size + group_size, ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

void group_norm_f32_cuda(
    const float* x, float* dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE> << <num_groups, block_dims, 0, stream >> > (x, dst, group_size, ne_elements, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024> << <num_groups, block_dims, 0, stream >> > (x, dst, group_size, ne_elements, eps);
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
    const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;

    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
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
        l2_norm_f32<WARP_SIZE> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        l2_norm_f32<1024> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void rms_norm_mul_f32_cuda(
    const float* x, const float* mul, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
    const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
    const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
    const float eps, cudaStream_t stream)
{
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE, true> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples);
    }
    else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024, true> << <blocks_num, block_dims, 0, stream >> > (x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples);
    }
}