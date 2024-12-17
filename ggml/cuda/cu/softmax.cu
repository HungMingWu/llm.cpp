#include "cuda_func.h"
#include "common.cuh"

static constexpr size_t CUDA_SOFT_MAX_BLOCK_SIZE = 1024;
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float)val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(
    const float* x, const T* mask, float* dst, const int ncols_par, const int nrows_y,
    const float scale, const float max_bias, const float m0, const float m1, uint32_t n_head_log2) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    x += int64_t(rowx) * ncols;
    mask += int64_t(rowy) * ncols * (mask != nullptr);
    dst += int64_t(rowx) * ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(max_bias, rowx / nrows_y, n_head_log2, m0, m1);

    extern __shared__ float data_soft_max_f32[];
    float* buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float* vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col] * scale + (mask ? slope * t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template<typename T>
static void soft_max_f32_cuda(const float* x, const T* mask, float* dst, const int ncols_x, const int nrows_x, const int nrows_y, const float scale, const float max_bias, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth, 1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t nbytes_shared = (GGML_PAD1(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    const uint32_t n_head = nrows_x / nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    if (nbytes_shared < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        switch (ncols_x) {
        case 32:
            soft_max_f32<true, 32, 32> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 64:
            soft_max_f32<true, 64, 64> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 128:
            soft_max_f32<true, 128, 128> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 256:
            soft_max_f32<true, 256, 256> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 512:
            soft_max_f32<true, 512, 512> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 1024:
            soft_max_f32<true, 1024, 1024> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 2048:
            soft_max_f32<true, 2048, 1024> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        case 4096:
            soft_max_f32<true, 4096, 1024> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        default:
            soft_max_f32<true, 0, 0> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
            break;
        }
    }
    else {
        const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);
        soft_max_f32<false, 0, 0> << <block_nums, block_dims, nbytes_shared_low, stream >> > (x, mask, dst, ncols_x, nrows_y, scale, max_bias, m0, m1, n_head_log2);
    }
}

void soft_max_f32_cuda(const softmax_context* ctx, cudaStream_t stream)
{
    if (ctx->use_f16) {
        soft_max_f32_cuda(ctx->src0_d, (const half*)ctx->src1_d,
            ctx->dst_d, ctx->ne00, ctx->nrows_x, ctx->nrows_y, ctx->scale, ctx->max_bias, stream);
    }
    else {
        soft_max_f32_cuda(ctx->src0_d, (const float*)ctx->src1_d,
            ctx->dst_d, ctx->ne00, ctx->nrows_x, ctx->nrows_y, ctx->scale, ctx->max_bias, stream);
    }
}

static __global__ void soft_max_back_f32(
    const float* grad, const float* dstf, float* dst, const int ncols, const float scale) {
    const int tid = threadIdx.x;
    const int rowx = blockIdx.x;

    grad += int64_t(rowx) * ncols;
    dstf += int64_t(rowx) * ncols;
    dst += int64_t(rowx) * ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col] * grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

void soft_max_back_f32_cuda(
    const float* grad, const float* dstf, float* dst,
    const int ncols, const int nrows, const float scale, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);

    soft_max_back_f32 << <block_nums, block_dims, 0, stream >> > (grad, dstf, dst, ncols, scale);
}
