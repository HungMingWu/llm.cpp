#include "cuda_func.h"
#include "common.cuh"
#include <utility>

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
    const float* x, const T* mask, float* dst, const soft_max_params p) {
    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    //TODO: noncontigous inputs/outputs
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x += int64_t(rowx) * ncols;
    mask += (i11 * p.nb11 + i12 * p.nb12 + i13 * p.nb13) / sizeof(T) * (mask != nullptr);
    dst += int64_t(rowx) * ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

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

        const float val = x[col] * p.scale + (mask ? slope * t2f32(mask[col]) : 0.0f);

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

template<int... Ns, typename T>
static void launch_soft_max_kernels(const float* x, const T* mask, float* dst,
    const soft_max_params& p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT(reinterpret_cast<const void*>(soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, mask, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT(reinterpret_cast<const void*>(soft_max_f32<true, 0, 0, T>), smpbo);
    soft_max_f32<true, 0, 0> << <block_nums, block_dims, nbytes_shared, stream >> > (x, mask, dst, p);
}


template<typename T>
static void soft_max_f32_cuda(const float* x, const T* mask, float* dst, const soft_max_params& params, cudaStream_t stream) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth, 1, 1);
    const dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared = (GGML_PAD1(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;


    if (nbytes_shared <= smpbo) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, dst, params, stream, block_dims, block_nums, nbytes_shared);
    }
    else {
        const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);
        soft_max_f32<false, 0, 0> << <block_nums, block_dims, nbytes_shared_low, stream >> > (x, mask, dst, params);
    }
}

void soft_max_f32_cuda(const softmax_context* ctx, cudaStream_t stream)
{
    if (ctx->use_f16) {
        soft_max_f32_cuda(ctx->src0_d, (const half*)ctx->src1_d,
            ctx->dst_d, ctx->params, stream);
    }
    else {
        soft_max_f32_cuda(ctx->src0_d, (const float*)ctx->src1_d,
            ctx->dst_d, ctx->params, stream);
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
