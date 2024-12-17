#include "common.cuh"
#include "internal_ds.h"

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

template <typename type_acc, int block_size>
static __global__ void mul_mat_vec(
    const half* __restrict__ x, const float* __restrict__ y, float* __restrict__ dst, const int64_t ncols2, const int64_t stride_row,
    const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst) {
    const int64_t row = blockIdx.x;
    const int64_t channel = blockIdx.z;
    const int     tid = threadIdx.x;

    x += (channel / channel_ratio) * stride_channel_x + row * stride_row;
    y += channel * stride_channel_y;
    dst += channel * stride_channel_dst;

    const half2* x2 = (const half2*)x;
    const float2* y2 = (const float2*)y;

    extern __shared__ char data_mmv[];
    float* buf_iw = (float*)data_mmv;

    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf;

    if (std::is_same<type_acc, float>::value) {
        sumf = 0.0f;

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = __half22float2(x2[col2]);
            const float2 tmpy = y2[col2];
            sumf += tmpx.x * tmpy.x;
            sumf += tmpx.y * tmpy.y;
        }
    }
    else {
#ifdef FP16_AVAILABLE
        half2 sumh2 = make_half2(0.0f, 0.0f);

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmp = y2[col2];
            sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
        }

        sumf = __low2float(sumh2) + __high2float(sumh2);
#else
        NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
    }

    sumf = warp_reduce_sum(sumf);

    if (block_size > WARP_SIZE) {
        buf_iw[tid / WARP_SIZE] = sumf;
        __syncthreads();
        if (tid >= WARP_SIZE) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[row] = sumf;
}

template <typename type_acc>
static void launch_mul_mat_vec_cuda(
    const half* x, const float* y, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
    cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);
    GGML_ASSERT(nchannels_y % nchannels_x == 0);
    const int64_t channel_ratio = nchannels_y / nchannels_x;

    int64_t block_size_best = WARP_SIZE;
    int64_t niter_best = (ncols + 2 * WARP_SIZE - 1) / (2 * WARP_SIZE);
    for (int64_t block_size = 2 * WARP_SIZE; block_size <= 256; block_size += WARP_SIZE) {
        const int64_t niter = (ncols + 2 * block_size - 1) / (2 * block_size);
        if (niter < niter_best) {
            niter_best = niter;
            block_size_best = block_size;
        }
    }

    const int smem = WARP_SIZE * sizeof(float);
    const dim3 block_nums(nrows, 1, nchannels_y);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
    case   32: {
        mul_mat_vec<type_acc, 32> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case   64: {
        mul_mat_vec<type_acc, 64> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case   96: {
        mul_mat_vec<type_acc, 96> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case  128: {
        mul_mat_vec<type_acc, 128> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case  160: {
        mul_mat_vec<type_acc, 160> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case  192: {
        mul_mat_vec<type_acc, 192> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case  224: {
        mul_mat_vec<type_acc, 224> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    case  256: {
        mul_mat_vec<type_acc, 256> << <block_nums, block_dims, smem, stream >> >
            (x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst);
    } break;
    default: {
        GGML_ABORT("fatal error");
    } break;
    }
}

void mul_mat_vec_cuda(
    const half* x, const float* y, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
    enum ggml_prec prec, cudaStream_t stream) {
    switch (prec) {
    case GGML_PREC_DEFAULT: 
        launch_mul_mat_vec_cuda<half>(x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y,
            stride_channel_x, stride_channel_y, stride_channel_dst, stream);
        break;
    case GGML_PREC_F32:
        launch_mul_mat_vec_cuda<float>(x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y,
            stride_channel_x, stride_channel_y, stride_channel_dst, stream);
        break;
    }
}