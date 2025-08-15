#include <assert.h>
#include "../common.h"
#include "common.cuh"
#include "cuda_func.h"
#include "convert.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size>
static __global__ void mul_mat_vec_f(
    const T* __restrict__ x, const float* __restrict__ y, const int32_t* __restrict__ ids, float* __restrict__ dst,
    const int ncols2, const int nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
    const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
    const int row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int channel_x = ids ? ids[channel_dst] : channel_dst / channel_ratio;
    const int channel_y = ids ? channel_dst % nchannels_y : channel_dst;
    const int sample_dst = blockIdx.z;
    const int sample_x = sample_dst / sample_ratio;
    const int sample_y = sample_dst;
    const int tid = threadIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    x += int64_t(sample_x) * stride_sample_x + channel_x * stride_channel_x + row * stride_row;
    y += int64_t(sample_y) * stride_sample_y + channel_y * stride_channel_y;
    dst += int64_t(sample_dst) * stride_sample_dst + channel_dst * stride_channel_dst;

    const float2* y2 = (const float2*)y;

    extern __shared__ char data_mmv[];
    float* buf_iw = (float*)data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = { 0.0f };

    if constexpr (std::is_same_v<T, float>) {
        const float2* x2 = (const float2*)x;

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                sumf[j] += tmpx.x * tmpy.x;
                sumf[j] += tmpx.y * tmpy.y;
            }
        }
    }
    else if constexpr (std::is_same_v<T, half>) {
        const half2* x2 = (const half2*)x;

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);

#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j * stride_col_y2 + col2];
                    sumf[j] += tmpx.x * tmpy.x;
                    sumf[j] += tmpx.y * tmpy.y;
                }
            }
        }
        else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = { {0.0f, 0.0f} };

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];

#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j * stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        const int* x2 = (const int*)x;
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                sumf[j] += ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16*>(&tmpx)[0]) * tmpy.x;
                sumf[j] += ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16*>(&tmpx)[1]) * tmpy.y;
            }
        }
    }
    else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if (block_size > warp_size) {
            buf_iw[tid / warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }
            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    dst[tid * stride_col_dst + row] = sumf[tid];
}

template <typename T, typename type_acc, int ncols_dst>
static void launch_mul_mat_vec_f_cuda(
    const T* x, const float* y, const int32_t* ids, float* dst,
    const int64_t ncols, const int64_t nrows,
    const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
    const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
    const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
    cudaStream_t stream) {
    assert(ncols % 2 == 0);
    assert(stride_row % 2 == 0);
    assert(stride_col_y % 2 == 0);
    assert(ids || nchannels_dst % nchannels_x == 0);
    assert(nsamples_dst % nsamples_x == 0);
    const int64_t channel_ratio = nchannels_dst / nchannels_x;
    const int64_t sample_ratio = nsamples_dst / nsamples_x;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best = (ncols + 2 * warp_size - 1) / (2 * warp_size);
    int64_t max_block_size = 256;
    if (ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2 * warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2 * block_size - 1) / (2 * block_size);
        if (niter < niter_best) {
            niter_best = niter;
            block_size_best = block_size;
        }
    }

    const int nbytes_shared = warp_size * sizeof(float);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_dst);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
    case   32: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 32> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case   64: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 64> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case   96: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 96> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case  128: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 128> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case  160: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 160> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case  192: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 192> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case  224: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 224> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    case  256: {
        mul_mat_vec_f<T, type_acc, ncols_dst, 256> << <block_nums, block_dims, nbytes_shared, stream >> >
            (x, y, ids, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } break;
    default: {
        GGML_ABORT("fatal error");
    } break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
    const T* x, const float* y, const int32_t* ids, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
    const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
    const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
    const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
    cudaStream_t stream) {
    switch (ncols_dst) {
    case 1:
        launch_mul_mat_vec_f_cuda<T, type_acc, 1>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 2:
        launch_mul_mat_vec_f_cuda<T, type_acc, 2>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 3:
        launch_mul_mat_vec_f_cuda<T, type_acc, 3>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 4:
        launch_mul_mat_vec_f_cuda<T, type_acc, 4>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 5:
        launch_mul_mat_vec_f_cuda<T, type_acc, 5>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 6:
        launch_mul_mat_vec_f_cuda<T, type_acc, 6>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 7:
        launch_mul_mat_vec_f_cuda<T, type_acc, 7>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 8:
        launch_mul_mat_vec_f_cuda<T, type_acc, 8>
            (x, y, ids, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
    const T* x, const float* y, const int32_t* ids, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
    const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
    const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
    const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
    enum ggml_prec prec, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                    nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                    stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
            stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t* src0_ne, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }
    switch (type) {
    case GGML_TYPE_F32:
        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            if (ampere_mma_available(cc)) {
                return ne11 <= 3;
            }
            if (cc >= GGML_CUDA_CC_TURING) {
                return ne11 <= 4;
            }
            return ne11 <= 3;
        }
        else if (GGML_CUDA_CC_IS_AMD(cc)) {
            if (fp32_mma_hardware_available(cc)) {
                return ne11 <= 3;
            }
            return ne11 <= 8;
        }
        return ne11 <= 8;
    case GGML_TYPE_F16:
        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
            if (ampere_mma_available(cc)) {
                return src0_small && ne11 == 1;
            }
            if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return src0_small && ne11 <= 4;
            }
            if (fp16_mma_hardware_available(cc)) {
                return src0_small && ne11 <= 3;
            }
            return ne11 <= 8;
        }
        else if (GGML_CUDA_CC_IS_AMD(cc)) {
            if (fp16_mma_hardware_available(cc)) {
                if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                    return ne11 <= 5;
                }
                return ne11 <= 2;
            }
            return ne11 <= 8;
        }
        return ne11 <= 8;
    case GGML_TYPE_BF16:
        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
            if (ampere_mma_available(cc)) {
                return src0_small && ne11 == 1;
            }
            if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return src0_small && ne11 <= 4;
            }
            if (bf16_mma_hardware_available(cc)) {
                return src0_small && ne11 <= 3;
            }
            return ne11 <= 8;
        }
        else if (GGML_CUDA_CC_IS_AMD(cc)) {
            if (bf16_mma_hardware_available(cc)) {
                return ne11 <= 3;
            }
            return ne11 <= 8;
        }
        return ne11 <= 8;
    default:
        return false;
    }
}

void mul_mat_vec_f_cuda(const mul_mat_vec_f_context* ctx, cudaStream_t stream)
{
    switch (ctx->src0_type) {
    case GGML_TYPE_F32: {
        const float* src0_d = (const float*)ctx->src0_d;
        mul_mat_vec_f_cuda(src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00, ctx->ne01,
            ctx->ncols_dst, ctx->s01, ctx->s11, ctx->s1,
            ctx->ne02, ctx->nchannels_y, ctx->nchannels_dst, ctx->s02,
            ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03, ctx->s13, ctx->s3, ctx->prec, stream);
    } break;
    case GGML_TYPE_F16: {
        const half* src0_d = (const half*)ctx->src0_d;
        mul_mat_vec_f_cuda(src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00, ctx->ne01,
            ctx->ncols_dst, ctx->s01, ctx->s11, ctx->s1,
            ctx->ne02, ctx->nchannels_y, ctx->nchannels_dst, ctx->s02,
            ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03, ctx->s13, ctx->s3, ctx->prec, stream);
    } break;
    case GGML_TYPE_BF16: {
        const nv_bfloat16* src0_d = (const nv_bfloat16*)ctx->src0_d;
        mul_mat_vec_f_cuda(src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00, ctx->ne01,
            ctx->ncols_dst, ctx->s01, ctx->s11, ctx->s1,
            ctx->ne02, ctx->nchannels_y, ctx->nchannels_dst, ctx->s02,
            ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03, ctx->s13, ctx->s3, ctx->prec, stream);
    } break;
    default:
        GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}