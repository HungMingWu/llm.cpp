#include <assert.h>
#include "../common.h"
#include "common.cuh"
#include "cuda_func.h"
#include "convert.cuh"
#include "unary.cuh"
#define GGML_ASSERT(x) assert(x)

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false>
static __global__ void mul_mat_vec_f(
    const T* __restrict__ x, const float* __restrict__ y, const int32_t* __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float* __restrict__ dst,
    const int ncols2, const int nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
    const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
    const int row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int channel_x = ids ? ids[channel_dst] : fastdiv((uint32_t)channel_dst, channel_ratio);
    const int channel_y = ids ? channel_dst % nchannels_y : channel_dst;
    const int sample_dst = blockIdx.z;
    const int sample_x = fastdiv((uint32_t)sample_dst, sample_ratio);
    const int sample_y = sample_dst;
    const int tid = threadIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    x += int64_t(sample_x) * stride_sample_x + channel_x * stride_channel_x + row * stride_row;
    y += int64_t(sample_y) * stride_sample_y + channel_y * stride_channel_y;
    dst += int64_t(sample_dst) * stride_sample_dst + channel_dst * stride_channel_dst;

    [[maybe_unused]] bool use_gate = false;
    [[maybe_unused]] bool use_bias = false;
    [[maybe_unused]] bool use_gate_bias = false;
    [[maybe_unused]] ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    [[maybe_unused]] const T* gate_x = nullptr;
    [[maybe_unused]] const float* x_bias = nullptr;
    [[maybe_unused]] const float* gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;

        if (use_gate) {
            gate_x = static_cast<const T*>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float*>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float*>(fusion.gate_bias);
            use_gate_bias = use_gate;
        }
        else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x) * stride_sample_x + channel_x * stride_channel_x + row * stride_row;
    }
    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst) * stride_sample_dst + channel_bias * stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst) * stride_sample_dst + channel_bias * stride_channel_dst;
        }
    }

    const float2* y2 = (const float2*)y;

    extern __shared__ char data_mmv[];
    float* buf_iw = (float*)data_mmv;
    float* buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float*)(data_mmv + warp_size * sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = { 0.0f };
    [[maybe_unused]] float sumf_gate[ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2* x2 = (const float2*)x;
        const float2* gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2*)gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    }
    else if constexpr (std::is_same_v<T, half>) {
        const half2* x2 = (const half2*)x;
        const half2* gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2*)gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j * stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        }
        else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = { {0.0f, 0.0f} };
            half2 sumh2_gate[ncols_dst] = { {0.0f, 0.0f} };

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j * stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        //TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int* x2 = (const int*)x;
        const int* gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int*)gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16*>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16*>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16*>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16*>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162* x2 = (const nv_bfloat162*)x;
        const nv_bfloat162* gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162*)gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    }
    else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid / warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid / warp_size] = sumf_gate[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid * stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid];
            if (use_gate_bias) {
                gate_value += gate_bias[tid * stride_col_dst + row];
            }
            switch (glu_op) {
            case GGML_GLU_OP_SWIGLU:
                value *= silu(gate_value);
                break;
            case GGML_GLU_OP_GEGLU:
                value *= gelu(gate_value);
                break;
            case GGML_GLU_OP_SWIGLU_OAI: {
                value = swiglu_oai(gate_value, value);
                break;
            }
            default:
                break;
            }
        }
    }

    dst[tid * stride_col_dst + row] = value;
}

template<typename T, typename type_acc, int ncols_dst, int block_size>
static void mul_mat_vec_f_switch_fusion(
    const T* x, const float* y, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const int64_t ncols, const int64_t nrows,
    const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
    const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
    const dim3& block_dims, const dim3& block_nums, const int nbytes_shared, const cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true> << <block_nums, block_dims, nbytes_shared, stream >> >
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_f<T, type_acc, ncols_dst, block_size> << <block_nums, block_dims, nbytes_shared, stream >> >
        (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);

}

template <typename T, typename type_acc, int ncols_dst>
void launch_mul_mat_vec_f_cuda(
    const T* x, const float* y, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const int64_t ncols, const int64_t nrows,
    const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
    const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
    const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
    cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(nsamples_dst % nsamples_x == 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd = init_fastdiv_values(nsamples_dst / nsamples_x);

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

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    const int nbytes_shared = warp_size * sizeof(float) + (has_fusion ? warp_size * sizeof(float) : 0);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_dst);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
    case   32: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case   64: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case   96: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case  128: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case  160: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case  192: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case  224: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    case  256: {
        mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256>
            (x, y, ids, fusion, dst, ncols / 2, nchannels_y, stride_row, stride_col_y / 2, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
    } break;
    default: {
        GGML_ABORT("fatal error");
    } break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
    const T* x, const float* y, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
    const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
    const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
    const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
    cudaStream_t stream) {
    switch (ncols_dst) {
    case 1:
        launch_mul_mat_vec_f_cuda<T, type_acc, 1>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 2:
        launch_mul_mat_vec_f_cuda<T, type_acc, 2>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 3:
        launch_mul_mat_vec_f_cuda<T, type_acc, 3>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 4:
        launch_mul_mat_vec_f_cuda<T, type_acc, 4>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 5:
        launch_mul_mat_vec_f_cuda<T, type_acc, 5>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 6:
        launch_mul_mat_vec_f_cuda<T, type_acc, 6>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 7:
        launch_mul_mat_vec_f_cuda<T, type_acc, 7>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
        break;
    case 8:
        launch_mul_mat_vec_f_cuda<T, type_acc, 8>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
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
    const T* x, const float* y, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
    const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
    const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
    const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
    enum ggml_prec prec, cudaStream_t stream) {

    if constexpr (std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                    nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                    stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
            stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
}

void mul_mat_vec_f_cuda(const mul_mat_vec_f_context* ctx, cudaStream_t stream)
{
    switch (ctx->src0_type) {
    case GGML_TYPE_F32: {
        const float* src0_d = (const float*)ctx->src0_d;
        mul_mat_vec_f_cuda(src0_d, ctx->src1_d, ctx->ids_d, ctx->fusion_local, ctx->dst_d, ctx->ne00, ctx->ne01,
            ctx->ncols_dst, ctx->s01, ctx->s11, ctx->s1,
            ctx->ne02, ctx->nchannels_y, ctx->nchannels_dst, ctx->s02,
            ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03, ctx->s13, ctx->s3, ctx->prec, stream);
    } break;
    case GGML_TYPE_F16: {
        const half* src0_d = (const half*)ctx->src0_d;
        mul_mat_vec_f_cuda(src0_d, ctx->src1_d, ctx->ids_d, ctx->fusion_local, ctx->dst_d, ctx->ne00, ctx->ne01,
            ctx->ncols_dst, ctx->s01, ctx->s11, ctx->s1,
            ctx->ne02, ctx->nchannels_y, ctx->nchannels_dst, ctx->s02,
            ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03, ctx->s13, ctx->s3, ctx->prec, stream);
    } break;
    case GGML_TYPE_BF16: {
        const nv_bfloat16* src0_d = (const nv_bfloat16*)ctx->src0_d;
        mul_mat_vec_f_cuda(src0_d, ctx->src1_d, ctx->ids_d, ctx->fusion_local, ctx->dst_d, ctx->ne00, ctx->ne01,
            ctx->ncols_dst, ctx->s01, ctx->s11, ctx->s1,
            ctx->ne02, ctx->nchannels_y, ctx->nchannels_dst, ctx->s02,
            ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03, ctx->s13, ctx->s3, ctx->prec, stream);
    } break;
    default:
        GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}