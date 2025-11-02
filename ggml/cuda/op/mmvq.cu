#include "../common.h"
#include "internal_ds.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include "cuda_func.h"
#include "unary.cuh"
#include "../vendor_constant.h"

#define GGML_ASSERT(...)
#define GGML_ABORT(...)
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA2) || defined(RDNA3) || defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
        case 1:
            return 1;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
            return 2;
        default:
            return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
        case 1:
        case 2:
        case 3:
        case 4:
            return 4;
        case 5:
        case 6:
        case 7:
        case 8:
            return 2;
        default:
            return 1;
        }
    }
    else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
        case 1:
        case 2:
        case 3:
        case 4:
            return 2;
        case 5:
        case 6:
        case 7:
        case 8:
        default:
            return 1;
        }
    }
    return 1;
}

static std::pair<dim3, dim3> calc_launch_params(
    const int ncols_dst, const int nrows_x, const int nchannels_y, const int nsamples_y,
    const int warp_size, const mmvq_parameter_table_id table_id) {
    const int64_t nblocks = (nrows_x + calc_rows_per_block(ncols_dst, table_id) - 1) / calc_rows_per_block(ncols_dst, table_id);
    const dim3 block_nums(nblocks, nchannels_y, nsamples_y);
    const dim3 block_dims(warp_size, calc_nwarps(ncols_dst, table_id), 1);
    return { block_nums, block_dims };
}

// tell the compiler to use as many registers as it wants, see nwarps definition below
template <ggml_type type, typename src_t, int ncols_dst, bool has_fusion>
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())* ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx, const void* __restrict__ vy, const int32_t* __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float* __restrict__ dst,
    const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
    const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
    const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
    const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    constexpr int qk = ggml_cuda_type_traits<src_t>::qk;
    constexpr int qi = ggml_cuda_type_traits<src_t>::qi;
    constexpr int vdr = ggml_cuda_type_traits<src_t>::mmvq;
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const     int tid = warp_size * threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block * blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    // The MUL_MAT_ID code path with ids != nullptr is only implemented for ncols_dst == 1.
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x = ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    [[maybe_unused]] bool use_gate = false;
    [[maybe_unused]] bool use_bias = false;
    [[maybe_unused]] bool use_gate_bias = false;
    [[maybe_unused]] const void* vgate = nullptr;
    [[maybe_unused]] const float* x_bias = nullptr;
    [[maybe_unused]] const float* gate_bias = nullptr;
    [[maybe_unused]] ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate = fusion.gate;
        x_bias = (const float*)fusion.x_bias;
        gate_bias = (const float*)fusion.gate_bias;
        active_glu = fusion.glu_op;
    }

    const uint32_t channel_bias = ids ? channel_x : channel_dst;

    float x_biases[ncols_dst][rows_per_cuda_block] = { { 0.0f } };
    float gate_biases[ncols_dst][rows_per_cuda_block] = { { 0.0f } };
    if constexpr (has_fusion) {
        if (use_bias) {
            x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j][threadIdx.x] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j][threadIdx.x] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = { {0.0f} };
    [[maybe_unused]] float tmp_gate[ncols_dst][rows_per_cuda_block] = { {0.0f} };

    const block_q8_1* y = ((const block_q8_1*)vy) + sample_y * stride_sample_y + channel_y * stride_channel_y;
    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row0 * stride_row_x;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q(
                    static_cast<const src_t*>(vx) + kbx_offset + i * stride_row_x + kbx, &y[j * stride_col_y + kby], kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q(
                            static_cast<const src_t*>(vgate) + kbx_offset + i * stride_row_x + kbx, &y[j * stride_col_y + kby], kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps - 1 > 0)) ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if constexpr (!has_fusion) {
        (void)tmp_shared_gate;
    }
    else if (!use_gate) {
        (void)tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y - 1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j][threadIdx.x];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j][threadIdx.x];
                    }
                    switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= silu(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= gelu(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI: {
                        result = swiglu_oai(gate_value, result);
                        break;
                    }
                    default:
                        result = result * gate_value;
                        break;
                    }
                }
            }
            dst[j * stride_col_dst + threadIdx.x] = result;
        }
    }

}

template<ggml_type type, typename src_t, int c_ncols_dst>
static void mul_mat_vec_q_switch_fusion(
    const void* vx, const void* vy, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
    const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
    const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
    const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
    const dim3& block_nums, const dim3& block_dims, const int nbytes_shared, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_q<type, src_t, c_ncols_dst, true> << <block_nums, block_dims, nbytes_shared, stream >> >
                (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, src_t, c_ncols_dst, false> << <block_nums, block_dims, nbytes_shared, stream >> >
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
}


template <ggml_type type, typename src_t>
static void mul_mat_vec_q_switch_ncols_dst(
    const void* vx, const void* vy, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const int ncols_x, const int nrows_x, const int ncols_dst,
    const int stride_row_x, const int stride_col_y, const int stride_col_dst,
    const int nchannels_x, const int nchannels_y, const int nchannels_dst,
    const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
    cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd = init_fastdiv_values(nsamples_dst / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    [[maybe_unused]] const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    GGML_ASSERT(!ids || ncols_dst == 1);
    switch (ncols_dst) {
    case 1: {
        constexpr int c_ncols_dst = 1;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 2: {
        constexpr int c_ncols_dst = 2;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 3: {
        constexpr int c_ncols_dst = 3;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 4: {
        constexpr int c_ncols_dst = 4;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 5: {
        constexpr int c_ncols_dst = 5;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 6: {
        constexpr int c_ncols_dst = 6;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 7: {
        constexpr int c_ncols_dst = 7;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    case 8: {
        constexpr int c_ncols_dst = 8;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, stream);
    } break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

void mul_mat_vec_q_switch_type(const mat_vec_q_switch_context &ctx, cudaStream_t stream)
{
    switch (ctx.type_x) {
    case GGML_TYPE_Q4_0:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0, block_q4_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
             ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
             ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
             ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
             ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
             ctx.nsamples_x, ctx.nsamples_dst,
             ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q4_1:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1, block_q4_1>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q5_0:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0, block_q5_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q5_1:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1, block_q5_1>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q8_0:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0, block_q8_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_MXFP4:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4, block_mxfp4>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q2_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K, block_q2_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q3_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K, block_q3_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q4_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K, block_q4_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q5_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K, block_q5_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q6_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K, block_q6_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ2_XXS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS, block_iq2_xxs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ2_XS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS, block_iq2_xs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ2_S:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S, block_iq2_s>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ3_XXS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS, block_iq3_xxs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ1_S:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S, block_iq1_s>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ1_M:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M, block_iq1_m>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ4_NL:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL, block_iq4_nl>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ4_XS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS, block_iq4_xs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ3_S:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S, block_iq3_s>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst,
                stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}
