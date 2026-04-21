#include "../common.h"
#include "common.cuh"
#include "reduce.cuh"
#include "vecdotq.cuh"
#include "cuda_func.h"
#include "unary.cuh"

#define GGML_ASSERT(...)
#define GGML_ABORT(...)
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
        case 1:
            return small_k ? nwarps : 1;
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

static constexpr __host__ __device__ int calc_nwarps(internal::ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
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
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
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
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case internal::GGML_TYPE_Q4_0:
                case internal::GGML_TYPE_Q4_1:
                case internal::GGML_TYPE_Q5_0:
                case internal::GGML_TYPE_Q5_1:
                case internal::GGML_TYPE_Q8_0:
                case internal::GGML_TYPE_Q2_K:
                case internal::GGML_TYPE_Q4_K:
                case internal::GGML_TYPE_Q5_K:
                case internal::GGML_TYPE_Q6_K:
                case internal::GGML_TYPE_IQ4_NL:
                case internal::GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case internal::GGML_TYPE_Q4_0:
                case internal::GGML_TYPE_Q4_1:
                case internal::GGML_TYPE_Q5_0:
                case internal::GGML_TYPE_Q5_1:
                case internal::GGML_TYPE_Q8_0:
                case internal::GGML_TYPE_Q4_K:
                case internal::GGML_TYPE_Q6_K:
                case internal::GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

template <internal::ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template <internal::ggml_type type, typename src_t, int ncols_dst, bool has_fusion, bool is_multi_token_id = false, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx, const void* __restrict__ vy, const int32_t* __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float* __restrict__ dst,
    const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
    const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
    const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
    const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
    const uint32_t ids_stride) {

    constexpr int qk = ggml_cuda_type_traits<src_t>::qk;
    constexpr int qi = ggml_cuda_type_traits<src_t>::qi;
    constexpr int vdr = ggml_cuda_type_traits<src_t>::mmvq;
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<warp_size>(block);

    const     int tid = warp_size * threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block * blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t token_idx = 0;
    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx = blockIdx.z;
        channel_x = ids[channel_dst + token_idx * ids_stride];
        channel_y = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    }
    else {
        channel_x = ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
        channel_y = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
        sample_dst = blockIdx.z;
    }

    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    [[maybe_unused]] bool use_gate = false;
    [[maybe_unused]] bool use_bias = false;
    [[maybe_unused]] bool use_gate_bias = false;
    [[maybe_unused]] const void* vgate = nullptr;
    [[maybe_unused]] const float* x_bias = nullptr;
    [[maybe_unused]] const float* gate_bias = nullptr;
    [[maybe_unused]] internal::ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate = fusion.gate;
        x_bias = (const float*)fusion.x_bias;
        gate_bias = (const float*)fusion.gate_bias;
        active_glu = fusion.glu_op;
    }


    float x_biases[ncols_dst] = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = { {0.0f} };
    [[maybe_unused]] float tmp_gate[ncols_dst][rows_per_cuda_block] = { {0.0f} };

    const block_q8_1* y = ((const block_q8_1*)vy) + sample_y * stride_sample_y + channel_y * stride_channel_y;
    if constexpr (is_multi_token_id) {
        y += token_idx * stride_col_y;
    }
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

    if constexpr (is_multi_token_id) {
        dst += token_idx * stride_col_dst;
    }

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
            tmp[j][i] = cooperative_groups::reduce(tile, tmp[j][i], cooperative_groups::plus<float>());
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = cooperative_groups::reduce(tile, tmp_gate[j][i], cooperative_groups::plus<float>());
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                    case internal::GGML_GLU_OP_SWIGLU:
                        result *= silu(gate_value);
                        break;
                    case internal::GGML_GLU_OP_GEGLU:
                        result *= gelu(gate_value);
                        break;
                    case internal::GGML_GLU_OP_SWIGLU_OAI: {
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

template <internal::ggml_type type, typename src_t, int c_ncols_dst, bool is_multi_token_id = false, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
    const void* vx, const void* vy, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
    const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
    const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
    const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
    const dim3& block_nums, const dim3& block_dims, const int nbytes_shared,
    const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_q<type, src_t, c_ncols_dst, true, is_multi_token_id, small_k> << <block_nums, block_dims, nbytes_shared, stream >> >
                (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, src_t, c_ncols_dst, false, is_multi_token_id, small_k> << <block_nums, block_dims, nbytes_shared, stream >> >
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}


template <internal::ggml_type type, typename src_t>
static void mul_mat_vec_q_switch_ncols_dst(
    const void* vx, const void* vy, const int32_t* ids, const ggml_cuda_mm_fusion_args_device fusion, float* dst,
    const int ncols_x, const int nrows_x, const int ncols_dst,
    const int stride_row_x, const int stride_col_y, const int stride_col_dst,
    const int nchannels_x, const int nchannels_y, const int nchannels_dst,
    const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
    const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd = init_fastdiv_values(nsamples_dst / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    [[maybe_unused]] const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk = ggml_cuda_type_traits<src_t>::qk;
        constexpr int qi = ggml_cuda_type_traits<src_t>::qi;
        constexpr int vdr = ggml_cuda_type_traits<src_t>::mmvq;
        const int     blocks_per_row_x = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        using namespace internal;
        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        }
        else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
            (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
            GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, ncols_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst, true>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
    case 1: {
            constexpr int c_ncols_dst = 1;

            const bool use_small_k = should_use_small_k(c_ncols_dst);
            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst,
                                                                    warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst, false, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                    dims.first, dims.second, 0, ids_stride, stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst,
                                                                    warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                    dims.first, dims.second, 0, ids_stride, stream);
            }
    } break;
    case 2: {
        constexpr int c_ncols_dst = 2;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    case 3: {
        constexpr int c_ncols_dst = 3;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    case 4: {
        constexpr int c_ncols_dst = 4;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    case 5: {
        constexpr int c_ncols_dst = 5;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    case 6: {
        constexpr int c_ncols_dst = 6;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    case 7: {
        constexpr int c_ncols_dst = 7;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    case 8: {
        constexpr int c_ncols_dst = 8;
        std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q_switch_fusion<type, src_t, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
            dims.first, dims.second, 0, ids_stride, stream);
    } break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

void mul_mat_vec_q_switch_type(const mat_vec_q_switch_context &ctx, cudaStream_t stream)
{
    switch (ctx.type_x) {
    case internal::GGML_TYPE_Q1_0:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q1_0, block_q1_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
             ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
             ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
             ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
             ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
             ctx.nsamples_x, ctx.nsamples_dst,
             ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q4_0:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q4_0, block_q4_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
             ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
             ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
             ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
             ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
             ctx.nsamples_x, ctx.nsamples_dst,
             ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q4_1:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q4_1, block_q4_1>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q5_0:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q5_0, block_q5_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q5_1:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q5_1, block_q5_1>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q8_0:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q8_0, block_q8_0>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_MXFP4:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_MXFP4, block_mxfp4>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_NVFP4:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_NVFP4, block_nvfp4>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q2_K:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q2_K, block_q2_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q3_K:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q3_K, block_q3_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q4_K:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q4_K, block_q4_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q5_K:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q5_K, block_q5_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_Q6_K:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_Q6_K, block_q6_K>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ2_XXS:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ2_XXS, block_iq2_xxs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ2_XS:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ2_XS, block_iq2_xs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ2_S:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ2_S, block_iq2_s>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ3_XXS:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ3_XXS, block_iq3_xxs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ1_S:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ1_S, block_iq1_s>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ1_M:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ1_M, block_iq1_m>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ4_NL:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ4_NL, block_iq4_nl>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ4_XS:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ4_XS, block_iq4_xs>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    case internal::GGML_TYPE_IQ3_S:
        mul_mat_vec_q_switch_ncols_dst<internal::GGML_TYPE_IQ3_S, block_iq3_s>
            (ctx.vx, ctx.vy, ctx.ids, ctx.fusion, ctx.dst,
                ctx.ncols_x, ctx.nrows_x, ctx.ncols_dst,
                ctx.stride_row_x, ctx.stride_col_y, ctx.stride_col_dst,
                ctx.nchannels_x, ctx.nchannels_y, ctx.nchannels_dst,
                ctx.stride_channel_x, ctx.stride_channel_y, ctx.stride_channel_dst,
                ctx.nsamples_x, ctx.nsamples_dst,
                ctx.stride_sample_x, ctx.stride_sample_y, ctx.stride_sample_dst, ctx.ids_stride,
                stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}


// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ1_S:   return 6;
    case GGML_TYPE_IQ1_M:   return 6;
    case GGML_TYPE_IQ2_S:   return 4;
    case GGML_TYPE_IQ2_XS:  return 5;
    case GGML_TYPE_IQ2_XXS: return 5;
    case GGML_TYPE_IQ3_S:   return 4;
    case GGML_TYPE_IQ3_XXS: return 4;
    case GGML_TYPE_IQ4_NL:  return 6;
    case GGML_TYPE_IQ4_XS:  return 5;
    case GGML_TYPE_MXFP4:   return 4;
    case GGML_TYPE_Q2_K:    return 4;
    case GGML_TYPE_Q3_K:    return 4;
    case GGML_TYPE_Q4_0:    return 6;
    case GGML_TYPE_Q4_1:    return 6;
    case GGML_TYPE_Q4_K:    return 5;
    case GGML_TYPE_Q5_0:    return 6;
    case GGML_TYPE_Q5_1:    return 6;
    case GGML_TYPE_Q5_K:    return 5;
    case GGML_TYPE_Q6_K:    return 4;
    case GGML_TYPE_Q8_0:    return 4;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ2_S:   return 7;
    case GGML_TYPE_IQ3_S:   return 6;
    case GGML_TYPE_IQ3_XXS: return 7;
    case GGML_TYPE_MXFP4:   return 7;
    case GGML_TYPE_Q2_K:    return 7;
    case GGML_TYPE_Q3_K:    return 5;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ1_S:   return 5;
    case GGML_TYPE_IQ1_M:   return 5;
    case GGML_TYPE_IQ2_S:   return 4;
    case GGML_TYPE_IQ2_XS:  return 4;
    case GGML_TYPE_IQ2_XXS: return 4;
    case GGML_TYPE_IQ3_S:   return 4;
    case GGML_TYPE_IQ3_XXS: return 4;
    case GGML_TYPE_IQ4_NL:  return 6;
    case GGML_TYPE_IQ4_XS:  return 4;
    case GGML_TYPE_Q2_K:    return 4;
    case GGML_TYPE_Q3_K:    return 4;
    case GGML_TYPE_Q4_0:    return 5;
    case GGML_TYPE_Q4_1:    return 5;
    case GGML_TYPE_Q4_K:    return 4;
    case GGML_TYPE_Q5_K:    return 4;
    case GGML_TYPE_Q6_K:    return 4;
    case GGML_TYPE_Q8_0:    return 4;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ2_S:   return 5;
    case GGML_TYPE_IQ2_XS:  return 5;
    case GGML_TYPE_IQ2_XXS: return 5;
    case GGML_TYPE_IQ3_S:   return 4;
    case GGML_TYPE_IQ3_XXS: return 5;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ2_S:   return 4;
    case GGML_TYPE_IQ2_XS:  return 4;
    case GGML_TYPE_IQ2_XXS: return 4;
    case GGML_TYPE_IQ3_S:   return 4;
    case GGML_TYPE_IQ3_XXS: return 4;
    case GGML_TYPE_Q2_K:    return 7;
    case GGML_TYPE_Q3_K:    return 4;
    case GGML_TYPE_Q4_K:    return 5;
    case GGML_TYPE_Q5_K:    return 6;
    case GGML_TYPE_Q6_K:    return 5;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ1_S:   return 6;
    case GGML_TYPE_IQ1_M:   return 6;
    case GGML_TYPE_IQ2_S:   return 4;
    case GGML_TYPE_IQ2_XS:  return 4;
    case GGML_TYPE_IQ2_XXS: return 4;
    case GGML_TYPE_IQ3_S:   return 4;
    case GGML_TYPE_IQ3_XXS: return 4;
    case GGML_TYPE_IQ4_NL:  return 6;
    case GGML_TYPE_IQ4_XS:  return 6;
    case GGML_TYPE_Q4_K:    return 4;
    case GGML_TYPE_Q5_K:    return 4;
    case GGML_TYPE_Q6_K:    return 4;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(internal::ggml_type type) {
    using namespace internal;
    switch (type) {
    case GGML_TYPE_IQ1_S:   return 7;
    case GGML_TYPE_IQ1_M:   return 7;
    case GGML_TYPE_IQ2_S:   return 4;
    case GGML_TYPE_IQ2_XS:  return 4;
    case GGML_TYPE_IQ2_XXS: return 4;
    case GGML_TYPE_IQ3_S:   return 4;
    case GGML_TYPE_IQ3_XXS: return 4;
    case GGML_TYPE_IQ4_NL:  return 7;
    case GGML_TYPE_IQ4_XS:  return 5;
    case GGML_TYPE_MXFP4:   return 5;
    case GGML_TYPE_Q3_K:    return 4;
    case GGML_TYPE_Q4_0:    return 7;
    case GGML_TYPE_Q4_1:    return 7;
    case GGML_TYPE_Q4_K:    return 4;
    case GGML_TYPE_Q5_0:    return 7;
    case GGML_TYPE_Q5_1:    return 7;
    case GGML_TYPE_Q5_K:    return 5;
    case GGML_TYPE_Q6_K:    return 5;
    case GGML_TYPE_Q8_0:    return 7;
    default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(internal::ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}