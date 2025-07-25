#include <tuple>
#include "../common.h"
#include "internal_ds.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include "cuda_func.h"
#include "mma.cuh"
#include "../vendor_constant.h"

#define MMQ_ITER_K 256
#define MMQ_NWARPS 8

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

template <ggml_type type, typename src_t, int ncols_dst>
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())* ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx, const void* __restrict__ vy, const int32_t* __restrict__ ids, float* __restrict__ dst,
    const int ncols_x, const int nchannels_y, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
    const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

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

    // The MUL_MAT_ID code path with ids != nullptr is only implemetned for ncols_dst == 1.
    const int channel_dst = blockIdx.y;
    const int channel_x = ncols_dst == 1 && ids ? ids[channel_dst] : channel_dst / channel_ratio;
    const int channel_y = ncols_dst == 1 && ids ? channel_dst % nchannels_y : channel_dst;
    const int sample_dst = blockIdx.z;
    const int sample_x = sample_dst / sample_ratio;
    const int sample_y = sample_dst;

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = { {0.0f} };

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
            }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
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
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + int(threadIdx.x) < stride_col_dst)) {
            dst[j * stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <ggml_type type, typename src_t>
static void mul_mat_vec_q_switch_ncols_dst(
    const void* vx, const void* vy, const int32_t* ids, float* dst,
    const int ncols_x, const int nrows_x, const int ncols_dst,
    const int stride_row_x, const int stride_col_y, const int stride_col_dst,
    const int nchannels_x, const int nchannels_y, const int nchannels_dst,
    const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
    cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const int channel_ratio = nchannels_dst / nchannels_x;
    const int sample_ratio = nsamples_dst / nsamples_x;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    GGML_ASSERT(!ids || ncols_dst == 1);
    switch (ncols_dst) {
    case 1:
    {
        constexpr int c_ncols_dst = 1;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 2:
    {
        constexpr int c_ncols_dst = 2;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 3:
    {
        constexpr int c_ncols_dst = 3;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 4:
    {
        constexpr int c_ncols_dst = 4;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 5:
    {
        constexpr int c_ncols_dst = 5;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 6:
    {
        constexpr int c_ncols_dst = 6;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 7:
    {
        constexpr int c_ncols_dst = 7;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    case 8:
    {
        constexpr int c_ncols_dst = 8;
        std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
        mul_mat_vec_q<type, src_t, c_ncols_dst> << <dims.first, dims.second, 0, stream >> >
            (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        break;
    }
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

void mul_mat_vec_q_switch_type(const mat_vec_q_switch_context* ctx, cudaStream_t stream)
{
    switch (ctx->type_x) {
    case GGML_TYPE_Q4_0:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0, block_q4_0>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
             ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
             ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
             ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
             ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
             ctx->nsamples_x, ctx->nsamples_dst,
             ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q4_1:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1, block_q4_1>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q5_0:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0, block_q5_0>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q5_1:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1, block_q5_1>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q8_0:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0, block_q8_0>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q2_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K, block_q2_K>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q3_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K, block_q3_K>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q4_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K, block_q4_K>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q5_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K, block_q5_K>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_Q6_K:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K, block_q6_K>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ2_XXS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS, block_iq2_xxs>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ2_XS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS, block_iq2_xs>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ2_S:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S, block_iq2_s>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ3_XXS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS, block_iq3_xxs>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ1_S:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S, block_iq1_s>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ1_M:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M, block_iq1_m>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ4_NL:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL, block_iq4_nl>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ4_XS:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS, block_iq4_xs>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    case GGML_TYPE_IQ3_S:
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S, block_iq3_s>
            (ctx->vx, ctx->vy, ctx->ids, ctx->dst,
                ctx->ncols_x, ctx->nrows_x, ctx->ncols_dst,
                ctx->stride_row_x, ctx->stride_col_y, ctx->stride_col_dst,
                ctx->nchannels_x, ctx->nchannels_y, ctx->nchannels_dst,
                ctx->stride_channel_x, ctx->stride_channel_y, ctx->stride_channel_dst,
                ctx->nsamples_x, ctx->nsamples_dst,
                ctx->stride_sample_x, ctx->stride_sample_y, ctx->stride_sample_dst,
                stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

static constexpr __device__ int get_mmq_x_max_device() {
#ifdef NEW_MMA_AVAILABLE
    return 128;
#else // NEW_MMA_AVAILABLE

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    return 128;
#else // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#ifdef GGML_CUDA_FORCE_MMQ
    return MMQ_DP4A_MAX_BATCH_SIZE;
#else // GGML_CUDA_FORCE_MMQ
    return 128;
#endif // GGML_CUDA_FORCE_MMQ
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

    return 64;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#endif // NEW_MMA_AVAILABLE
}

static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    if constexpr (new_mma_available_v) {
        return mmq_x >= 48 ? 16 : 8;
    }
    else {
        return 8;
    }
}

static constexpr __device__ int get_mmq_y_device() {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA1)
    return 64;
#else
    return 128;
#endif // defined RDNA1
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 128;
#else
    return 64;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
}

using vec_dot_mmq_t = void (*)(const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00);

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_0   + mmq_y/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_1   + mmq_y/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*2/QI8_0 + mmq_y/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*4/QI8_0 + mmq_y/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE*2/QI8_1 + mmq_y/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE         + mmq_y,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y,                                     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_K,                     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_K   + mmq_y/QI5_K,     mmq_y*WARP_SIZE/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI6_K   + mmq_y/QI6_K,     mmq_y*WARP_SIZE/8 + mmq_y/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    return type == GGML_TYPE_Q4_0 ? MMQ_DP4A_TXS_Q4_0 :
        type == GGML_TYPE_Q4_1 ? MMQ_DP4A_TXS_Q4_1 :
        type == GGML_TYPE_Q5_0 ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_Q5_1 ? MMQ_DP4A_TXS_Q8_1 :
        type == GGML_TYPE_Q8_0 ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_Q2_K ? MMQ_DP4A_TXS_Q2_K :
        type == GGML_TYPE_Q3_K ? MMQ_DP4A_TXS_Q3_K :
        type == GGML_TYPE_Q4_K ? MMQ_DP4A_TXS_Q4_K :
        type == GGML_TYPE_Q5_K ? MMQ_DP4A_TXS_Q5_K :
        type == GGML_TYPE_Q6_K ? MMQ_DP4A_TXS_Q6_K :
        type == GGML_TYPE_IQ2_XXS ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ2_XS ? MMQ_DP4A_TXS_Q8_0_16 :
        type == GGML_TYPE_IQ2_S ? MMQ_DP4A_TXS_Q8_0_16 :
        type == GGML_TYPE_IQ3_XXS ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ3_S ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ1_S ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ4_XS ? MMQ_DP4A_TXS_Q8_0 :
        type == GGML_TYPE_IQ4_NL ? MMQ_DP4A_TXS_Q8_0 :
        tile_x_sizes{ 0, 0, 0 };
}


template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q4_0* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + 2 * WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0* bxi = x + kbx0 + i * stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + 0] = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0* bxi = (const block_q4_0*)x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(std::bit_cast<half>(bxi->d));
#else
        x_df[i * (WARP_SIZE / QI4_0) + i / QI4_0 + kbxd] = __half2float(std::bit_cast<half>(bxi->d));
#endif // NEW_MMA_AVAILABLE
    }
}

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

template <typename T>
struct mma_A_I16K8 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int I = 16;
    static constexpr int K = 8;
    static constexpr int ne = 4;

    T x[ne];

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l % 2) * (I / 2) + threadIdx.x / (K / 2);
        [[assume(ret >= 0)]];
        [[assume(ret < I)]];
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = (l / 2) * (K / 2) + threadIdx.x % (K / 2);
        [[assume(ret >= 0)]];
        [[assume(ret < K)]];
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T* __restrict__ xs0, const int& stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l) * stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T* __restrict__ xs0, const int& stride) {
#ifdef NEW_MMA_AVAILABLE
        int* xi = (int*)x;
        const int* xs = (const int*)xs0 + (threadIdx.x % I) * stride + (threadIdx.x / I) * (K / 2);
        asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "l"(xs));
#else
        (void)(xs0);
        (void)(stride);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void load_ldmatrix_trans(const T* __restrict__ xs0, const int& stride) {
#ifdef NEW_MMA_AVAILABLE
        int* xi = (int*)x;
        const int* xs = (const int*)xs0 + (threadIdx.x % I) * stride + (threadIdx.x / I) * (K / 2);
        asm("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(xi[0]), "+r"(xi[2]), "+r"(xi[1]), "+r"(xi[3])
            : "l"(xs));
#else
        (void)(xs0);
        (void)(stride);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void transpose() {
        int* xi = (int*)x;
        xi[0] = ggml_cuda_movmatrix(xi[0]);

        const int tmp = ggml_cuda_movmatrix(xi[1]);
        xi[1] = ggml_cuda_movmatrix(xi[2]);
        xi[2] = tmp;

        xi[3] = ggml_cuda_movmatrix(xi[3]);
    }
};

template <typename T>
struct mma_A_I16K4 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int I = 16;
    static constexpr int K = 4;
    static constexpr int ne = 2;

    T x[ne];

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l % 2) * (I / 2) + threadIdx.x / K;
        [[assume(ret >= 0)]];
        [[assume(ret < I)]];
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        [[assume(ret >= 0)]];
        [[assume(ret < K)]];
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T* __restrict__ xs0, const int& stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l) * stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T* __restrict__ xs0, const int& stride) {
#ifdef NEW_MMA_AVAILABLE
        int* xi = (int*)x;
        const int* xs = (const int*)xs0 + (threadIdx.x % I) * stride;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(xi[0]), "+r"(xi[1])
            : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_B_J8K4 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int J = 8;
    static constexpr int K = 4;
    static constexpr int ne = 1;

    T x[ne];

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / K;
        [[assume(ret >= 0)]];
        [[assume(ret < J)]];
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        [[assume(ret >= 0)]];
        [[assume(ret < K)]];
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T* __restrict__ xs0, const int& stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l) * stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T* __restrict__ xs0, const int& stride) {
#ifdef NEW_MMA_AVAILABLE
        int* xi = (int*)x;
        const int* xs = (const int*)xs0 + (threadIdx.x % J) * stride;
        asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];"
            : "+r"(xi[0]) : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_B_J8K8 {
    static_assert(sizeof(T) == 4, "bad type size");

    static constexpr int J = 8;
    static constexpr int K = 8;
    static constexpr int ne = 2;

    T x[ne];

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / (K / 2);
        [[assume(ret >= 0)]];
        [[assume(ret < J)]];
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = l * (K / 2) + threadIdx.x % (K / 2);
        [[assume(ret >= 0)]];
        [[assume(ret < K)]];
        return ret;
    }

    __device__ __forceinline__ void load_generic(const T* __restrict__ xs0, const int& stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l) * stride + get_k(l)];
        }
    }

    __device__ __forceinline__ void load_ldmatrix(const T* __restrict__ xs0, const int& stride) {
#ifdef NEW_MMA_AVAILABLE
        int* xi = (int*)x;
        const int* xs = (const int*)xs0 + (threadIdx.x % J) * stride + ((threadIdx.x / J) * (K / 2)) % K;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(xi[0]), "+r"(xi[1])
            : "l"(xs));
#else
        load_generic(xs0, stride);
#endif // NEW_MMA_AVAILABLE
    }
};

template <typename T>
struct mma_C_I16J8 {};

template <>
struct mma_C_I16J8<int> {
    static constexpr int I = 16;
    static constexpr int J = 8;
    static constexpr int ne = 4;

    int x[ne] = { 0 };

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l / 2) * (I / 2) + threadIdx.x / (J / 2);
        [[assume(ret >= 0)]];
        [[assume(ret < I)]];
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J / 2)) + l % 2;
        [[assume(ret >= 0)]];
        [[assume(ret < J)]];
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K4<int>& mma_A, const mma_B_J8K4<int>& mma_B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        (void)(mma_A);
        (void)(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<int>& mma_A, const mma_B_J8K8<int>& mma_B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_A.x[2]), "r"(mma_A.x[3]), "r"(mma_B.x[0]), "r"(mma_B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[2]), "r"(mma_B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[3]), "r"(mma_B.x[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        (void)(mma_A);
        (void)(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }
};

template <>
struct mma_C_I16J8<half2> {
    static constexpr int I = 16;
    static constexpr int J = 4;
    static constexpr int ne = 2;

    half2 x[ne] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = l * (I / 2) + threadIdx.x / J;
        [[assume(ret >= 0)]];
        [[assume(ret < I)]];
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x % J;
        [[assume(ret >= 0)]];
        [[assume(ret < J)]];
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<half2>& mma_A, const mma_B_J8K8<half2>& mma_B) {
#ifdef NEW_MMA_AVAILABLE
        int* Axi = (int*)mma_A.x;
        int* Bxi = (int*)mma_B.x;
        int* xi = (int*)x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(xi[0]), "+r"(xi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        (void)(mma_A);
        (void)(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ mma_B_J8K8<half2> to_mma_B() {
        mma_B_J8K8<half2> mma_B;

        int* xi = (int*)x;
        int* Bxi = (int*)mma_B.x;
        Bxi[0] = ggml_cuda_movmatrix(xi[0]);
        Bxi[1] = ggml_cuda_movmatrix(xi[1]);

        return mma_B;
    }
};

template <>
struct mma_C_I16J8<float> {
    static constexpr int I = 16;
    static constexpr int J = 8;
    static constexpr int ne = 4;

    float x[ne] = { 0.0f, 0.0f, 0.0f, 0.0f };

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l / 2) * (I / 2) + threadIdx.x / (J / 2);
        [[assume(ret >= 0)]];
        [[assume(ret < I)]];
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J / 2)) + l % 2;
        [[assume(ret >= 0)]];
        [[assume(ret < J)]];
        return ret;
    }

    __device__ __forceinline__ void mma(const mma_A_I16K8<half2>& mma_A, const mma_B_J8K8<half2>& mma_B) {
#ifdef NEW_MMA_AVAILABLE
        int* Axi = (int*)mma_A.x;
        int* Bxi = (int*)mma_B.x;
        int* xi = (int*)x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(xi[0]), "+r"(xi[1]), "+r"(xi[2]), "+r"(xi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        (void)(mma_A);
        (void)(mma_B);
        NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
    }

    __device__ __forceinline__ mma_B_J8K8<half2> to_mma_B() {
        mma_B_J8K8<half2> mma_B;
        mma_B.x[0] = make_half2(x[0], x[1]);
        mma_B.x[1] = make_half2(x[2], x[3]);

        int* Bxi = (int*)mma_B.x;
        Bxi[0] = ggml_cuda_movmatrix(Bxi[0]);
        Bxi[1] = ggml_cuda_movmatrix(Bxi[1]);

        return mma_B;
    }

    __device__ __forceinline__ void load_generic(const float* __restrict__ xs0, const int& stride) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l) * stride + get_i(l)];
        }
    }
};

#define MMQ_MMA_TILE_X_K_Q8_0 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q8_1 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q2_K (2*WARP_SIZE + WARP_SIZE                         + 4)
#define MMQ_MMA_TILE_X_K_Q3_K (2*WARP_SIZE + WARP_SIZE/2                       + 4)
#define MMQ_MMA_TILE_X_K_Q6_K (2*WARP_SIZE + WARP_SIZE/QI6_K     + WARP_SIZE/8 + 7)

#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)
template <int mmq_x, int mmq_y, int nwarps, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    typedef mma_A_I16K8<int> mma_A;
    typedef mma_B_J8K8<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J * MMQ_TILE_Y_K);

    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + 2 * WARP_SIZE;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;
    const half2* y_ds = (const half2*)y;

    mma_A A[ntx][WARP_SIZE / QI8_0];
    float dA[ntx][mma_C::ne / 2][WARP_SIZE / QI8_0];

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
            const int k0 = k00 + k01;

            A[n][k01 / QI8_0].load_ldmatrix(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne / 2; ++l) {
            const int i = i0 + n * mma_A::I + mma_C::get_i(2 * l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01 / QI8_0] = x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 / QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
            mma_B  B;
            float dB[mma_C::ne / 2];

            B.load_generic(y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < mma_C::ne / 2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
                }
                else {
                    dB[l] = __low2float(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma(A[n][k01 / QI8_0], B);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] += C.x[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
                }
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, mmq_y);
    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + txs.qs;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_0 * VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2)) + (k01 / 2) % (QI8_1 / 2);

                int u[2 * VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
                for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
                    u[2 * l + 0] = y_qs[j * MMQ_TILE_Y_K + kyqs + l];
                    u[2 * l + 1] = y_qs[j * MMQ_TILE_Y_K + kyqs + (l + QI4_0)];
                }

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i * (WARP_SIZE + 1) + k0 / QR4_0], u,
                        x_df[i * (WARP_SIZE / QI4_0) + i / QI4_0 + k0 / (QR4_0 * QI4_0)], y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    typedef mma_A_I16K8<int> mma_A;
    typedef mma_B_J8K8<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J * MMQ_TILE_Y_K);

    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + 2 * WARP_SIZE;
    const int* y_qs = (const int*)y + 4;
    const half2* y_dm = (const half2*)y;

    mma_A    A[ntx][WARP_SIZE / QI8_1];
    float2 dmA[ntx][mma_C::ne / 2][WARP_SIZE / QI8_1];

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            A[n][k01 / QI8_1].load_ldmatrix(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne / 2; ++l) {
            const int i = i0 + n * mma_A::I + mma_C::get_i(2 * l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01 / QI8_1] = __half22float2(x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + k0 / QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B    B;
            float2 dsB[mma_C::ne / 2];

            B.load_generic(y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < mma_C::ne / 2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C;
                C.mma(A[n][k01 / QI8_1], B);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] += dmA[n][l / 2][k01 / QI8_1].x * dsB[l % 2].x * C.x[l];
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] += dmA[n][l / 2][k01 / QI8_1].y * dsB[l % 2].y;
                }
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q4_1* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + 2 * WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = threadIdx.x / QI4_1;
    const int kqsx = threadIdx.x % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1* bxi = x + kbx0 + i * stride + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI4_1) + kqsx + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + threadIdx.y * QI4_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1* bxi = (const block_q4_1*)x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE / QI4_1) + i / QI4_1 + kbxd] = __tohalf2(bxi->dm);
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, mmq_y);
    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + txs.qs;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_1 * VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2)) + (k01 / 2) % (QI8_1 / 2);

                int u[2 * VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
                for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
                    u[2 * l + 0] = y_qs[j * MMQ_TILE_Y_K + kyqs + l];
                    u[2 * l + 1] = y_qs[j * MMQ_TILE_Y_K + kyqs + (l + QI4_1)];
                }

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i * (WARP_SIZE + 1) + k0 / QR4_1], u,
                        x_dm[i * (WARP_SIZE / QI4_1) + i / QI4_1 + k0 / (QR4_1 * QI4_1)], y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q5_0* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = threadIdx.x / QI5_0;
    const int kqsx = threadIdx.x % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0* bxi = x + kbx0 + i * stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_0));

        int qs0 = (ql >> 0) & 0x0F0F0F0F;
        qs0 |= (qh << 4) & 0x00000010;  // 0 ->  4
        qs0 |= (qh << 11) & 0x00001000;  // 1 -> 12
        qs0 |= (qh << 18) & 0x00100000;  // 2 -> 20
        qs0 |= (qh << 25) & 0x10000000;  // 3 -> 28
        qs0 = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >> 4) & 0x0F0F0F0F;
        qs1 |= (qh >> 12) & 0x00000010;  // 16 ->  4
        qs1 |= (qh >> 5) & 0x00001000;  // 17 -> 12
        qs1 |= (qh << 2) & 0x00100000;  // 18 -> 20
        qs1 |= (qh << 9) & 0x10000000;  // 19 -> 28
        qs1 = __vsubss4(qs1, 0x10101010); // subtract 16

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI5_0) + kqsx + 0] = qs0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i * (2 * WARP_SIZE + 1) + kbx * (2 * QI5_0) + kqsx + 0] = qs0;
        x_qs[i * (2 * WARP_SIZE + 1) + kbx * (2 * QI5_0) + kqsx + QI5_0] = qs1;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + threadIdx.y * QI5_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0* bxi = (const block_q5_0*)x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(std::bit_cast<half>(bxi->d));
#else
        x_df[i * (WARP_SIZE / QI5_0) + i / QI5_0 + kbxd] = __half2float(std::bit_cast<half>(bxi->d));
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + txs.qs;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q8_1_q8_1_impl<QR5_1 * VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i * (2 * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k01],
                        x_dm[i * (WARP_SIZE / QI5_1) + i / QI5_1 + k0 / QI8_1], y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + txs.qs;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i * (2 * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k0 % WARP_SIZE],
                        x_df[i * (2 * WARP_SIZE / QI8_0) + i / (QI8_0 / 2) + k0 / QI8_0], y_df[j * MMQ_TILE_Y_K + (k0 / QI8_1) % (WARP_SIZE / QI8_1)]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq4_nl* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = threadIdx.x / QI4_NL;
    const int kqsx = threadIdx.x % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl* bxi = x + kbx0 + i * stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;

        if constexpr (new_mma_available_v) {
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
        }
        else {
            x_qs[i * (2 * WARP_SIZE + 1) + k0 + 0] = v.x;
            x_qs[i * (2 * WARP_SIZE + 1) + k0 + 4] = v.y;
        }
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_NL;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_NL) {
        int i = i0 + threadIdx.y * QI4_NL + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_nl* bxi = (const block_iq4_nl*)x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(bxi->d);
#else
        x_df[i * (WARP_SIZE / 4) + i / 4 + kbxd] = __half2float(bxi->d);
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q5_1* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + 2 * WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = threadIdx.x / QI5_1;
    const int kqsx = threadIdx.x % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1* bxi = x + kbx0 + i * stride + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_1));

        int qs0 = (ql >> 0) & 0x0F0F0F0F;
        qs0 |= (qh << 4) & 0x00000010; // 0 ->  4
        qs0 |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0 |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0 |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >> 4) & 0x0F0F0F0F;
        qs1 |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1 |= (qh >> 5) & 0x00001000; // 17 -> 12
        qs1 |= (qh << 2) & 0x00100000; // 18 -> 20
        qs1 |= (qh << 9) & 0x10000000; // 19 -> 28

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI5_1) + kqsx + 0] = qs0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i * (2 * WARP_SIZE + 1) + kbx * (2 * QI5_1) + kqsx + 0] = qs0;
        x_qs[i * (2 * WARP_SIZE + 1) + kbx * (2 * QI5_1) + kqsx + QI5_1] = qs1;
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + threadIdx.y * QI5_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1* bxi = (const block_q5_1*)x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE / QI5_1) + i / QI5_1 + kbxd] = __tohalf2(bxi->dm);
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q8_0* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_tile + 2 * WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = threadIdx.x / QI8_0;
    const int kqsx = threadIdx.x % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0* bxi = x + kbx0 + i * stride + kbx;

        if constexpr (new_mma_available_v) {
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 0 + threadIdx.x] = get_int_b2(bxi[0].qs, kqsx);
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE / QI8_0].qs, kqsx);
        }
        else {
            x_qs[i * (2 * WARP_SIZE + 1) + 0 + threadIdx.x] = get_int_b2(bxi[0].qs, kqsx);
            x_qs[i * (2 * WARP_SIZE + 1) + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE / QI8_0].qs, kqsx);
        }
    }

    const int blocks_per_tile_x_row = 2 * WARP_SIZE / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0 / 2) {
        int i = i0 + threadIdx.y * (QI8_0 / 2) + threadIdx.x / blocks_per_tile_x_row;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0* bxi = (const block_q8_0*)x + kbx0 + i * stride + kbxd;

        if constexpr (new_mma_available_v) {
            x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = __half2float(bxi->d);
        }
        else {
            x_df[i * (2 * WARP_SIZE / QI8_0) + i / (QI8_0 / 2) + kbxd] = __half2float(bxi->d);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {
#ifdef NEW_MMA_AVAILABLE

    typedef mma_A_I16K4<int> mma_A;
    typedef mma_A_I16K8<int> mma_A_K8;
    typedef mma_B_J8K4<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J * MMQ_TILE_Y_K);

    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + WARP_SIZE * 2;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * (ntx * mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne / 2][8];
    float  mA[ntx][mma_C::ne / 2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            ((mma_A_K8*)A[n])[k01 / QI8_1].load_ldmatrix(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_C::ne / 2; ++l) {
            const int i = i0 + n * mma_C::I + mma_C::get_i(2 * l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1 / 2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i * MMQ_MMA_TILE_X_K_Q2_K + k0 / (QI8_1 / 2)]);

                dA[n][l][k01 / (QI8_1 / 2)] = dm.x;
                mA[n][l][k01 / (QI8_1 / 2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * mma_C::J) {
        float2 dB[mma_C::ne / 2];

#pragma unroll
        for (int l = 0; l < mma_C::ne / 2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = __half22float2(y_ds[j * MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            B[0].load_generic(y_qs + j0 * MMQ_TILE_Y_K + (k01 + 0), MMQ_TILE_Y_K);
            B[1].load_generic(y_qs + j0 * MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

            mma_C Cm[2];
            if (k01 >= WARP_SIZE * 3 / 4) {
                mma_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                Cm[0].mma(A1, B[0]);
                Cm[1].mma(A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C Cd[2];

                Cd[0].mma(A[n][k01 / 4 + 0], B[0]);
                Cd[1].mma(A[n][k01 / 4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    float tmp = Cd[0].x[l] * dA[n][l / 2][k01 / 4 + 0] + Cd[1].x[l] * dA[n][l / 2][k01 / 4 + 1];
                    if (k01 >= WARP_SIZE * 3 / 4) {
                        tmp -= Cm[0].x[l] * mA[n][l / 2][k01 / 4 + 0] + Cm[1].x[l] * mA[n][l / 2][k01 / 4 + 1];
                    }
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] += tmp * (k01 < WARP_SIZE / 2 ? dB[l % 2].x : dB[l % 2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE * 3 / 4; k01 += QI8_1) {
            float2 sB[mma_C::ne / 2];

#pragma unroll
            for (int l = 0; l < mma_C::ne / 2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                sB[l] = __half22float2(y_ds[j * MMQ_TILE_Y_K + (1 + k01 / QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] -= mA[n][l / 2][k01 / 4 + 0] * sB[l % 2].x;
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] -= mA[n][l / 2][k01 / 4 + 1] * sB[l % 2].y;
                }
            }
        }
    }
#else
    (void)(x); (void)(y); (void)(sum);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq4_xs* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kbx = 0;           // threadIdx.x / QI4_XS
    const int kqsx = threadIdx.x; // threadIdx.x % QI4_XS

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs* bxi = x + kbx0 + i * stride + kbx;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4);
        const int k0 = 8 * (threadIdx.x / 4) + threadIdx.x % 4;
        if constexpr (new_mma_available_v) {
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 + 4] = v.y;
        }
        else {
            x_qs[i * (2 * WARP_SIZE + 1) + k0 + 0] = v.x;
            x_qs[i * (2 * WARP_SIZE + 1) + k0 + 4] = v.y;
        }
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE / 4);

        if constexpr (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_xs* bxi = (const block_iq4_xs*)x + kbx0 + i * stride;

        const float d = __half2float(std::bit_cast<half>(bxi->d));

        const int ls = ((bxi->scales_l[(threadIdx.x % 8) / 2] >> (4 * (threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2 * (threadIdx.x % 8))) & 0x03) << 4);

        if constexpr (new_mma_available_v) {
            x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * (ls - 32);
        }
        else {
            x_df[i * (WARP_SIZE / 4) + i / 4 + threadIdx.x % 8] = d * (ls - 32);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {
#ifdef NEW_MMA_AVAILABLE

    typedef mma_A_I16K4<int> mma_A;
    typedef mma_A_I16K8<int> mma_A_K8;
    typedef mma_B_J8K4<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J * MMQ_TILE_Y_K);

    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + WARP_SIZE * 2;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;

    const int i0 = (threadIdx.y / ntx) * (ntx * mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne / 2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            ((mma_A_K8*)A[n])[k01 / 8].load_ldmatrix(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q3_K + k0, MMQ_MMA_TILE_X_K_Q3_K);
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne / 2; ++l) {
            const int i = i0 + n * mma_C::I + mma_C::get_i(2 * l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01 / 4] = x_df[i * MMQ_MMA_TILE_X_K_Q3_K + k0 / 4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * mma_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QR3_K * VDR_Q3_K_Q8_1_MMQ) {
            mma_B B[2];
            float dB[mma_C::ne / 2];

            // Here load_generic is faster than load_ldmatrix.
            B[0].load_generic(y_qs + j0 * MMQ_TILE_Y_K + (k01 + 0), MMQ_TILE_Y_K);
            B[1].load_generic(y_qs + j0 * MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne / 2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma(A[n][k01 / 4 + 0], B[0]);
                C[1].mma(A[n][k01 / 4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] += dB[l % 2] * (C[0].x[l] * dA[n][l / 2][k01 / 4 + 0] + C[1].x[l] * dA[n][l / 2][k01 / 4 + 1]);
                }
            }
        }
    }
#else
    (void)(x); (void)(y); (void)(sum);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq3_s* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI3_S / 2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / (QI3_S / 2)) {
        int i = i0 + threadIdx.y * (2 * WARP_SIZE / QI3_S) + threadIdx.x / (QI3_S / 2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_s* bxi = x + kbx0 + i * stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2 * kqsx + 0), get_int_b2(bxi->qs, 2 * kqsx + 1));
        const uint8_t* qs = (const uint8_t*)&qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)],
                iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 8 * kqsx + (2 * l + 1)] = grid_h;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = 1 + 2 * ((bxi->scales[kqsx / 2] >> (((2 * kqsx) << 1) & 0x04)) & 0x0F);
        const float d = __half2float(std::bit_cast<half>(bxi->d));
#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = ls * d;
#else
        x_df[i * (WARP_SIZE / 4) + i / 4 + kqsx] = ls * d;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + txs.qs;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    float2 y_df[mmq_x / nwarps];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0 / nwarps] = __half22float2(y_ds[j * MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR2_K * VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                if (k01 < WARP_SIZE / 2) {
                    constexpr int ns = 2;
                    sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                        &x_qs[i * (2 * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k01],
                        &x_dm[i * (WARP_SIZE + 1) + k0 / 4], k01 < WARP_SIZE / 2 ? y_df[j0 / nwarps].x : y_df[j0 / nwarps].y,
                        &y_ds[j * MMQ_TILE_Y_K + (1 + k01 / QI8_1)]);
                }
                else {
                    constexpr int ns = 1;
                    sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                        &x_qs[i * (2 * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k01],
                        &x_dm[i * (WARP_SIZE + 1) + k0 / 4], k01 < WARP_SIZE / 2 ? y_df[j0 / nwarps].x : y_df[j0 / nwarps].y,
                        &y_ds[j * MMQ_TILE_Y_K + (1 + k01 / QI8_1)]);
                }
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q2_K* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + 2 * WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, mmq_y);
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI2_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / QI2_K) {
        int i = i0 + threadIdx.y * (WARP_SIZE / QI2_K) + threadIdx.x / QI2_K;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K* bxi = x + kbx0 + i * stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2 * l)) & 0x03030303;

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + k] = x_qs_k;
#endif // NEW_MMA_AVAILABLE
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x * (sc_m & 0x0F), bxi_dmf.y * (sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#ifdef NEW_MMA_AVAILABLE
        x_dm[i * MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
#else
        x_dm[i * (WARP_SIZE + 1) + kqsx] = x_dm_ik;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q3_K* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
    int* x_sc = (int*)(x_df + txs.dm);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI3_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / QI3_K) {
        int i = i0 + threadIdx.y * (WARP_SIZE / QI3_K) + threadIdx.x / QI3_K;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K* bxi = x + kbx0 + i * stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K / 2)) >> (4 * (kqsx / (QI3_K / 2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;

            const int x_ql_k = (x_ql_0 >> (2 * l)) & 0x03030303;
            const int x_qh_k = ((x_qh_0 >> l) << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q3_K + k] = x_qs_k;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + k] = x_qs_k;
#endif // NEW_MMA_AVAILABLE
        }
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE / 8);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K* bxi = (const block_q3_K*)x + kbx0 + i * stride;

        const int ksc = threadIdx.x % (WARP_SIZE / 8);

        const int ksc_low = ksc % (QI3_K / 8);
        const int shift_low = 4 * (ksc / (QI3_K / 8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K / 8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

#ifdef NEW_MMA_AVAILABLE
        const int8_t* sc8 = (const int8_t*)&sc;
        const float d = __half2float(std::bit_cast<half>(bxi->d));

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_df[i * MMQ_MMA_TILE_X_K_Q3_K + sizeof(int) * (threadIdx.x % (WARP_SIZE / 8)) + l] = d * sc8[l];
        }
#else
        x_sc[i * (WARP_SIZE / 8) + i / 8 + threadIdx.x % (WARP_SIZE / 8)] = sc;
#endif // NEW_MMA_AVAILABLE
    }

#ifndef NEW_MMA_AVAILABLE
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE) {
        int i = (i0 + threadIdx.y * WARP_SIZE + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K* bxi = (const block_q3_K*)x + kbx0 + i * stride;

        x_df[i] = __half2float(std::bit_cast<half>(bxi->d));
    }
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + txs.qs;
    const int* x_sc = (const int*)x_df + txs.dm;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR3_K * VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int8_t* scales = ((const int8_t*)(x_sc + i * (WARP_SIZE / 8) + i / 8)) + k0 / 4;

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i * (2 * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int* scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F) | // lower 4 bits
        ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);  // upper 2 bits
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q4_K* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + 2 * WARP_SIZE);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
    int* x_sc = (int*)(x_dm + txs.dm);
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K* bxi = x + kbx0 + i * stride;
        const int qs0 = get_int_b4(bxi->qs, threadIdx.x);

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (threadIdx.x / 8) + threadIdx.x % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (threadIdx.x / 8) + threadIdx.x % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = qs0;
#endif // NEW_MMA_AVAILABLE
    }

#ifdef NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 16) {
        int i = (i0 + threadIdx.y * 16 + threadIdx.x / (WARP_SIZE / 16)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i * stride;

        const int* scales = (const int*)bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE / 16);

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t* sc8 = (const uint8_t*)&sc32;
        const uint8_t* m8 = (const uint8_t*)&m32;

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int) * ksc + l] = dm * make_half2(sc8[l], m8[l]);
        }
    }

#else

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + threadIdx.y * QI4_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i * stride;

        x_dm[i] = __tohalf2(bxi->dm);
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE / 8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i * stride + (threadIdx.x % (WARP_SIZE / 8)) / (QI4_K / 8);

        const int* scales = (const int*)bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE / 8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i * (WARP_SIZE / 8) + i / 8 + ksc] = scales8;
    }
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q5_K* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    int* x_qs = (int*)x_tile;
    half2* x_dm = (half2*)(x_qs + txs.qs);
    int* x_sc = (int*)(x_dm + txs.dm);
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K* bxi = x + kbx0 + i * stride;
        const int ky = QR5_K * threadIdx.x;

        const int ql = get_int_b4(bxi->qs, threadIdx.x);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, threadIdx.x % (QI5_K / 4));
        const int qh0 = ((qh >> (2 * (threadIdx.x / (QI5_K / 4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (threadIdx.x / (QI5_K / 4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K / 2) + threadIdx.x % (QI5_K / 4) + 0;
        const int kq1 = ky - ky % (QI5_K / 2) + threadIdx.x % (QI5_K / 4) + QI5_K / 4;

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
#else
        x_qs[i * (2 * WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_qs[i * (2 * WARP_SIZE + 1) + kq1] = ql1 | qh1;
#endif // NEW_MMA_AVAILABLE
    }

#ifdef NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 16) {
        int i = (i0 + threadIdx.y * 16 + threadIdx.x / (WARP_SIZE / 16)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K* bxi = (const block_q5_K*)x + kbx0 + i * stride;

        const int* scales = (const int*)bxi->scales;
        const int ksc = threadIdx.x % (WARP_SIZE / 16);

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uint8_t* sc8 = (const uint8_t*)&sc32;
        const uint8_t* m8 = (const uint8_t*)&m32;

        const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int) * ksc + l] = dm * make_half2(sc8[l], m8[l]);
        }
    }

#else

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + threadIdx.y * QI5_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K* bxi = (const block_q5_K*)x + kbx0 + i * stride;

        x_dm[i] = __tohalf2(bxi->dm);
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE / 8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K* bxi = (const block_q5_K*)x + kbx0 + i * stride;

        const int* scales = (const int*)bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE / 8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i * (WARP_SIZE / 8) + i / 8 + ksc] = scales8;
    }
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {
#ifdef NEW_MMA_AVAILABLE

    typedef mma_A_I16K4<int> mma_A;
    typedef mma_B_J8K4<int>  mma_B;
    typedef mma_C_I16J8<int> mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J * MMQ_TILE_Y_K);

    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + WARP_SIZE * 2;
    const int* x_sc = (const int*)x_df + WARP_SIZE / QI6_K;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;

    const int i0 = (threadIdx.y / ntx) * (ntx * mma_A::I);

    mma_A   A[ntx][8];
    int   scA[ntx][mma_C::ne / 2][8];
    float  dA[ntx][mma_C::ne / 2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            const int k0 = k00 + k01;

            A[n][k01 / 4 + 0].load_ldmatrix(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q6_K + (k0 + 0), MMQ_MMA_TILE_X_K_Q6_K);
            A[n][k01 / 4 + 1].load_ldmatrix(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q6_K + (k0 + mma_A::K), MMQ_MMA_TILE_X_K_Q6_K);
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < mma_C::ne / 2; ++l) {
                const int i = i0 + n * mma_C::I + mma_C::get_i(2 * l);

                const int      sc_packed = x_sc[i * MMQ_MMA_TILE_X_K_Q6_K + k0 / 16];
                const int8_t* sc = (const int8_t*)&sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01 / 4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne / 2; ++l) {
            const int i = i0 + n * mma_C::I + mma_C::get_i(2 * l);

            dA[n][l] = x_df[i * MMQ_MMA_TILE_X_K_Q6_K];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * mma_C::J) {
        float tmp[ntx][mma_C::ne] = { {0.0f} };

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += 8) {
            mma_B B[2];
            float dB[mma_C::ne / 2];

            // Here load_generic is faster than load_ldmatrix.
            B[0].load_generic(y_qs + j0 * MMQ_TILE_Y_K + 0 + k01, MMQ_TILE_Y_K);
            B[1].load_generic(y_qs + j0 * MMQ_TILE_Y_K + mma_B::K + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < mma_C::ne / 2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C C[2];
                C[0].mma(A[n][k01 / 4 + 0], B[0]);
                C[1].mma(A[n][k01 / 4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l] * scA[n][l / 2][k01 / 4 + 0] + C[1].x[l] * scA[n][l / 2][k01 / 4 + 1]) * dB[l % 2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                sum[(j0 / mma_C::J + n) * mma_C::ne + l] += tmp[n][l] * dA[n][l / 2];
            }
        }
    }
#else
    (void)(x); (void)(y); (void)(sum);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq3_xxs* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI3_XXS / 2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / (QI3_XXS / 2)) {
        int i = i0 + threadIdx.y * (2 * WARP_SIZE / QI3_XXS) + threadIdx.x / (QI3_XXS / 2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_xxs* bxi = x + kbx0 + i * stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2 * kqsx + 0), get_int_b2(bxi->qs, 2 * kqsx + 1));
        const uint8_t* q3 = (const uint8_t*)&q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K / 16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2 * l + 0]], iq3xxs_grid[q3[2 * l + 1]]);

            const int* signs = (const int*)(ksigns64 + ((aux32 >> (7 * l)) & 0x7F));

            const int grid_l = __vsub4(grid_pos.x ^ signs[0], signs[0]);
            const int grid_h = __vsub4(grid_pos.y ^ signs[1], signs[1]);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 8 * kqsx + (2 * l + 1)] = grid_h;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = aux32 >> 28;
        const float d = __half2float(std::bit_cast<half>(bxi->d));
#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = (ls * d + d / 2) / 2;
#else
        x_df[i * (WARP_SIZE / 4) + i / 4 + kqsx] = (ls * d + d / 2) / 2;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + txs.qs;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i * (2 * WARP_SIZE + 1) + k0],
                    &y_qs[j * MMQ_TILE_Y_K + k01],
                    &x_df[i * (2 * WARP_SIZE * 2 / QI8_0) + i / (QI8_0 / 4) + k0 / (QI8_0 / 2)],
                    y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq2_xxs* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_XXS / 2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / (QI2_XXS / 2)) {
        int i = i0 + threadIdx.y * (2 * WARP_SIZE / QI2_XXS) + threadIdx.x / (QI2_XXS / 2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xxs* bxi = x + kbx0 + i * stride;

        const int q2 = get_int_b2(bxi->qs, 2 * kqsx + 0);
        const uint8_t* aux8 = (const uint8_t*)&q2;
        const uint32_t aux32 = get_int_b2(bxi->qs, 2 * kqsx + 1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const int* grid_pos = (const int*)(iq2xxs_grid + aux8[l]);
            const int signs_packed = ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];

            const int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
            const int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);

            const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
            const int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 8 * kqsx + (2 * l + 0)] = grid0;
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 8 * kqsx + (2 * l + 1)] = grid1;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 0)] = grid0;
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 1)] = grid1;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = aux32 >> 28;
        const float d = __half2float(std::bit_cast<half>(bxi->d));
#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = (ls * d + d / 2) / 4;
#else
        x_df[i * (WARP_SIZE / 4) + i / 4 + kqsx] = (ls * d + d / 2) / 4;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_q6_K* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
    int* x_sc = (int*)(x_df + WARP_SIZE / QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
    int* x_sc = (int*)(x_df + txs.dm);
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K* bxi = x + kbx0 + i * stride;

        const int ql = get_int_b2(bxi->ql, threadIdx.x);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K / 4) * (threadIdx.x / (QI6_K / 2)) + threadIdx.x % (QI6_K / 4));
        const int qh0 = ((qh >> ((threadIdx.x & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 = (qh >> ((threadIdx.x & 0x08) >> 2)) & 0x30303030;

        const int kq0 = 2 * threadIdx.x - threadIdx.x % (QI6_K / 2) + 0;
        const int kq1 = 2 * threadIdx.x - threadIdx.x % (QI6_K / 2) + QI6_K / 2;

#ifdef NEW_MMA_AVAILABLE
        x_qs[i * MMQ_MMA_TILE_X_K_Q6_K + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i * MMQ_MMA_TILE_X_K_Q6_K + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i * (2 * WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i * (2 * WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // NEW_MMA_AVAILABLE
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + threadIdx.y * QI6_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K* bxi = (const block_q6_K*)x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q6_K + kbxd] = __half2float(std::bit_cast<half>(bxi->d));
#else
        x_df[i * (WARP_SIZE / QI6_K) + i / QI6_K + kbxd] = __half2float(std::bit_cast<half>(bxi->d));
#endif // NEW_MMA_AVAILABLE
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE / 8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K* bxi = (const block_q6_K*)x + kbx0 + i * stride + (threadIdx.x % (WARP_SIZE / 8)) / 4;

#ifdef NEW_MMA_AVAILABLE
        x_sc[i * MMQ_MMA_TILE_X_K_Q6_K + threadIdx.x % (WARP_SIZE / 8)] = get_int_b2(bxi->scales, threadIdx.x % (QI6_K / 8));
#else
        x_sc[i * (WARP_SIZE / 8) + i / 8 + threadIdx.x % (WARP_SIZE / 8)] = get_int_b2(bxi->scales, threadIdx.x % (QI6_K / 8));
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + txs.qs;
    const int* x_sc = (const int*)x_dm + txs.dm;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t* sc = (const uint8_t*)&x_sc[i * (WARP_SIZE / 8) + i / 8 + k0 / 32] + 2 * (k01 / 16);

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i * (WARP_SIZE + 1) + k0 / 2], &y_qs[j * MMQ_TILE_Y_K + k01], sc, sc + 8,
                    x_dm[i], &y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + txs.qs;
    const int* x_sc = (const int*)x_dm + txs.dm;
    const int* y_qs = (const int*)y + 4;
    const half2* y_ds = (const half2*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR5_K * VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t* sc = ((const uint8_t*)&x_sc[i * (WARP_SIZE / 8) + i / 8 + k00 / 32]) + 2 * (k01 / 16);

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i * (QR5_K * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k01], sc, sc + 8,
                    x_dm[i], &y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq2_xs* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_XS / 2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / (QI2_XS / 2)) {
        int i = i0 + threadIdx.y * (2 * WARP_SIZE / QI2_XS) + threadIdx.x / (QI2_XS / 2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_xs* bxi = x + kbx0 + i * stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2 * kqsx + 0), get_int_b2(bxi->qs, 2 * kqsx + 1));
        const uint16_t* q2 = (const uint16_t*)&q2_packed;

#pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint32_t* grid_pos = (const uint32_t*)(iq2xs_grid + (q2[l] & 0x000001FF));
            const uint32_t* signs = (const uint32_t*)(ksigns64 + (q2[l] >> 9));

            const int grid_l = __vsub4(grid_pos[0] ^ signs[0], signs[0]);
            const int grid_h = __vsub4(grid_pos[1] ^ signs[1], signs[1]);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q3_K + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * MMQ_MMA_TILE_X_K_Q3_K + 8 * kqsx + (2 * l + 1)] = grid_h;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = bxi->scales[kqsx];
        const float d = __half2float(std::bit_cast<half>(bxi->d));
#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q3_K + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
        x_df[i * MMQ_MMA_TILE_X_K_Q3_K + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
#else
        x_df[i * (2 * WARP_SIZE * 2 / QI8_0) + i / (QI8_0 / 4) + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
        x_df[i * (2 * WARP_SIZE * 2 / QI8_0) + i / (QI8_0 / 4) + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq2_s* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, mmq_y);
    int* x_qs = (int*)x_tile;
    float* x_df = (float*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % (QI2_S / 2);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / (QI2_S / 2)) {
        int i = i0 + threadIdx.y * (2 * WARP_SIZE / QI2_S) + threadIdx.x / (QI2_S / 2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_s* bxi = x + kbx0 + i * stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t* qs = (const uint8_t*)&qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K / 32 + kqsx);
        const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int* grid_pos = (const int*)(iq2s_grid + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q3_K + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * MMQ_MMA_TILE_X_K_Q3_K + 8 * kqsx + (2 * l + 1)] = grid_h;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 0)] = grid_l;
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 1)] = grid_h;
#endif // NEW_MMA_AVAILABLE
        }

        const int ls = bxi->scales[kqsx];
        const float d = __half2float(std::bit_cast<half>(bxi->d));
#ifdef NEW_MMA_AVAILABLE
        x_df[i * MMQ_MMA_TILE_X_K_Q3_K + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
        x_df[i * MMQ_MMA_TILE_X_K_Q3_K + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
#else
        x_df[i * (2 * WARP_SIZE * 2 / QI8_0) + i / (QI8_0 / 4) + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
        x_df[i * (2 * WARP_SIZE * 2 / QI8_0) + i / (QI8_0 / 4) + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles(
    const block_iq1_s* __restrict__ x, int* __restrict__ x_tile, const int& kbx0, const int& i_max, const int& stride) {

#ifdef NEW_MMA_AVAILABLE
    int* x_qs = (int*)x_tile;
    half2* x_ds = (half2*)(x_qs + WARP_SIZE * 2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, mmq_y);
    int* x_qs = (int*)x_tile;
    half2* x_ds = (half2*)(x_qs + txs.qs);
#endif // NEW_MMA_AVAILABLE

    const int kqsx = threadIdx.x % QI1_S;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE / QI1_S) {
        int i = i0 + threadIdx.y * (WARP_SIZE / QI1_S) + threadIdx.x / QI1_S;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq1_s* bxi = x + kbx0 + i * stride;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t* qs = (const uint8_t*)&qs_packed;

        const int qh = bxi->qh[kqsx];

#pragma unroll
        for (int l = 0; l < QR1_S / 2; ++l) {
            const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3 * l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

#ifdef NEW_MMA_AVAILABLE
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 8 * kqsx + (2 * l + 0)] = grid0;
            x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 8 * kqsx + (2 * l + 1)] = grid1;
#else
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 0)] = grid0;
            x_qs[i * (2 * WARP_SIZE + 1) + 8 * kqsx + (2 * l + 1)] = grid1;
#endif // NEW_MMA_AVAILABLE
        }

        const float  d1q = __half2float(std::bit_cast<half>(bxi->d)) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);

#ifdef NEW_MMA_AVAILABLE
        x_ds[i * MMQ_MMA_TILE_X_K_Q8_1 + kqsx] = make_half2(d1q, d1q * delta);
#else
        x_ds[i * (WARP_SIZE / 4) + i / 4 + kqsx] = make_half2(d1q, d1q * delta);
#endif // NEW_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    const int* x_qs = (const int*)x;
    const float* x_df = (const float*)x_qs + txs.qs;
    const int* x_sc = (const int*)x_df + txs.dm;
    const int* y_qs = (const int*)y + 4;
    const float* y_df = (const float*)y;

    // #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR6_K * VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const int8_t* sc = ((const int8_t*)&x_sc[i * (WARP_SIZE / 8) + i / 8 + k0 / 16]);

                sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i * (QR6_K * WARP_SIZE + 1) + k0], &y_qs[j * MMQ_TILE_Y_K + k01], sc,
                    x_df[i * (WARP_SIZE / QI6_K) + i / QI6_K], &y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps, bool need_check, ggml_type type>
struct mmq_type_traits;

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr = VDR_Q4_0_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_DS4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr = VDR_Q4_1_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_0> {
    static constexpr int              vdr = VDR_Q5_0_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_1> {
    static constexpr int              vdr = VDR_Q5_1_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q8_0> {
    static constexpr int              vdr = VDR_Q8_0_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q2_K> {
    static constexpr int              vdr = VDR_Q2_K_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q3_K> {
    static constexpr int              vdr = VDR_Q3_K_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr = VDR_Q4_K_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_K> {
    static constexpr int              vdr = VDR_Q5_K_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_K> {
    static constexpr int              vdr = VDR_Q6_K_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_XXS> {
    static constexpr int              vdr = VDR_IQ2_XXS_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_XS> {
    static constexpr int              vdr = VDR_IQ2_XS_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_S> {
    static constexpr int              vdr = VDR_IQ2_S_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_XXS> {
    static constexpr int              vdr = VDR_IQ3_XXS_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ3_S> {
    static constexpr int              vdr = VDR_IQ3_S_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ1_S> {
    static constexpr int              vdr = VDR_IQ1_S_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_NL> {
    static constexpr int              vdr = VDR_IQ4_NL_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_XS> {
    static constexpr int              vdr = VDR_IQ4_XS_Q8_1_MMQ;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
    const float* __restrict__ sum, const int32_t* __restrict__ ids_dst, float* __restrict__ dst,
    const int stride, const int i_max, const int j_max) {
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[ids_dst[j] * stride + i] = sum[(j0 / nwarps) * (mmq_y / WARP_SIZE) + i0 / WARP_SIZE];
        }
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
    const float* __restrict__ sum, const int* __restrict__ ids_dst, float* __restrict__ dst,
    const int stride, const int i_max, const int j_max) {
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
#ifdef NEW_MMA_AVAILABLE
    static_assert(nwarps * tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");
#endif // NEW_MMA_AVAILABLE

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n * tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[ids_dst[j] * stride + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

template <ggml_type type, typename src_t, int mmq_x, int nwarps, bool need_check, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
    const char* __restrict__ x, const int offset_x, const int* __restrict__ y,
    const int* __restrict__ ids_dst, float* __restrict__ dst, float* __restrict__ tmp_fixup,
    const int stride_row_x, const int ncols_y, const int stride_col_dst,
    const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              qk = ggml_cuda_type_traits<src_t>::qk;
    constexpr int              mmq_y = get_mmq_y_device();

    extern __shared__ int data_mul_mat_q[];
    int* tile_y = data_mul_mat_q + mmq_x;
    int* tile_x = tile_y + GGML_PAD1(mmq_x * (WARP_SIZE + WARP_SIZE / QI8_1), nwarps * WARP_SIZE);

#ifdef NEW_MMA_AVAILABLE
    constexpr vec_dot_mmq_t    vec_dot = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_mma;
#else
    constexpr vec_dot_mmq_t    vec_dot = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_dp4a;
#endif // NEW_MMA_AVAILABLE

    constexpr int blocks_per_iter = MMQ_ITER_K / qk;

    float sum[mmq_x * mmq_y / (nwarps * WARP_SIZE)] = { 0.0f };

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles<mmq_y, nwarps, need_check>((const src_t*)(x), tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

        {
            const int* by0 = y + ncols_y * (kb0 * (qk * sizeof(block_q8_1_mmq) / (4 * QK8_1 * sizeof(int))) + 0 * sizeof(block_q8_1_mmq) / sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * WARP_SIZE) {
                int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int* by0 = y + ncols_y * (kb0 * (qk * sizeof(block_q8_1_mmq) / (4 * QK8_1 * sizeof(int))) + 1 * sizeof(block_q8_1_mmq) / sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * WARP_SIZE) {
                int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 32/*WARP_SIZE*/);

        __syncthreads();
    }

    auto [write_dst, stride, i_max, j_max] = [=]() -> std::tuple<float*, int, int, int> {
        if constexpr (fixup) {
            return { tmp_fixup + blockIdx.x * (mmq_x * mmq_y), mmq_y, mmq_y, mmq_x };
        }
        else {
            return { dst, stride_col_dst, tile_x_max_i, tile_y_max_j };
        }
    }();

    if constexpr (new_mma_available_v) {
        mmq_write_back_mma<mmq_x, mmq_y, nwarps, need_check>(sum, ids_dst, write_dst, stride, i_max, j_max);
    }
    else {
        mmq_write_back_dp4a<mmq_x, mmq_y, nwarps, need_check>(sum, ids_dst, write_dst, stride, i_max, j_max);
    }
}

// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, typename src_t, int mmq_x, int nwarps, bool need_check>
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
__launch_bounds__(WARP_SIZE* nwarps, 2)
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
__launch_bounds__(WARP_SIZE* nwarps, 1)
#else
__launch_bounds__(WARP_SIZE* nwarps, 2)
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
static __global__ void mul_mat_q(
    const char* __restrict__ x, const int* __restrict__ y, const int32_t* __restrict__ ids_dst,
    const int32_t* __restrict__ expert_bounds, float* __restrict__ dst, float* __restrict__ tmp_fixup,
    const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
    const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int sample_ratio, const int nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int qk = ggml_cuda_type_traits<src_t>::qk;
    constexpr int mmq_y = get_mmq_y_device();

    const int ntx = (ncols_dst + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (nrows_x + mmq_y - 1) / mmq_y; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * WARP_SIZE) {
        const int j = j0 + threadIdx.y * WARP_SIZE + threadIdx.x;

        if (j0 + nwarps * WARP_SIZE > mmq_x && j >= mmq_x) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    // On AMD or old CUDA the performance with stream-k was worse, use conventional tiling instead:
#if (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    {
        const int wt = blockIdx.z / nchannels_y;
        const int zt = blockIdx.z - wt * nchannels_y;
        const int jt = blockIdx.y;
        const int it = blockIdx.x;

        // Defaults for regular matrix multiplication:
        int col_low = 0;
        int col_high = ncols_dst;
        int col_diff = ncols_dst;
        int offset_y = wt * stride_sample_y + zt * stride_channel_y;
        int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * mmq_x * stride_col_dst;

        if (ids_dst) {
            col_low = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y = 0;
            offset_dst = 0;

            if (jt * mmq_x >= col_diff) {
                return;
            }

            // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps * WARP_SIZE) {
                const int j = j0 + threadIdx.y * WARP_SIZE + threadIdx.x;

                if (j0 + nwarps * WARP_SIZE > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt * mmq_x + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it * mmq_y;

        const int tile_x_max_i = nrows_x - it * mmq_y - 1;
        const int tile_y_max_j = col_diff - jt * mmq_x - 1;

        const int offset_x = (wt / sample_ratio) * stride_sample_x + (zt / channel_ratio) * stride_channel_x + it * mmq_y * stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile<type, src_t, mmq_x, nwarps, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
                tile_x_max_i, tile_y_max_j, 0, ncols_x / qk);
        return;
    }
#endif // (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA

    const     int64_t blocks_per_ne00 = ncols_x / qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc = (int64_t)blockIdx.x * nsamples_y * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1) * nsamples_y * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;

    kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        int tmp = kbc;
        const int it = tmp / (nsamples_y * nchannels_y * ntx * blocks_per_ne00);
        tmp -= it * (nsamples_y * nchannels_y * ntx * blocks_per_ne00);
        const int wt = tmp / (nchannels_y * ntx * blocks_per_ne00);
        tmp -= wt * (nchannels_y * ntx * blocks_per_ne00);
        const int zt = tmp / (ntx * blocks_per_ne00);
        tmp -= zt * (ntx * blocks_per_ne00);
        const int jt = tmp / blocks_per_ne00;

        // Defaults for regular matrix multiplication:
        int col_low = 0;
        int col_high = ncols_dst;
        int col_diff = ncols_dst;
        int offset_y = wt * stride_sample_y + zt * stride_channel_y;
        int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * mmq_x * stride_col_dst;

        if (ids_dst) {
            col_low = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y = 0;
            offset_dst = 0;

            if (jt * mmq_x >= col_diff) {
                kbc += blocks_per_ne00;
                kbc -= kbc % blocks_per_ne00;

                kb0_start = 0;
                kb0_stop = min(blocks_per_ne00, kbc_stop - kbc);

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps * WARP_SIZE) {
                const int j = j0 + threadIdx.y * WARP_SIZE + threadIdx.x;

                if (j0 + nwarps * WARP_SIZE > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt * mmq_x + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it * mmq_y;

        const int tile_x_max_i = nrows_x - it * mmq_y - 1;
        const int tile_y_max_j = col_diff - jt * mmq_x - 1;

        const int offset_x = (wt / sample_ratio) * stride_sample_x + (zt / channel_ratio) * stride_channel_x + it * mmq_y * stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<type, src_t, mmq_x, nwarps, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
                tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;

        kb0_start = 0;
        kb0_stop = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = kbc;
    const int it = tmp / (nsamples_y * nchannels_y * ntx * blocks_per_ne00);
    tmp -= it * (nsamples_y * nchannels_y * ntx * blocks_per_ne00);
    const int wt = tmp / (nchannels_y * ntx * blocks_per_ne00);
    tmp -= wt * (nchannels_y * ntx * blocks_per_ne00);
    const int zt = tmp / (ntx * blocks_per_ne00);
    tmp -= zt * (ntx * blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    // Defaults for regular matrix multiplication:
    int col_low = 0;
    int col_high = ncols_dst;
    int col_diff = ncols_dst;
    int offset_y = wt * stride_sample_y + zt * stride_channel_y;
    int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * mmq_x * stride_col_dst;

    if (ids_dst) {
        col_low = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y = 0;
        offset_dst = 0;

        if (jt * mmq_x >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps * WARP_SIZE) {
            const int j = j0 + threadIdx.y * WARP_SIZE + threadIdx.x;

            if (j0 + nwarps * WARP_SIZE > mmq_x && j >= mmq_x) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it * mmq_y;

    const int tile_x_max_i = nrows_x - it * mmq_y - 1;
    const int tile_y_max_j = col_diff - jt * mmq_x - 1;

    const int offset_x = (wt / sample_ratio) * stride_sample_x + (zt / channel_ratio) * stride_channel_x + it * mmq_y * stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<type, src_t, mmq_x, nwarps, need_check, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
            tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

template <ggml_type type, typename src_t, int mmq_x, int nwarps, bool need_check>
static __global__ void mul_mat_q_stream_k_fixup(
    const int32_t* ids_dst, const int32_t* expert_bounds, float* __restrict__ dst, const float* __restrict__ tmp_last_tile,
    const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_col_dst,
    const int nchannels_y, const int stride_channel_dst, const int nsamples_y, const int stride_sample_dst) {
    constexpr int     mmq_y = get_mmq_y_device();
    constexpr int     qk = ggml_cuda_type_traits<src_t>::qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;
    const     int64_t blocks_per_ne00 = ncols_x / qk;

    float sum[mmq_x * mmq_y / (nwarps * WARP_SIZE)] = { 0.0f };

    const int ntx = (ncols_dst + mmq_x - 1) / mmq_x;
    const int nty = (nrows_x + mmq_y - 1) / mmq_y;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc0 = (int64_t)bidx0 * nsamples_y * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;
    int64_t kbc0_stop = (int64_t)(bidx0 + 1) * nsamples_y * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;

    kbc0 -= (kbc0 % blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= (kbc0_stop % blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % blocks_per_ne00 == 0;
    const bool did_not_write_last = kbc0 / blocks_per_ne00 == kbc0_stop / blocks_per_ne00 && kbc0_stop % blocks_per_ne00 != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int64_t bidx = bidx0 - 1;
    int64_t kbc_stop = kbc0;
    while (true) {
        int64_t kbc = bidx * nsamples_y * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;
        kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[(j0 / nwarps) * (mmq_y / WARP_SIZE) + i0 / WARP_SIZE] += tmp_last_tile[bidx * (mmq_x * mmq_y) + j * mmq_y + i];
            }
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % blocks_per_ne00 == 0 || kbc / blocks_per_ne00 < kbc0 / blocks_per_ne00) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = kbc0;
    const int it = tmp / (nsamples_y * nchannels_y * ntx * blocks_per_ne00);
    tmp -= it * (nsamples_y * nchannels_y * ntx * blocks_per_ne00);
    const int wt = tmp / (nchannels_y * ntx * blocks_per_ne00);
    tmp -= wt * (nchannels_y * ntx * blocks_per_ne00);
    const int zt = tmp / (ntx * blocks_per_ne00);
    tmp -= zt * (ntx * blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    if (!ids_dst) {
        const int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * mmq_x * stride_col_dst + it * mmq_y;
        dst += offset_dst;

        const int i_max = nrows_x - it * mmq_y - 1;
        const int j_max = ncols_dst - jt * mmq_x - 1;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                if (need_check && i > i_max) {
                    continue;
                }

                dst[j * stride_col_dst + i] += sum[(j0 / nwarps) * (mmq_y / WARP_SIZE) + i0 / WARP_SIZE];
            }
        }
        return;
    }

    __shared__ int ids_dst_shared[mmq_x];
    const int col_low = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y * WARP_SIZE + threadIdx.x; j < mmq_x; j += nwarps * WARP_SIZE) {
        ids_dst_shared[j] = ids_dst[col_low + j];
    }
    __syncthreads();

    const int offset_dst = it * mmq_y;
    dst += offset_dst;

    const int i_max = nrows_x - it * mmq_y - 1;
    const int j_max = col_diff - jt * mmq_x - 1;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[ids_dst_shared[j] * stride_col_dst + i] += sum[(j0 / nwarps) * (mmq_y / WARP_SIZE) + i0 / WARP_SIZE];
        }
    }
}

#define MMQ_MMA_TILE_X_K_Q8_0 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q8_1 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0                 + 4)
#define MMQ_MMA_TILE_X_K_Q2_K (2*WARP_SIZE + WARP_SIZE                         + 4)
#define MMQ_MMA_TILE_X_K_Q3_K (2*WARP_SIZE + WARP_SIZE/2                       + 4)
#define MMQ_MMA_TILE_X_K_Q6_K (2*WARP_SIZE + WARP_SIZE/QI6_K     + WARP_SIZE/8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q6_K % 8 == 4, "Wrong padding.");

static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q4_1 ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q5_0 ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q5_1 ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q8_0 ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_Q2_K ? MMQ_MMA_TILE_X_K_Q2_K :
        type == GGML_TYPE_Q3_K ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_Q4_K ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q5_K ? MMQ_MMA_TILE_X_K_Q8_1 :
        type == GGML_TYPE_Q6_K ? MMQ_MMA_TILE_X_K_Q6_K :
        type == GGML_TYPE_IQ2_XXS ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ2_XS ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_IQ2_S ? MMQ_MMA_TILE_X_K_Q3_K :
        type == GGML_TYPE_IQ3_XXS ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ3_S ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ1_S ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ4_XS ? MMQ_MMA_TILE_X_K_Q8_0 :
        type == GGML_TYPE_IQ4_NL ? MMQ_MMA_TILE_X_K_Q8_0 :
        0;
}

template <ggml_type type>
static int mmq_get_shmem(const int mmq_x, const int mmq_y, const int cc) {
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const int shmem_x = new_mma_available(cc) ? mmq_y * mmq_tile_x_k * sizeof(int) : txs.qs * sizeof(int) + txs.dm * sizeof(half2) + txs.sc * sizeof(int);
    const int shmem_y = mmq_x * sizeof(block_q8_1_mmq);
    return shmem_x + GGML_PAD1(shmem_y, MMQ_NWARPS * WARP_SIZE * sizeof(int));
}

template<ggml_type type>
static size_t mmq_get_nbytes_shared(const int mmq_x, const int mmq_y, const int cc) {
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const size_t nbs_ids = mmq_x * sizeof(int);
    const size_t nbs_x = new_mma_available(cc) ? mmq_y * mmq_tile_x_k * sizeof(int) : txs.qs * sizeof(int) + txs.dm * sizeof(half2) + txs.sc * sizeof(int);
    const size_t nbs_y = mmq_x * sizeof(block_q8_1_mmq);
    return nbs_ids + nbs_x + GGML_PAD1(nbs_y, MMQ_NWARPS * WARP_SIZE * sizeof(int));
}

template <ggml_type type, typename src_t, int mmq_x>
static void launch_mul_mat_q(ggml_cuda_pool& pool, const mmq_args& args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int mmq_y = get_mmq_y_host(cc);

    const dim3 block_dims(WARP_SIZE, MMQ_NWARPS, 1);

    const int nbytes_shared = mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc);

    CUDA_SET_SHARED_MEMORY_LIMIT(reinterpret_cast<const void*>(mul_mat_q<type, src_t, mmq_x, MMQ_NWARPS, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT(reinterpret_cast<const void*>(mul_mat_q<type, src_t, mmq_x, MMQ_NWARPS, true>), nbytes_shared);

    const int nty = (args.nrows_x + mmq_y - 1) / mmq_y;
    const int ntx = (args.ncols_dst + mmq_x - 1) / mmq_x;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y % args.nsamples_x == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio = args.nsamples_y / args.nsamples_x;

    if (!args.use_stream_k) {
        if (args.nrows_x % mmq_y == 0) {
            constexpr bool need_check = false;
            mul_mat_q<type, src_t, mmq_x, MMQ_NWARPS, need_check> << <block_nums_xy_tiling, block_dims, nbytes_shared, stream >> >
                (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                    args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                    channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                    sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);
        }
        else {
            constexpr bool need_check = true;
            mul_mat_q<type, src_t, mmq_x, MMQ_NWARPS, need_check> << <block_nums_xy_tiling, block_dims, nbytes_shared, stream >> >
                (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                    args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                    channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                    sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);
        }
        return;
    }

    const dim3 block_nums_stream_k(nsm, 1, 1);
    const bool fixup_needed = ntx * nty * ntzw % nsm != 0;

    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) {
        tmp_fixup.alloc(block_nums_stream_k.x * mmq_x * mmq_y);
    }

    if (args.nrows_x % mmq_y == 0) {
        constexpr bool need_check = false;

        mul_mat_q<type, src_t, mmq_x, MMQ_NWARPS, need_check> << <block_nums_stream_k, block_dims, nbytes_shared, stream >> >
            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);

        if (!fixup_needed) {
            return;
        }

        mul_mat_q_stream_k_fixup<type, src_t, mmq_x, MMQ_NWARPS, need_check> << <block_nums_stream_k, block_dims, 0, stream >> >
            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.ncols_x, args.nrows_x, args.ncols_dst,
                args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y, args.stride_sample_dst);
    }
    else {
        constexpr bool need_check = true;

        mul_mat_q<type, src_t, mmq_x, MMQ_NWARPS, need_check> << <block_nums_stream_k, block_dims, nbytes_shared, stream >> >
            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
                args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);

        if (!fixup_needed) {
            return;
        }

        mul_mat_q_stream_k_fixup<type, src_t, mmq_x, MMQ_NWARPS, need_check> << <block_nums_stream_k, block_dims, 0, stream >> >
            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.ncols_x, args.nrows_x, args.ncols_dst,
                args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y, args.stride_sample_dst);
    }
}

static int mmq_get_granularity_host(const int mmq_x, const int cc) {
    return new_mma_available(cc) && mmq_x >= 48 ? 16 : 8;
}

template <ggml_type type, typename src_t>
void mul_mat_q_case(ggml_cuda_pool& pool, const mmq_args& args, cudaStream_t stream) {
    const int    id = ggml_cuda_get_device();
    const int    cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc);

    int mmq_x_best = 0;
    int ntiles_x_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && ntiles_x_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);

        if (mmq_x % granularity != 0 || mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ncols_y + mmq_x - 1) / mmq_x;

        if (ntiles_x < ntiles_x_best) {
            mmq_x_best = mmq_x;
            ntiles_x_best = ntiles_x;
        }
    }

    switch (mmq_x_best) {
    case   8:
        launch_mul_mat_q<type, src_t, 8>(pool, args, stream);
        break;
    case  16:
        launch_mul_mat_q<type, src_t, 16>(pool, args, stream);
        break;
    case  24:
        launch_mul_mat_q<type, src_t, 24>(pool, args, stream);
        break;
    case  32:
        launch_mul_mat_q<type, src_t, 32>(pool, args, stream);
        break;
    case  40:
        launch_mul_mat_q<type, src_t, 40>(pool, args, stream);
        break;
    case  48:
        launch_mul_mat_q<type, src_t, 48>(pool, args, stream);
        break;
    case  56:
        launch_mul_mat_q<type, src_t, 56>(pool, args, stream);
        break;
    case  64:
        launch_mul_mat_q<type, src_t, 64>(pool, args, stream);
        break;
    case  72:
        launch_mul_mat_q<type, src_t, 72>(pool, args, stream);
        break;
    case  80:
        launch_mul_mat_q<type, src_t, 80>(pool, args, stream);
        break;
    case  88:
        launch_mul_mat_q<type, src_t, 88>(pool, args, stream);
        break;
    case  96:
        launch_mul_mat_q<type, src_t, 96>(pool, args, stream);
        break;
    case 104:
        launch_mul_mat_q<type, src_t, 104>(pool, args, stream);
        break;
    case 112:
        launch_mul_mat_q<type, src_t, 112>(pool, args, stream);
        break;
    case 120:
        launch_mul_mat_q<type, src_t, 120>(pool, args, stream);
        break;
    case 128:
        launch_mul_mat_q<type, src_t, 128>(pool, args, stream);
        break;
    default:
        fprintf(stderr, "mmq_x_best=%d\n", mmq_x_best);
        GGML_ABORT("fatal error");
        break;
    }
}

void ggml_cuda_mul_mat_q_switch_type(ggml_cuda_pool& pool, const mmq_args& args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, block_q4_0>(pool, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, block_q4_1>(pool, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, block_q5_0>(pool, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, block_q5_1>(pool, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, block_q8_0>(pool, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, block_q2_K>(pool, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, block_q3_K>(pool, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, block_q4_K>(pool, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, block_q5_K>(pool, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, block_q6_K>(pool, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, block_iq2_xxs>(pool, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, block_iq2_xs>(pool, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, block_iq2_s>(pool, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, block_iq3_xxs>(pool, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, block_iq3_s>(pool, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, block_iq1_s>(pool, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, block_iq4_xs>(pool, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, block_iq4_nl>(pool, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

