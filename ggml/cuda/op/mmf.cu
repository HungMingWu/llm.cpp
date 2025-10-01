#include <assert.h>
#include "cuda_func.h"
#include "../common.h"
#include "mmf.cuh"
#include "mma.cuh"
#include "internal_ds.h"

#define GGML_ABORT(...)

bool ggml_cuda_should_use_mmf(enum ggml_type type, 
    size_t type_size, int cc, int warp_size, const int64_t* src0_ne, int64_t src1_ncols, bool mul_mat_id) {

    if (src0_ne[0] % (warp_size * (4 / type_size)) != 0) {
        return false;
    }
    if (src0_ne[1] % MMF_ROWS_PER_BLOCK != 0) {
        return false;
    }

    if (mul_mat_id) {
        if (type == GGML_TYPE_F32 && src1_ncols > 32) {
            return false;
        }
        if ((type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) && src1_ncols > 64) {
            return false;
        }
    }
    else {
        if (src1_ncols > 16) {
            return false;
        }
    }

    switch (type) {
    case GGML_TYPE_F32:
        return ampere_mma_available(cc);
    case GGML_TYPE_F16:
        return turing_mma_available(cc);
    case GGML_TYPE_BF16:
        return ampere_mma_available(cc);
    default:
        return false;
    }
}

void mul_mat_f_cuda(const mul_mat_f_context* ctx, cudaStream_t stream)
{
    switch (ctx->src0_type) {
    case GGML_TYPE_F32: {
        const float* src0_d = (const float*)ctx->src0_d;
        constexpr int vals_per_T = 1;
        mul_mat_f_switch_cols_per_block(
            src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00 / vals_per_T,
            ctx->ne01, ctx->ncols_dst, ctx->s01 / vals_per_T, ctx->stride_col_y / vals_per_T,
            ctx->stride_col_dst,  ctx->ids_s0, ctx->ids_s1, ctx->ne02, ctx->nchannels_y,
            ctx->nchannels_dst, ctx->s02 / vals_per_T, ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03 / vals_per_T, ctx->s13, ctx->s3, stream);
    } break;
    case GGML_TYPE_F16: {
        const half2* src0_d = (const half2*)ctx->src0_d;
        constexpr int vals_per_T = 2;
        mul_mat_f_switch_cols_per_block(
            src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00 / vals_per_T,
            ctx->ne01, ctx->ncols_dst, ctx->s01 / vals_per_T, ctx->stride_col_y / vals_per_T,
            ctx->stride_col_dst, ctx->ids_s0, ctx->ids_s1, ctx->ne02, ctx->nchannels_y,
            ctx->nchannels_dst, ctx->s02 / vals_per_T, ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03 / vals_per_T, ctx->s13, ctx->s3, stream);
    } break;
    case GGML_TYPE_BF16: {
        const nv_bfloat162* src0_d = (const nv_bfloat162*)ctx->src0_d;
        constexpr int vals_per_T = 2;
        mul_mat_f_switch_cols_per_block(
            src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00 / vals_per_T,
            ctx->ne01, ctx->ncols_dst, ctx->s01 / vals_per_T, ctx->stride_col_y / vals_per_T,
            ctx->stride_col_dst, ctx->ids_s0, ctx->ids_s1, ctx->ne02, ctx->nchannels_y,
            ctx->nchannels_dst, ctx->s02 / vals_per_T, ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03 / vals_per_T, ctx->s13, ctx->s3, stream);
    } break;
    default:
        GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}