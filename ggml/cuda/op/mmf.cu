#include <assert.h>
#include "cuda_func.h"
#include "../common.h"
#include "mmf.cuh"
#include "mma.cuh"

#define GGML_ABORT(...)

void mul_mat_f_cuda(const mul_mat_f_context* ctx, cudaStream_t stream)
{
    switch (ctx->src0_type) {
    case internal::GGML_TYPE_F32: {
        const float* src0_d = (const float*)ctx->src0_d;
        constexpr int vals_per_T = 1;
        mul_mat_f_switch_cols_per_block(
            src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00 / vals_per_T,
            ctx->ne01, ctx->ncols_dst, ctx->s01 / vals_per_T, ctx->stride_col_y / vals_per_T,
            ctx->stride_col_dst,  ctx->ids_s0, ctx->ids_s1, ctx->ne02, ctx->nchannels_y,
            ctx->nchannels_dst, ctx->s02 / vals_per_T, ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03 / vals_per_T, ctx->s13, ctx->s3, stream, ctx->ids_info_ptr);
    } break;
    case internal::GGML_TYPE_F16: {
        const half2* src0_d = (const half2*)ctx->src0_d;
        constexpr int vals_per_T = 2;
        mul_mat_f_switch_cols_per_block(
            src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00 / vals_per_T,
            ctx->ne01, ctx->ncols_dst, ctx->s01 / vals_per_T, ctx->stride_col_y / vals_per_T,
            ctx->stride_col_dst, ctx->ids_s0, ctx->ids_s1, ctx->ne02, ctx->nchannels_y,
            ctx->nchannels_dst, ctx->s02 / vals_per_T, ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03 / vals_per_T, ctx->s13, ctx->s3, stream, ctx->ids_info_ptr);
    } break;
    case internal::GGML_TYPE_BF16: {
        const nv_bfloat162* src0_d = (const nv_bfloat162*)ctx->src0_d;
        constexpr int vals_per_T = 2;
        mul_mat_f_switch_cols_per_block(
            src0_d, ctx->src1_d, ctx->ids_d, ctx->dst_d, ctx->ne00 / vals_per_T,
            ctx->ne01, ctx->ncols_dst, ctx->s01 / vals_per_T, ctx->stride_col_y / vals_per_T,
            ctx->stride_col_dst, ctx->ids_s0, ctx->ids_s1, ctx->ne02, ctx->nchannels_y,
            ctx->nchannels_dst, ctx->s02 / vals_per_T, ctx->stride_channel_y, ctx->stride_channel_dst,
            ctx->ne03, ctx->ne3, ctx->s03 / vals_per_T, ctx->s13, ctx->s3, stream, ctx->ids_info_ptr);
    } break;
    default:
        GGML_ABORT("unsupported type: %s", internal::GGML_TYPE_name(src0->type));
    }
}