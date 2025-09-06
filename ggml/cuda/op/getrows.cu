#define GGML_ASSERT(...)
#include <bit>
#include "internal_ds.h"
#include "cuda_func.h"
#include "block.h"
#include "common.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

static constexpr size_t CUDA_GET_ROWS_BLOCK_SIZE = 256;

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
    const src0_t* __restrict__ src0, const int32_t* __restrict__ src1, dst_t* __restrict__ dst,
    const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
    /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
    /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
    /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
    const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        for (int64_t i00 = blockIdx.y * blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y * blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 = blockIdx.x;
            const int i11 = z / ne12; // TODO fastdiv
            const int i12 = z % ne12;

            if (i00 >= ne00) {
                return;
            }

            const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

            dst_t* dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;
            const src0_t* src0_row = (const src0_t*)((const char*)src0 + i01 * nb01 + i11 * nb02 + i12 * nb03);

            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(const get_row_context* ctx) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ctx->ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ctx->ne10, std::min<uint16_t>(block_num_y, UINT16_MAX), std::min<uint16_t>(ctx->ne11 * ctx->ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = ctx->nb0 / sizeof(dst_t);
    const size_t s1 = ctx->nb1 / sizeof(dst_t);
    const size_t s2 = ctx->nb2 / sizeof(dst_t);
    const size_t s3 = ctx->nb3 / sizeof(dst_t);

    const size_t s10 = ctx->nb10 / sizeof(int32_t);
    const size_t s11 = ctx->nb11 / sizeof(int32_t);
    const size_t s12 = ctx->nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    k_get_rows_float << <block_nums, block_dims, 0, ctx->stream >> > (
        static_cast<const src0_t*>(ctx->src0_d), ctx->src1_d, static_cast<dst_t*>(ctx->dst_d),
        ctx->ne00, /*ne01, ne02, ne03,*/
        /*ne10, */ ctx->ne11, ctx->ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ ctx->nb01, ctx->nb02, ctx->nb03,
        s10, s11, s12/*, s13*/);
}

template <typename block_type, int qr, typename dst_t>
static __global__ void k_get_rows(
    const void* __restrict__ src0, const int32_t* __restrict__ src1, dst_t* __restrict__ dst,
    const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
    /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
    /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
    /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
    const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    static constexpr size_t qk = block_type::block_size;
    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        for (int64_t i00 = 2 * (blockIdx.y * blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y * blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 = blockIdx.x;
            const int i11 = z / ne12; // TODO fastdiv
            const int i12 = z % ne12;

            const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

            dst_t* dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;
            const void* src0_row = (const char*)src0 + i01 * nb01 + i11 * nb02 + i12 * nb03;

            const int ib = i00 / qk;      // block index
            const int iqs = (i00 % qk) / qr;  // quant index
            const int iybs = i00 - i00 % qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk / 2;

            // dequantize
            float2 v;
            dequantize(static_cast<const block_type*>(src0_row), ib, iqs, v);

            dst_row[iybs + iqs + 0] = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <typename block_type, int qr, typename dst_t>
static void get_rows_cuda(const get_row_context* ctx) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ctx->ne00 + 2 * CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2 * CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ctx->ne10, std::min<uint16_t>(block_num_y, UINT16_MAX), std::min<uint16_t>(ctx->ne11 * ctx->ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = ctx->nb1 / sizeof(dst_t);
    const size_t s2 = ctx->nb2 / sizeof(dst_t);
    const size_t s3 = ctx->nb3 / sizeof(dst_t);

    const size_t s10 = ctx->nb10 / sizeof(int32_t);
    const size_t s11 = ctx->nb11 / sizeof(int32_t);
    const size_t s12 = ctx->nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ctx->ne00 % 2 == 0);

    k_get_rows<block_type, qr, dst_t> << <block_nums, block_dims, 0, ctx->stream >> > (
        ctx->src0_d, ctx->src1_d, static_cast<dst_t*>(ctx->dst_d),
        ctx->ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ctx->ne11, ctx->ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ ctx->nb01, ctx->nb02, ctx->nb03,
        s10, s11, s12/*, s13*/);
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
    const grad_t* __restrict__ grad, const int32_t* __restrict__ rows, dst_t* __restrict__ dst, const int64_t ncols, const int64_t nrows_grad) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int dst_row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += grad[i * ncols + col];
    }

    dst[dst_row * ncols + col] = sum;
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(const get_row_context* ctx)
{
    switch (ctx->src0_type) {
    case GGML_TYPE_F16:
        get_rows_cuda_float<const half, dst_t>(ctx);
        break;
    case GGML_TYPE_F32:
        get_rows_cuda_float<const float, dst_t>(ctx);
        break;
    case GGML_TYPE_I32:
        get_rows_cuda_float<const int32_t, dst_t>(ctx);
        break;
    case GGML_TYPE_BF16:
        get_rows_cuda_float<const nv_bfloat16, dst_t>(ctx);
        break;
    case GGML_TYPE_Q4_0:
        get_rows_cuda<block_q4_0, QR4_0, dst_t>(ctx);
        break;
    case GGML_TYPE_Q4_1:
        get_rows_cuda<block_q4_1, QR4_1, dst_t>(ctx);
        break;
    case GGML_TYPE_Q5_0:
        get_rows_cuda<block_q5_0, QR5_0, dst_t>(ctx);
        break;
    case GGML_TYPE_Q5_1:
        get_rows_cuda<block_q5_1, QR5_1, dst_t>(ctx);
        break;
    case GGML_TYPE_Q8_0:
        get_rows_cuda<block_q8_0, QR8_0, dst_t>(ctx);
        break;
    default:
        // TODO: k-quants
        GGML_ABORT("%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
        break;
    }
}

void get_rows_cuda(const get_row_context* ctx)
{
    switch (ctx->dst_type) {
    case GGML_TYPE_F32:
        ggml_cuda_get_rows_switch_src0_type<float>(ctx);
        break;
    case GGML_TYPE_I32:
        ggml_cuda_get_rows_switch_src0_type<int32_t>(ctx);
        break;
    case GGML_TYPE_F16:
        ggml_cuda_get_rows_switch_src0_type<half>(ctx);
        break;
    case GGML_TYPE_BF16:
        ggml_cuda_get_rows_switch_src0_type<nv_bfloat16>(ctx);
        break;
    default:
        GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
        break;
    }
}

void get_rows_back_cuda(const get_row_back_context* ctx, cudaStream_t stream)
{
    static constexpr size_t CUDA_GET_ROWS_BACK_BLOCK_SIZE = 256;
    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ctx->ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ctx->ne1, 1);

    k_get_rows_back_float << <block_nums, block_dims, 0, stream >> > (ctx->src0_d, ctx->src1_d, ctx->dst_d, ctx->ne00, ctx->ne10);
}