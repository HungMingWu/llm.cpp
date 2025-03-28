#define GGML_ASSERT(...)
#include <bit>
#include "internal_ds.h"
#include "cuda_func.h"
#include "block.h"
#include "common.cuh"

static constexpr size_t CUDA_GET_ROWS_BLOCK_SIZE = 256;

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
    const src0_t* __restrict__ src0, const int32_t* __restrict__ src1, dst_t* __restrict__ dst,
    const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
    /*const int64_t ne10, const int64_t ne11,*/ const int64_t ne12, /*const int64_t ne13,*/
    /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
    /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
    const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    const int i00 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i10 = blockDim.y * blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z * blockDim.z + threadIdx.z) / ne12;
    const int i12 = (blockIdx.z * blockDim.z + threadIdx.z) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

    dst_t* dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;
    const src0_t* src0_row = (const src0_t*)((const char*)src0 + i01 * nb01 + i11 * nb02 + i12 * nb03);

    dst_row[i00] = src0_row[i00];
}

template<typename src0_t>
static void get_rows_cuda_float(const get_row_context* ctx) {
    GGML_ASSERT(ctx->ne13 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ctx->ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ctx->ne10, ctx->ne11 * ctx->ne12);

    k_get_rows_float<src0_t> << <block_nums, block_dims, 0, ctx->stream >> > (
        (const src0_t*)ctx->src0_d, ctx->src1_d, ctx->dst_d,
        ctx->ne00, /*ne01, ne02, ne03,*/
        /*ne10, ne11,*/ ctx->ne12, /*ne13,*/
        /* s0,*/ ctx->s1, ctx->s2, ctx->s3,
        /* nb00,*/ ctx->nb01, ctx->nb02, ctx->nb03,
        ctx->s10, ctx->s11, ctx->s12/*, s13*/);
}

template <typename block_type, int qr>
static __global__ void k_get_rows(get_row_context ctx) {

    const int i00 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int i10 = blockDim.y * blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z * blockDim.z + threadIdx.z) / ctx.ne12;
    const int i12 = (blockIdx.z * blockDim.z + threadIdx.z) % ctx.ne12;

    if (i00 >= ctx.ne00) {
        return;
    }

    const int i01 = ctx.src1_d[i10 * ctx.s10 + i11 * ctx.s11 + i12 * ctx.s12];

    float* dst_row = ctx.dst_d + i10 * ctx.s1 + i11 * ctx.s2 + i12 * ctx.s3;
    const void* src0_row = (const char*)ctx.src0_d + i01 * ctx.nb01 + i11 * ctx.nb02 + i12 * ctx.nb03;

    static constexpr size_t qk = block_type::block_size;
    const int ib = i00 / qk;      // block index
    const int iqs = (i00 % qk) / qr;  // quant index
    const int iybs = i00 - i00 % qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    dequantize(static_cast<const block_type*>(src0_row), ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x;
    dst_row[iybs + iqs + y_offset] = v.y;
}

template <typename block_type, int qr>
static void get_rows_cuda(const get_row_context* ctx) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ctx->ne00 + 2 * CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2 * CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(block_num_x, ctx->ne10, ctx->ne11 * ctx->ne12);

    GGML_ASSERT(ctx->ne00 % 2 == 0);

    k_get_rows<block_type, qr> << <block_nums, block_dims, 0, ctx->stream >> > (*ctx);
}

static __device__ __forceinline__ void dequantize(const block_q4_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = std::bit_cast<half>(x[ib].d);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, { 8.0f, 8.0f });
    v = __hmul2(v, { d, d });
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q4_1* x, const int64_t ib, const int iqs, dfloat2& v) {
    const auto dm = std::bit_cast<std::array<half, 2>>(x[ib].dm);
    const dfloat d = dm[0];
    const dfloat m = dm[1];

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
    v = __hadd2(v, { m, m });
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q5_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = std::bit_cast<half>(x[ib].d);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, { 16.0f, 16.0f });
    v = __hmul2(v, { d, d });
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q5_1* x, const int64_t ib, const int iqs, dfloat2& v) {
    const auto dm = std::bit_cast<std::array<half, 2>>(x[ib].dm);
    const dfloat d = dm[0];
    const dfloat m = dm[1];

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
    v = __hadd2(v, { m, m });
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q8_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = std::bit_cast<half>(x[ib].d);

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
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

void get_rows_cuda(const get_row_context* ctx)
{
    switch (ctx->type) {
    case GGML_TYPE_F16:
        get_rows_cuda_float<const half>(ctx);
        break;
    case GGML_TYPE_F32:
        get_rows_cuda_float<const float>(ctx);
        break;
    case GGML_TYPE_Q4_0:
        get_rows_cuda<block_q4_0, QR4_0>(ctx);
        break;
    case GGML_TYPE_Q4_1:
        get_rows_cuda<block_q4_1, QR4_1>(ctx);
        break;
    case GGML_TYPE_Q5_0:
        get_rows_cuda<block_q5_0, QR5_0>(ctx);
        break;
    case GGML_TYPE_Q5_1:
        get_rows_cuda<block_q5_1, QR5_1>(ctx);
        break;
    case GGML_TYPE_Q8_0:
        get_rows_cuda<block_q8_0, QR8_0>(ctx);
        break;
    default:
        // TODO: k-quants
        GGML_ABORT("%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
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