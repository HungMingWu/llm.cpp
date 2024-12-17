#include <float.h>
#include <bit>
#include "cuda_func.h"
#include "common.cuh"
#include "table.h"
#include "dequantize.cuh"

#define GGML_ASSERT(...)

template <typename src_t, typename dst_t>
concept simplest_case_v =
    (std::is_same_v<src_t, float> && (
        std::is_same_v<dst_t, float> || std::is_same_v<dst_t, half> || std::is_same_v<dst_t, nv_bfloat16>)) ||
    (std::is_same_v<src_t, half> && (std::is_same_v<dst_t, float> || std::is_same_v<dst_t, half>));

template <typename T>
concept block_v =
    std::is_same_v<T, block_q8_0> ||
    std::is_same_v<T, block_q5_1> ||
    std::is_same_v<T, block_q4_0> ||
    std::is_same_v<T, block_q4_1> ||
    std::is_same_v<T, block_iq4_nl> ||
    std::is_same_v<T, block_q5_0>;

template <typename src_t, typename dst_t>
static __device__ void copy_block(const src_t* xi, dst_t* dsti) {
    if constexpr (std::is_same_v<src_t, float>) {
        if constexpr (std::is_same_v<dst_t, float>) {
            *dsti = *xi;
        }
        else if constexpr (std::is_same_v<dst_t, half>) {
            *dsti = __float2half(*xi);
        }
        else if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            *dsti = *xi;
        }
    }
    else if constexpr (std::is_same_v<src_t, half>) {
        if constexpr (std::is_same_v<dst_t, half>) {
            *dsti = *xi;
        }
        else if constexpr (std::is_same_v<dst_t, float>) {
            *dsti = __half2float(*xi);
        }
    }
}

template <block_v src_t>
static __device__ void copy_block(const src_t* xi, float* dsti) {
    if constexpr (std::is_same_v<src_t, block_q8_0>) {
        const float d = __half2float(std::bit_cast<half>(xi->d));
#pragma unroll
        for (int j = 0; j < block_q8_0::block_size; j++) {
            dsti[j] = xi->qs[j] * d;
        }
    }
    else {
#pragma unroll
        for (int j = 0; j < src_t::block_size / 2; j++) {
            dfloat2 dq;
            dequantize(xi, 0, j, dq);
            *(dsti + j) = dq.x;
            *(dsti + j + src_t::block_size / 2) = dq.y;
        }
    }
};

static __device__ void copy_block(const float* xi, block_q8_0* dsti) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < block_q8_0::block_size; j++) {
        const float v = xi[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = std::bit_cast<uint16_t>(__float2half(d));

    for (int j = 0; j < block_q8_0::block_size; ++j) {
        const float x0 = xi[j] * id;

        dsti->qs[j] = roundf(x0);
    }
}

static __device__ void copy_block(const float* xi, block_q4_0* dsti) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < block_q4_0::block_size; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d = vmax / -8;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = std::bit_cast<uint16_t>(__float2half(d));

    for (int j = 0; j < block_q4_0::block_size / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[block_q4_0::block_size / 2 + j] * id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void copy_block(const float* xi, block_q4_1* dsti) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < block_q4_1::block_size; ++j) {
        const float v = xi[j];

        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    std::array<half, 2> dm{ __float2half(d), __float2half(vmin) };
    dsti->dm = std::bit_cast<uint32_t>(dm);

    for (int j = 0; j < block_q4_1::block_size / 2; ++j) {
        const float x0 = (xi[0 + j] - vmin) * id;
        const float x1 = (xi[block_q4_1::block_size / 2 + j] - vmin) * id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void copy_block(const float* xi, block_q5_0* dsti) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < block_q5_0::block_size; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d = vmax / -16;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = std::bit_cast<uint16_t>(__float2half(d));

    uint32_t qh = 0;
    for (int j = 0; j < block_q5_0::block_size / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[block_q5_0::block_size / 2 + j] * id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + block_q5_0::block_size / 2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

static __device__ __forceinline__ int best_index_int8(int n, const int8_t* val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n - 1]) return n - 1;
    int ml = 0, mu = n - 1;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

static __device__ void copy_block(const float* xi, block_iq4_nl* dsti) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < block_iq4_nl::block_size; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f / d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < block_iq4_nl::block_size / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[block_iq4_nl::block_size / 2 + j] * id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = xi[0 + j] * xi[0 + j];
        const float w1 = xi[block_iq4_nl::block_size / 2 + j] * xi[block_iq4_nl::block_size / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[block_iq4_nl::block_size / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = std::bit_cast<uint16_t>(__float2half(sumq2 > 0 ? sumqx / sumq2 : d));
}

static __device__ void copy_block(const float* xi, block_q5_1* dsti) {
    float min = xi[0];
    float max = xi[0];

    for (int j = 1; j < block_q5_1::block_size; ++j) {
        const float v = xi[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f / d : 0.0f;

    std::array<half, 2> dm{ __float2half(d), __float2half(min) };
    dsti->dm = std::bit_cast<uint32_t>(dm);

    uint32_t qh = 0;
    for (int j = 0; j < block_q5_1::block_size / 2; ++j) {
        const float x0 = (xi[0 + j] - min) * id;
        const float x1 = (xi[block_q5_1::block_size / 2 + j] - min) * id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + block_q5_1::block_size / 2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

template <typename src_t, typename dst_t>
static __global__ void cpy(dup_context ctx) {
    const int64_t i = [=] {
        int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if constexpr (block_v<src_t>) i *= src_t::block_size;
        else if constexpr (block_v<dst_t>) i *= dst_t::block_size;
        return i;
    }();
    if (i >= ctx.ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i / (ctx.ne00 * ctx.ne01 * ctx.ne02);
    const int64_t i02 = (i - i03 * ctx.ne00 * ctx.ne01 * ctx.ne02) / (ctx.ne00 * ctx.ne01);
    const int64_t i01 = (i - i03 * ctx.ne00 * ctx.ne01 * ctx.ne02 - i02 * ctx.ne01 * ctx.ne00) / ctx.ne00;
    const int64_t i00 = i - i03 * ctx.ne00 * ctx.ne01 * ctx.ne02 - i02 * ctx.ne01 * ctx.ne00 - i01 * ctx.ne00;
    const int64_t i13 = i / (ctx.ne10 * ctx.ne11 * ctx.ne12);
    const int64_t i12 = (i - i13  * ctx.ne10 * ctx.ne11 *ctx. ne12) / (ctx.ne10 * ctx.ne11);
    const int64_t i11 = (i - i13 * ctx.ne10 * ctx.ne11 * ctx.ne12 - i12 * ctx.ne10 * ctx.ne11) / ctx.ne10;
    const int64_t i10 = i - i13 * ctx.ne10 * ctx.ne11 * ctx.ne12 - i12 * ctx.ne10 * ctx.ne11 - i11 * ctx.ne10;;
    
    const int64_t x_offset = [=] {
        if constexpr (simplest_case_v<src_t, dst_t>) {
            return i00 * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
        else if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
            return i00 * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
        else if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
            return (i00 / src_t::block_size) * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
        else {
            return 0;
        }
    }();
    const int64_t dst_offset = [=] {
        if constexpr (simplest_case_v<src_t, dst_t>) {
            return i10 * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
        }
        else if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
            return (i10 / dst_t::block_size) * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
        }
        else if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
            return i10 * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
        } else {
            return 0;
        }
    }();
    copy_block(
        (const src_t*)((const char*)ctx.src_d + x_offset), 
        (dst_t*)((char *)ctx.dst_d + dst_offset));
}

template <typename src_t, typename dst_t>
static void ggml_cpy_cuda(const dup_context &ctx, cudaStream_t stream) {
    // copy context by value
    if constexpr (simplest_case_v<src_t, dst_t>) {
        static constexpr size_t CUDA_CPY_BLOCK_SIZE = 64;
        const int num_blocks = (ctx.ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        cpy<src_t, dst_t> << <num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream >> > (ctx);
    }
    else if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
        GGML_ASSERT(ctx.ne % dst_t::block_size == 0);
        const int num_blocks = ctx.ne / dst_t::block_size;
        cpy<src_t, dst_t> << <num_blocks, 1, 0, stream >> > (ctx);
    }
    else if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
        const int num_blocks = ctx.ne;
        cpy<src_t, dst_t> << <num_blocks, 1, 0, stream >> > (ctx);
    }
}

void dup_cuda(const dup_context* ctx, cudaStream_t stream)
{
    if (ctx->src_type == ctx->dst_type && ctx->src_is_contiguous && ctx->dst_is_contiguous) {
        CUDA_CHECK(cudaMemcpyAsync(ctx->dst_d, ctx->src_d, ctx->src_length, cudaMemcpyDeviceToDevice, stream));
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<float, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_BF16) {
        ggml_cpy_cuda<float, nv_bfloat16>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_F16) {
        ggml_cpy_cuda<float, half>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_Q8_0) {
        ggml_cpy_cuda<float, block_q8_0>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_Q8_0 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q8_0, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_Q4_0) {
        ggml_cpy_cuda<float, block_q4_0>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_Q4_0 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q4_0, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_Q4_1) {
        ggml_cpy_cuda<float, block_q4_1>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_Q4_1 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q4_1, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_Q5_0) {
        ggml_cpy_cuda<float, block_q5_0>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_Q5_0 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q5_0, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_cuda<float, block_iq4_nl>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_Q5_1) {
        ggml_cpy_cuda<float, block_q5_1>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_Q5_1 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q5_1, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F16 && ctx->dst_type == GGML_TYPE_F16) {
        ggml_cpy_cuda<half, half>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F16 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<half, float>(*ctx, stream);
    }
    else {
        assert(false);
        //GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
          //  ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}
