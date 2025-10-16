#include <float.h>
#include <bit>
#include "cuda_func.h"
#include "common.cuh"
#include "table.h"
#include "dequantize.cuh"
#include "cpy-utils.cuh"

#define GGML_ASSERT(...)

template <typename T>
concept block_v =
    std::is_same_v<T, block_q8_0> ||
    std::is_same_v<T, block_q5_1> ||
    std::is_same_v<T, block_q4_0> ||
    std::is_same_v<T, block_q4_1> ||
    std::is_same_v<T, block_iq4_nl> ||
    std::is_same_v<T, block_q5_0>;

template <typename src_t, typename dst_t>
concept simplest_case_v = !block_v<src_t> && !block_v<dst_t>;

template <typename src_t, typename dst_t>
static __device__ void copy_block(const src_t* xi, dst_t* dsti) {
    *dsti = ggml_cuda_cast<dst_t>(*xi);
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
            float2 dq;
            dequantize(xi, 0, j, dq);
            *(dsti + j) = dq.x;
            *(dsti + j + src_t::block_size / 2) = dq.y;
        }
    }
};

static __device__ void copy_block(const float* xi, block_q4_0* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q4_1* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q5_0* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q5_1* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q8_0* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_iq4_nl* dsti) {
    quantize_block(xi, dsti);
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
        if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
            return (i00 / src_t::block_size) * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
        else {
            return i00 * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
    }();
    const int64_t dst_offset = [=] {
        if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
            return (i10 / dst_t::block_size) * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
        } else {
            return i10 * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
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
        GGML_ASSERT(ctx->src_length == ctx->dst_length);
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        }
        else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            CUDA_CHECK(cudaMemcpyAsync(ctx->dst_d, ctx->src_d, ctx->src_length, cudaMemcpyDeviceToDevice, stream));
        }
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
    else if (ctx->src_type == GGML_TYPE_F16 && ctx->dst_type == GGML_TYPE_BF16) {
        ggml_cpy_cuda<half, nv_bfloat16>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F16 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<half, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_BF16 && ctx->dst_type == GGML_TYPE_BF16) {
        ggml_cpy_cuda<nv_bfloat16, nv_bfloat16>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_BF16 && ctx->dst_type == GGML_TYPE_F16) {
        ggml_cpy_cuda<nv_bfloat16, half>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_BF16 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<nv_bfloat16, float>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_I32) {
        ggml_cpy_cuda<float, int32_t>(*ctx, stream);
    }
    else if (ctx->src_type == GGML_TYPE_I32 && ctx->dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<int32_t, float>(*ctx, stream);
    }
    else {
        assert(false);
        //GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
          //  ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}
