#include <float.h>
#include <bit>
#include "cuda_func.h"
#include "common.cuh"
#include "table.h"
#include "dequantize.cuh"
#include "cpy-utils.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

#define GGML_ASSERT(...)

template <typename T>
concept block_v =
    std::is_same_v<T, block_q8_0> ||
    std::is_same_v<T, block_q5_1> ||
    std::is_same_v<T, block_q4_0> ||
    std::is_same_v<T, block_q4_1> ||
    std::is_same_v<T, block_iq4_nl> ||
    std::is_same_v<T, block_q5_0>;

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
            dequantize(xi, j, dq);
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
requires (block_v<src_t> || block_v<dst_t>)
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
    const int64_t i03 = i / (ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2]);
    const int64_t i02 = (i - i03 * ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2]) / (ctx.src_ne[0] * ctx.src_ne[1]);
    const int64_t i01 = (i - i03 * ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2] - i02 * ctx.src_ne[1] * ctx.src_ne[0]) / ctx.src_ne[0];
    const int64_t i00 = i - i03 * ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2] - i02 * ctx.src_ne[1] * ctx.src_ne[0] - i01 * ctx.src_ne[0];
    const int64_t i13 = i / (ctx.dst_ne[0] * ctx.dst_ne[1] * ctx.dst_ne[2]);
    const int64_t i12 = (i - i13 * ctx.dst_ne[0] * ctx.dst_ne[1] * ctx.dst_ne[2]) / (ctx.dst_ne[0] * ctx.dst_ne[1]);
    const int64_t i11 = (i - i13 * ctx.dst_ne[0] * ctx.dst_ne[1] * ctx.dst_ne[2] - i12 * ctx.dst_ne[0] * ctx.dst_ne[1]) / ctx.dst_ne[0];
    const int64_t i10 = i - i13 * ctx.dst_ne[0] * ctx.dst_ne[1] * ctx.dst_ne[2] - i12 * ctx.dst_ne[0] * ctx.dst_ne[1] - i11 * ctx.dst_ne[0];;

    const int64_t x_offset = [=] {
        if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
            return (i00 / src_t::block_size) * ctx.src_nb[0] + i01 * ctx.src_nb[1] + i02 * ctx.src_nb[2] + i03 * ctx.src_nb[3];
        }
        else {
            return i00 * ctx.src_nb[0] + i01 * ctx.src_nb[1] + i02 * ctx.src_nb[2] + i03 * ctx.src_nb[3];
        }
    }();
    const int64_t dst_offset = [=] {
        if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
            return (i10 / dst_t::block_size) * ctx.dst_nb[0] + i11 * ctx.dst_nb[1] + i12 * ctx.dst_nb[2] + i13 * ctx.dst_nb[3];
        }
        else {
            return i10 * ctx.dst_nb[0] + i11 * ctx.dst_nb[1] + i12 * ctx.dst_nb[2] + i13 * ctx.dst_nb[3];
        }
    }();
    copy_block(
        (const src_t*)((const char*)ctx.src_d + x_offset),
        (dst_t*)((char*)ctx.dst_d + dst_offset));
}

template <typename src_t, typename dst_t>
requires (block_v<src_t> || block_v<dst_t>)
static void ggml_cpy_cuda(const dup_context& ctx, cudaStream_t stream) {
    // copy context by value
    if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
        GGML_ASSERT(ctx.ne % dst_t::block_size == 0);
        const int num_blocks = ctx.ne / dst_t::block_size;
        cpy<src_t, dst_t> << <num_blocks, 1, 0, stream >> > (ctx);
    }
    else if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
        const int num_blocks = ctx.ne;
        cpy<src_t, dst_t> << <num_blocks, 1, 0, stream >> > (ctx);
    }
}

void ggml_cpy_type_erasure_cuda(const dup_context& ctx, cudaStream_t stream)
{
    std::array<int64_t, 4> src_ne = { ctx.src_ne[0] / (int64_t)ctx.src_block_size, ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::array<size_t, 4> src_nb = { ctx.src_nb[0], ctx.src_nb[1], ctx.src_nb[2], ctx.src_nb[3] };
    std::array<int64_t, 4> dst_ne = { ctx.dst_ne[0] / (int64_t)ctx.dst_block_size, ctx.dst_ne[1], ctx.dst_ne[2], ctx.dst_ne[3] };
    std::array<size_t, 4> dst_nb = { ctx.dst_nb[0], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3] };
    launch_functor(stream, std::make_tuple(dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
            // then combine those indices with the corresponding byte offsets to get the total offsets
            const int64_t i = i3 * dst_ne[2] * dst_ne[1] * dst_ne[0] + i2 * dst_ne[1] * dst_ne[0] + i1 * dst_ne[0] + i0;
            const int64_t i13 = i / (src_ne[0] * src_ne[1] * src_ne[2]);
            const int64_t i12 = (i - i13 * src_ne[0] * src_ne[1] * src_ne[2]) / (src_ne[0] * src_ne[1]);
            const int64_t i11 = (i - i13 * src_ne[0] * src_ne[1] * src_ne[2] - i12 * src_ne[0] * src_ne[1]) / src_ne[0];
            const int64_t i10 = i - i13 * src_ne[0] * src_ne[1] * src_ne[2] - i12 * src_ne[0] * src_ne[1] - i11 * src_ne[0];;

            const char* src_ptr = ((char*)ctx.src_d + i10 * src_nb[0] + i11 * src_nb[1] + i12 * src_nb[2] + i13 * src_nb[3]);
            char* dst_ptr = ((char*)ctx.dst_d + i0 * dst_nb[0] + i1 * dst_nb[1] + i2 * dst_nb[2] + i3 * dst_nb[3]);
            memcpy(dst_ptr, src_ptr, ctx.src_type_size);
        }
    );
}

template <typename src_t, typename dst_t>
requires (!block_v<src_t> && !block_v<dst_t> && !std::is_same_v<src_t, dst_t>)
void ggml_cpy_cuda(const dup_context& ctx, cudaStream_t stream)
{
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(ctx.src_d), ctx.src_ne, ctx.src_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
            // then combine those indices with the corresponding byte offsets to get the total offsets
            int64_t i = i3 * ctx.dst_ne[2] * ctx.dst_ne[1] * ctx.dst_ne[0];
            i += i2 * ctx.dst_ne[1] * ctx.dst_ne[0];
            i += i1 * ctx.dst_ne[0];
            i += i0;;

            const int64_t i13 = i / (ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2]);
            const int64_t i12 = (i - i13 * ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2]) / (ctx.src_ne[0] * ctx.src_ne[1]);
            const int64_t i11 = (i - i13 * ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2] - i12 * ctx.src_ne[0] * ctx.src_ne[1]) / ctx.src_ne[0];
            const int64_t i10 = i - i13 * ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2] - i12 * ctx.src_ne[0] * ctx.src_ne[1] - i11 * ctx.src_ne[0];;
            dst_data(i3, i2, i1, i0) = ggml_cuda_cast<dst_t>(src_data(i13, i12, i11, i10));
        }
    );
}

void dup_cuda(const dup_context& ctx, cudaStream_t stream)
{
    if (ctx.src_type == ctx.dst_type) {
        if (ctx.contiguous) {
            GGML_ASSERT(ctx.src_length == ctx.dst_length);
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
            if (src0->type == internal::GGML_TYPE_F32 || src0->type == internal::GGML_TYPE_F16) {
                CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
            }
            else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
            {
                CUDA_CHECK(cudaMemcpyAsync(ctx.dst_d, ctx.src_d, ctx.src_length, cudaMemcpyDeviceToDevice, stream));
            }
        }
        else {
            ggml_cpy_type_erasure_cuda(ctx, stream);
        }
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_BF16) {
        ggml_cpy_cuda<float, nv_bfloat16>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F16) {
        ggml_cpy_cuda<float, half>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_Q8_0) {
        ggml_cpy_cuda<float, block_q8_0>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_Q8_0 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q8_0, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_Q4_0) {
        ggml_cpy_cuda<float, block_q4_0>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_Q4_0 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q4_0, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_Q4_1) {
        ggml_cpy_cuda<float, block_q4_1>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_Q4_1 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q4_1, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_Q5_0) {
        ggml_cpy_cuda<float, block_q5_0>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_Q5_0 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q5_0, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_IQ4_NL) {
        ggml_cpy_cuda<float, block_iq4_nl>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_Q5_1) {
        ggml_cpy_cuda<float, block_q5_1>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_Q5_1 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q5_1, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_BF16) {
        ggml_cpy_cuda<half, nv_bfloat16>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<half, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_BF16 && ctx.dst_type == internal::GGML_TYPE_F16) {
        ggml_cpy_cuda<nv_bfloat16, half>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_BF16 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<nv_bfloat16, float>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_I32) {
        ggml_cpy_cuda<float, int32_t>(ctx, stream);
    }
    else if (ctx.src_type == internal::GGML_TYPE_I32 && ctx.dst_type == internal::GGML_TYPE_F32) {
        ggml_cpy_cuda<int32_t, float>(ctx, stream);
    }
    else {
        assert(false);
        //GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
          //  internal::GGML_TYPE_name(src0->type), internal::GGML_TYPE_name(src1->type));
    }
}

