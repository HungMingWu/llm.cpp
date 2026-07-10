#include "cuda_func.h"
#include "table.h"
#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "block.h"
#include <bit>
#include <assert.h>
#include <type_traits>
#include "launch.cuh"
#include "mdspan_helper.h"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template <typename src_t, typename dst_t>
void convert_unary_cuda(const convert_context& ctx, const void* vx, dst_t* y, cudaStream_t stream) {
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(vx), ctx.src_ne, ctx.src_nb);
    int64_t dst_ne[4] = { ctx.src_ne[0], ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::mdspan dst_data(y, dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]);
    launch_functor(stream, std::make_tuple(dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            dst_data(i3, i2, i1, i0) = ggml_cuda_cast<dst_t>(src_data(i3, i2, i1, i0));
        }
    );
}

template <typename src_t, typename dst_t>
static __device__ void dequantize_block(const int tid, const src_t* x, dst_t* y) {
    dequantize(x, 0, y, tid);
}

template <typename src_t, int qr, typename dst_t>
void dequantize_block_cuda(const convert_context &ctx, const void* x, dst_t* y, cudaStream_t stream) {
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(x), ctx.src_ne, ctx.src_nb);
    int64_t dst_ne[4] = { ctx.src_ne[0], ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::mdspan dst_data(y, dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]);
    const int qk = src_t::block_size;
    launch_functor(stream, std::make_tuple(ctx.src_ne[3], ctx.src_ne[2], ctx.src_ne[1], ctx.src_ne[0] / 2),
        [=] __device__(int64_t i03, int64_t i02, int64_t i01, int64_t i00) {
            i00 *= 2;

            const int64_t ib = i00 / qk; // block index
            const int64_t iqs = (i00 % qk) / qr; // quant index
            const int64_t iybs = i00 - i00 % qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk / 2;

            // dequantize
            float2 v;
            dequantize(&src_data(i03, i02, i01, ib), iqs, v);

            dst_data(i03, i02, i01, iybs + iqs) = ggml_cuda_cast<dst_t>(v.x);
            dst_data(i03, i02, i01, iybs + iqs + y_offset) = ggml_cuda_cast<dst_t>(v.y);
        }
    );
}

template <typename src_t, typename dst_t>
void dequantize_block_cuda(const convert_context& ctx, const void* x, dst_t* y, cudaStream_t stream) {
    assert(ctx.src_ne[0] % QK_K == 0);
    int64_t src_ne[4] = { ctx.src_ne[0] / QK_K, ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    if constexpr (std::is_same_v<src_t, block_iq4_nl>) {
        src_ne[0] /= block_iq4_nl::block_size;
    }
    else if constexpr (std::is_same_v<src_t, block_mxfp4>) {
        src_ne[0] /= block_mxfp4::block_size;
    }
    else {
        src_ne[0] /= QK_K;
    }
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(x), src_ne, ctx.src_nb);
    int64_t dst_ne[4] = { ctx.src_ne[0], ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::mdspan dst_data(y, dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]);
    constexpr auto threads = [] {
        if constexpr (std::is_same_v<src_t, block_q2_K>) return 64;
        if constexpr (std::is_same_v<src_t, block_q3_K>) return 64;
        if constexpr (std::is_same_v<src_t, block_q5_K>) return 64;
        if constexpr (std::is_same_v<src_t, block_q6_K>) return 64;
        return 32;
    }();

    launch_functor_with_threads(stream, std::make_tuple(src_ne[3], src_ne[2], src_ne[1], ctx.src_ne[0] / QK_K), threads, 0,
        [=] __device__(int64_t i03, int64_t i02, int64_t i01, int64_t i00, int64_t tid) {
            const int64_t real_i00 = [=]() {
                if constexpr (std::is_same_v<src_t, block_iq4_nl>) {
                    return i00 * (QK_K / block_iq4_nl::block_size);
                }
                else if constexpr (std::is_same_v<src_t, block_mxfp4>) {
                    return i00 * (QK_K / block_mxfp4::block_size);
                }
                else {
                    return i00;
                }
            }();
            dequantize_block(tid, &src_data(i03, i02, i01, real_i00),
                &dst_data(i03, i02, i01, i00 * QK_K));
        }
    );
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * block_nvfp4::block_size;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    assert(k % block_nvfp4::block_size == 0);
    const int nb = k / block_nvfp4::block_size;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}

template <typename dst_t>
static void convert_to(const convert_context& ctx, const void* x, dst_t* y, cudaStream_t stream)
{
    switch (ctx.src_type) {
    case internal::GGML_TYPE_F16:
        return convert_unary_cuda<half>(ctx, x, y, stream);
    case internal::GGML_TYPE_BF16:
        return convert_unary_cuda<nv_bfloat16>(ctx, x, y, stream);
    case internal::GGML_TYPE_F32:
        return convert_unary_cuda<float>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q1_0:
        return dequantize_block_cuda<block_q1_0, QR1_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q2_0:
        return dequantize_block_cuda<block_q2_0, QR2_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q4_0:
        return dequantize_block_cuda<block_q4_0, QR4_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q4_1:
        return dequantize_block_cuda<block_q4_1, QR4_1>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q5_0:
        return dequantize_block_cuda<block_q5_0, QR5_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q5_1:
        return dequantize_block_cuda<block_q5_1, QR5_1>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q8_0:
        return dequantize_block_cuda<block_q8_0, QR8_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q2_K:
        return dequantize_block_cuda<block_q2_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q3_K:
        return dequantize_block_cuda<block_q3_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q4_K:
        return dequantize_block_cuda<block_q4_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q5_K:
        return dequantize_block_cuda<block_q5_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q6_K:
        return dequantize_block_cuda<block_q6_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ1_S:
        return dequantize_block_cuda<block_iq1_s>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ1_M:
        return dequantize_block_cuda<block_iq1_m>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ2_S:
        return dequantize_block_cuda<block_iq2_s>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ2_XS:
        return dequantize_block_cuda<block_iq2_xs>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ2_XXS:
        return dequantize_block_cuda<block_iq2_xxs>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ3_S:
        return dequantize_block_cuda<block_iq3_s>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ3_XXS:
        return dequantize_block_cuda<block_iq3_xxs>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ4_NL:
        return dequantize_block_cuda<block_iq4_nl>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ4_XS:
        return dequantize_block_cuda<block_iq4_xs>(ctx, x, y, stream);
    case internal::GGML_TYPE_MXFP4:
        return dequantize_block_cuda<block_mxfp4>(ctx, x, y, stream);
    case internal::GGML_TYPE_NVFP4:
        return dequantize_row_nvfp4_cuda(x, y, ctx.src_ne[0] * ctx.src_ne[1] * ctx.src_ne[2] * ctx.src_ne[3], stream);
    default:
        assert(false);
        return GGML_ABORT("Fatal error");
    }
}

void convert_to_cuda(const convert_context& ctx, const void* x, half* y, cudaStream_t stream) {
    return convert_to(ctx, x, y, stream);
}

void convert_to_cuda(const convert_context& ctx, const void* x, nv_bfloat16* y, cudaStream_t stream) {
    return convert_to(ctx, x, y, stream);
}

void convert_to_cuda(const convert_context& ctx, const void* x, float* y, cudaStream_t stream) {
    return convert_to(ctx, x, y, stream);
}
