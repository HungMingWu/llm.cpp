#define GGML_ASSERT(...)
#include <bit>
#include "cuda_func.h"
#include "block.h"
#include "common.cuh"
#include "dequantize.cuh"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

template <typename src_t>
concept is_non_block_v =
    std::is_same_v<src_t, half> ||
    std::is_same_v<src_t, float> ||
    std::is_same_v<src_t, int32_t> ||
    std::is_same_v<src_t, nv_bfloat16>;

template <typename src_t>
concept is_block_kq_v =
    std::is_same_v<src_t, block_q2_K> ||
    std::is_same_v<src_t, block_q3_K> ||
    std::is_same_v<src_t, block_q4_K> ||
    std::is_same_v<src_t, block_q5_K> ||
    std::is_same_v<src_t, block_q6_K> ||
    std::is_same_v<src_t, block_iq2_xxs> ||
    std::is_same_v<src_t, block_iq2_xs> ||
    std::is_same_v<src_t, block_iq2_s> ||
    std::is_same_v<src_t, block_iq3_xxs> ||
    std::is_same_v<src_t, block_iq3_s> ||
    std::is_same_v<src_t, block_iq1_s> ||
    std::is_same_v<src_t, block_iq1_m> ||
    std::is_same_v<src_t, block_iq4_nl> ||
    std::is_same_v<src_t, block_iq4_xs> ||
    std::is_same_v<src_t, block_mxfp4>;

template <typename src0_t, typename dst_t>
requires (is_non_block_v<src0_t>)
void get_rows_cuda(const get_row_context &ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    if constexpr (std::is_same_v<src0_t, dst_t>) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const bool can_vec = VEC > 1 &&
            (ctx.src0_ne[0] % VEC == 0) &&
            (ctx.src0_nb[1] % 16 == 0) && (ctx.src0_nb[2] % 16 == 0) && (ctx.src0_nb[3] % 16 == 0) &&
            (ctx.dst_nb[1]  % 16 == 0) && (ctx.dst_nb[2]  % 16 == 0) && (ctx.dst_nb[3]  % 16 == 0) &&
            (((uintptr_t) ctx.src0_d) % 16 == 0) && (((uintptr_t) ctx.dst_d) % 16 == 0);

        if (can_vec) {
            launch_functor(stream, std::make_tuple(ctx.src0_ne[0] / VEC, ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2]),
                [=] __device__(int64_t i00v, int64_t i10, int64_t i11, int64_t i12) {
                    const int i01 = src1_data(i12, i11, i10);
                    const int64_t i00 = i00v * VEC;
                    auto expand = [&]<size_t ...Is>(std::index_sequence<Is...>)
                    {
                        ((dst_data(i12, i11, i10, i00 + Is) = src0_data(i12, i11, i01, i00 + Is)), ...);
                    };

                    expand(std::make_index_sequence<16 / sizeof(dst_t)>{});
                }
            );
            return;
        }
    }
    launch_functor(stream, std::make_tuple(ctx.src1_ne[2], ctx.src1_ne[1], ctx.src1_ne[0], ctx.src0_ne[0]),
        [=] __device__(int64_t i12, int64_t i11, int64_t i10, int64_t i00) {
            const int i01 = src1_data(i12, i11, i10);

            dst_data(i12, i11, i10, i00) = ggml_cuda_cast<dst_t>(src0_data(i12, i11, i01, i00));
        }
    );
}

template <typename src_t>
constexpr int get_block_dim() {
    if constexpr (
        std::is_same_v<src_t, block_q2_K> ||
        std::is_same_v<src_t, block_q3_K> ||
        std::is_same_v<src_t, block_q5_K> ||
        std::is_same_v<src_t, block_q6_K>
    ) {
        return 64;
    }
    return 32;
}

template <typename src_t, typename dst_t>
requires (is_block_kq_v<src_t>)
void get_rows_cuda(const get_row_context &ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const src_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    GGML_ASSERT(ctx.src0_ne[0] % QK_K == 0);

    const int64_t nsb = ctx.src0_ne[0] / QK_K;
    static constexpr int block_dim = get_block_dim<src_t>();

    launch_functor(stream, std::make_tuple(block_dim, ctx.src1_ne[2], ctx.src1_ne[1], ctx.src1_ne[0], nsb),
        [=] __device__(int64_t tid, int64_t i12, int64_t i11, int64_t i10, int64_t ib) {
            const int i01 = src1_data(i12, i11, i10);

            dequantize(&src0_data(i12, i11, i01, 0), ib,
            &dst_data(i12, i11, i10, ib * QK_K), tid);
        }
    );
}

template <typename src_t, typename dst_t>
requires (!is_block_kq_v<src_t> && !is_non_block_v<src_t>)
void get_rows_cuda(const get_row_context &ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const src_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    GGML_ASSERT(ctx.src0_ne[0] % 2 == 0);

    static constexpr int qr = ggml_cuda_type_traits<src_t>::qr;
    static constexpr size_t qk = src_t::block_size;
    launch_functor(stream, std::make_tuple(ctx.src1_ne[2], ctx.src1_ne[1], ctx.src1_ne[0], ctx.src0_ne[0] / 2),
        [=] __device__(int64_t i12, int64_t i11, int64_t i10, int64_t i00) {
            i00 *= 2;
            const int i01 = src1_data(i12, i11, i10);

            const int ib = i00 / qk;      // block index
            const int iqs = (i00 % qk) / qr;  // quant index
            const int iybs = i00 - i00 % qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk / 2;

            // dequantize
            float2 v;
            dequantize(&src0_data(i12, i11, i01, ib), iqs, v);

            dst_data(i12, i11, i10, iybs + iqs) = ggml_cuda_cast<dst_t>(v.x);
            dst_data(i12, i11, i10, iybs + iqs + y_offset) = ggml_cuda_cast<dst_t>(v.y);
        }
    );
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(const get_row_context &ctx, cudaStream_t stream)
{
    switch (ctx.src0_type) {
    case internal::GGML_TYPE_F16:
        get_rows_cuda<half, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_F32:
        get_rows_cuda<float, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_I32:
        get_rows_cuda<int32_t, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_BF16:
        get_rows_cuda<nv_bfloat16, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q1_0:
        get_rows_cuda<block_q1_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q2_0:
        get_rows_cuda<block_q2_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q4_0:
        get_rows_cuda<block_q4_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q4_1:
        get_rows_cuda<block_q4_1, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q5_0:
        get_rows_cuda<block_q5_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q5_1:
        get_rows_cuda<block_q5_1, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q8_0:
        get_rows_cuda<block_q8_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q2_K:
        get_rows_cuda<block_q2_K, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q3_K:
        get_rows_cuda<block_q3_K, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q4_K:
        get_rows_cuda<block_q4_K, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q5_K:
        get_rows_cuda<block_q5_K, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q6_K:
        get_rows_cuda<block_q6_K, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ2_XXS:
        get_rows_cuda<block_iq2_xxs, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ2_XS:
        get_rows_cuda<block_iq2_xs, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ2_S:
        get_rows_cuda<block_iq2_s, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ3_XXS:
        get_rows_cuda<block_iq3_xxs, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ3_S:
        get_rows_cuda<block_iq3_s, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ1_S:
        get_rows_cuda<block_iq1_s, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ1_M:
        get_rows_cuda<block_iq1_m, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ4_NL:
        get_rows_cuda<block_iq4_nl, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_IQ4_XS:
        get_rows_cuda<block_iq4_xs, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_MXFP4:
        get_rows_cuda<block_mxfp4, dst_t>(ctx, stream);
        break;
    default:
        GGML_ABORT("%s: unsupported type: %s\n", __func__, internal::GGML_TYPE_name(src0->type));
        break;
    }
}

void get_rows_cuda(const get_row_context &ctx, cudaStream_t stream)
{
    switch (ctx.dst_type) {
    case internal::GGML_TYPE_F32:
        ggml_cuda_get_rows_switch_src0_type<float>(ctx, stream);
        break;
    case internal::GGML_TYPE_I32:
        ggml_cuda_get_rows_switch_src0_type<int32_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_F16:
        ggml_cuda_get_rows_switch_src0_type<half>(ctx, stream);
        break;
    case internal::GGML_TYPE_BF16:
        ggml_cuda_get_rows_switch_src0_type<nv_bfloat16>(ctx, stream);
        break;
    default:
        GGML_ABORT("%s: unsupported dst type: %s\n", __func__, internal::GGML_TYPE_name(dst_type));
        break;
    }
}

void get_rows_back_cuda(const get_row_back_context &ctx, cudaStream_t stream)
{
    const int64_t nrows_grad = ctx.src1_ne[0];
    std::span rows_data(ctx.src1_d, ctx.src1_ne[0]);
    auto grad_data = make_strided_mdspan<2>(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto dst_data = make_strided_mdspan<2>(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src0_ne[0], ctx.dst_ne[1]),
        [=] __device__(int64_t col, int64_t dst_row) {
            ggml_cuda_pdl_sync();

            float sum = 0.0f;
            for (int64_t i = 0; i < nrows_grad; ++i) {
                if (rows_data[i] == dst_row) {
                    sum += grad_data(i, col);
                }
            }

            dst_data(dst_row, col) = sum;
        }
    );
}
