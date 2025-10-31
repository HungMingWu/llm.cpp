#include <assert.h>
#include "cuda_func.h"
#include "internal_ds.h"
#include "cpy-utils.cuh"
#include "convert.cuh"
#include "helper.h"
#include "launch.cuh"

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

// Template dispatch function for quantized set_rows
template <typename src_t, typename idx_t, typename block_type>
void set_rows_cuda_quant(const set_rows_context& ctx, cudaStream_t stream) {

	static constexpr size_t qk = block_type::block_size;
    GGML_ASSERT(ctx.src0_ne[0] % qk == 0);
    auto src0_data = make_strided_mdspan(static_cast<const src_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(static_cast<const idx_t*>(ctx.src1_d), ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<block_type*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    launch_functor(stream, std::make_tuple(ctx.src0_ne[3], ctx.src0_ne[2], ctx.src0_ne[1], ctx.src0_ne[0] / qk),
        [=] __device__(int64_t i03, int64_t i02, int64_t i01, int64_t i00) {
            i00 *= qk;
            const int64_t i12 = i03 % ctx.src1_ne[2];
            const int64_t i11 = i02 % ctx.src1_ne[1];
            const int64_t i10 = i01;

            const int64_t dst_row = src1_data(i12, i11, i10);
            quantize_block(&src0_data(i03, i02, i01, i00), &dst_data(i03, i02, dst_row, i00 /qk));
        }
    );
}

template <typename src_t, typename idx_t, typename dst_t>
void set_rows_cuda(const set_rows_context& ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const src_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(static_cast<const idx_t*>(ctx.src1_d), ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src0_ne[3], ctx.src0_ne[2], ctx.src0_ne[1], ctx.src0_ne[0]),
        [=] __device__(int64_t i03, int64_t i02, int64_t i01, int64_t i00) {
            const int64_t i12 = i03 % ctx.src1_ne[2];
            const int64_t i11 = i02 % ctx.src1_ne[1];
            const int64_t i10 = i01;

            const int64_t dst_row = src1_data(i12, i11, i10);
            dst_data(i03, i02, dst_row, i00) = ggml_cuda_cast<dst_t>(src0_data(i03, i02, i01, i00));
        }
    );
}

template <typename src_t, typename idx_t>
static void set_rows_cuda(const set_rows_context &ctx, cudaStream_t stream)
{
    if (ctx.dst_type == GGML_TYPE_F32) {
        set_rows_cuda<src_t, idx_t, float>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_F16) {
        set_rows_cuda<src_t, idx_t, half>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_BF16) {
        set_rows_cuda<src_t, idx_t, nv_bfloat16>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q4_0>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<src_t, idx_t, block_q4_1>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q5_0>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<src_t, idx_t, block_q5_1>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q8_0>(ctx, stream);
    }
    else if (ctx.dst_type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<src_t, idx_t, block_iq4_nl>(ctx, stream);
    }
    else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

void set_rows_cuda(const set_rows_context &ctx, cudaStream_t stream) {
    if (ctx.src1_type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, stream);
    }
    else {
        set_rows_cuda<float, int32_t>(ctx, stream);
    }
}