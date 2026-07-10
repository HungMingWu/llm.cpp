#include "common.cuh"
#include "cuda_func.h"
#include "mdspan_helper.h"
#include "launch.cuh"
#define GGML_ABORT(...)

template <typename T>
void concat_cuda(const concat_context &ctx, cudaStream_t stream)
{
    auto dst_data = make_strided_mdspan(static_cast<T*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    auto src0_data = make_strided_mdspan(static_cast<const T*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan(static_cast<const T*>(ctx.src1_d), ctx.src1_ne, ctx.src1_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            const auto value = [&]() {
                if (i0 < src0_data.extent(3) && i1 < src0_data.extent(2) && i2 < src0_data.extent(1) && i3 < src0_data.extent(0)) {
                    return src0_data(i3, i2, i1, i0);
                }
                else if (ctx.dim == 0) {
                    return src1_data(i3, i2, i1, i0 - src0_data.extent(3));
                }
                else if (ctx.dim == 1) {
                    return src1_data(i3, i2, i1 - src0_data.extent(2), i0);
                }
                else if (ctx.dim == 2) {
                    return src1_data(i3, i2 - src0_data.extent(1), i1, i0);
                }
                else {
                    return src1_data(i3 - src0_data.extent(0), i2, i1, i0);
                }
            }();

            dst_data(i3, i2, i1, i0) = value;
        }
    );
}

void concat_cuda(const concat_context &ctx, cudaStream_t stream) {
    switch (ctx.src0_type) {
        case internal::GGML_TYPE_F16:
        case internal::GGML_TYPE_BF16:
        case internal::GGML_TYPE_I16:
            concat_cuda<uint16_t>(ctx, stream);
            break;
        case internal::GGML_TYPE_I8:
            concat_cuda<uint8_t>(ctx, stream);
            break;
        case internal::GGML_TYPE_F32:
        case internal::GGML_TYPE_I32:
            concat_cuda<uint32_t>(ctx, stream);
            break;
        case internal::GGML_TYPE_I64:
            concat_cuda<uint64_t>(ctx, stream);
            break;
        case internal::GGML_TYPE_Q4_0:
            concat_cuda<block_q4_0>(ctx, stream);
            break;
        case internal::GGML_TYPE_Q4_1:
            concat_cuda<block_q4_1>(ctx, stream);
            break;
        case internal::GGML_TYPE_Q5_0:
            concat_cuda<block_q5_0>(ctx, stream);
            break;
        case internal::GGML_TYPE_Q5_1:
            concat_cuda<block_q5_1>(ctx, stream);
            break;
        case internal::GGML_TYPE_Q8_0:
            concat_cuda<block_q8_0>(ctx, stream);
            break;
    default:
            GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
            break;
    }
}