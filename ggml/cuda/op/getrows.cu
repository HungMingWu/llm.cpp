#define GGML_ASSERT(...)
#include <bit>
#include "cuda_func.h"
#include "block.h"
#include "common.cuh"
#include "dequantize.cuh"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

template <typename src0_t, typename dst_t>
void get_rows_cuda_float(const get_row_context &ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src1_ne[2], ctx.src1_ne[1], ctx.src1_ne[0], ctx.src0_ne[0]),
        [=] __device__(int64_t i12, int64_t i11, int64_t i10, int64_t i00) {
            const int i01 = src1_data(i12, i11, i10);

            dst_data(i12, i11, i10, i00) = ggml_cuda_cast<dst_t>(src0_data(i12, i11, i01, i00));
        }
    );
}

template <typename block_type, int qr, typename dst_t>
void get_rows_cuda(const get_row_context &ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const block_type*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    GGML_ASSERT(ctx.src0_ne[0] % 2 == 0);

    static constexpr size_t qk = block_type::block_size;
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
static void ggml_cuda_get_rows_switch_src0_type(const get_row_context &ctx, cudaStream_t stream)
{
    switch (ctx.src0_type) {
    case internal::GGML_TYPE_F16:
        get_rows_cuda_float<half, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_F32:
        get_rows_cuda_float<float, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_I32:
        get_rows_cuda_float<int32_t, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_BF16:
        get_rows_cuda_float<nv_bfloat16, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q4_0:
        get_rows_cuda<block_q4_0, QR4_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q4_1:
        get_rows_cuda<block_q4_1, QR4_1, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q5_0:
        get_rows_cuda<block_q5_0, QR5_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q5_1:
        get_rows_cuda<block_q5_1, QR5_1, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q8_0:
        get_rows_cuda<block_q8_0, QR8_0, dst_t>(ctx, stream);
        break;
    default:
        // TODO: k-quants
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
    static constexpr size_t CUDA_GET_ROWS_BACK_BLOCK_SIZE = 256;
    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ctx.ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ctx.ne1, 1);

    k_get_rows_back_float << <block_nums, block_dims, 0, stream >> > (ctx.src0_d, ctx.src1_d, ctx.dst_d, ctx.ne00, ctx.ne10);
}
