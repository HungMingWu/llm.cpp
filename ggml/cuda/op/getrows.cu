#define GGML_ASSERT(...)
#include <bit>
#include "cuda_func.h"
#include "block.h"
#include "common.cuh"
#include "dequantize.cuh"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

template <typename dst_t>
static __global__ void k_get_rows_float_vec(
        mdspan_stride_t<const dst_t, 4> src0_data, mdspan_stride_t<const int32_t, 3> src1_data, mdspan_stride_t<dst_t, 4> dst_data,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_data(i12, i11, i10);

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (&dst_data(i12, i11, i10, 0));
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)(&src0_data(i12, i11, i01, 0));

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

template <typename src0_t, typename dst_t>
void get_rows_cuda_float(const get_row_context &ctx, cudaStream_t stream) {
    auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<3>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    if constexpr (std::is_same_v<src0_t, dst_t>) {
        assert(ctx.src1_ne[2] > 0);
        assert(ctx.src1_ne[1] <= std::numeric_limits<uint32_t>::max() / ctx.src1_ne[2]);
        const uint3 ne12_fdv = init_fastdiv_values(ctx.src1_ne[2]);
        constexpr size_t CUDA_GET_ROWS_BLOCK_SIZE = 256;
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ctx.src0_ne[0] / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ctx.src1_ne[0] * ctx.src1_ne[1] * ctx.src1_ne[2] >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ctx.src0_ne[0] % VEC == 0) &&
            (ctx.src0_nb[1] % 16 == 0) && (ctx.src0_nb[2] % 16 == 0) && (ctx.src0_nb[3] % 16 == 0) &&
            (ctx.dst_nb[1]  % 16 == 0) && (ctx.dst_nb[2]  % 16 == 0) && (ctx.dst_nb[3]  % 16 == 0) &&
            (((uintptr_t) ctx.src0_d) % 16 == 0) && (((uintptr_t) ctx.dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
            const dim3 block_nums(ctx.src1_ne[0], std::min<int64_t>(block_num_y, UINT16_MAX), std::min<int64_t>(ctx.src1_ne[1]*ctx.src1_ne[2], UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                src0_data, src1_data, dst_data,
                ne00v, ctx.src1_ne[1], ne12_fdv);
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

template <typename src_t, typename dst_t>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize(static_cast<const src_t*>(src0_row), ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
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
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template <int block_dim, typename src_t, typename dst_t>
void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, void * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, std::min<int64_t>(nsb, UINT16_MAX), std::min<int64_t>(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<src_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, static_cast<dst_t *>(dst_d),
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
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
    case internal::GGML_TYPE_Q1_0:
        get_rows_cuda<block_q1_0, QR1_0, dst_t>(ctx, stream);
        break;
    case internal::GGML_TYPE_Q2_0:
        get_rows_cuda<block_q2_0, QR2_0, dst_t>(ctx, stream);
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
    case internal::GGML_TYPE_Q2_K:
        get_rows_cuda_kq<64, block_q2_K, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_Q3_K:
        get_rows_cuda_kq<64, block_q3_K, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_Q4_K:
        get_rows_cuda_kq<32, block_q4_K, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_Q5_K:
        get_rows_cuda_kq<64, block_q5_K, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_Q6_K:
        get_rows_cuda_kq<64, block_q6_K, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ2_XXS:
        get_rows_cuda_kq<32, block_iq2_xxs, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ2_XS:
        get_rows_cuda_kq<32, block_iq2_xs, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ2_S:
        get_rows_cuda_kq<32, block_iq2_s, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ3_XXS:
        get_rows_cuda_kq<32, block_iq3_xxs, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ3_S:
        get_rows_cuda_kq<32, block_iq3_s, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ1_S:
        get_rows_cuda_kq<32, block_iq1_s, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ1_M:
        get_rows_cuda_kq<32, block_iq1_m, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ4_NL:
        get_rows_cuda_kq<32, block_iq4_nl, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_IQ4_XS:
        get_rows_cuda_kq<32, block_iq4_xs, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
        break;
    case internal::GGML_TYPE_MXFP4:
        get_rows_cuda_kq<32, block_mxfp4, dst_t>(ctx.src0_d, ctx.src1_d, ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3], ctx.src1_ne[0], ctx.src1_ne[1], ctx.src1_ne[2], ctx.src1_nb[0], ctx.src1_nb[1], ctx.src1_nb[2], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3], stream);
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
    static constexpr size_t CUDA_GET_ROWS_BACK_BLOCK_SIZE = 256;
    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ctx.ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, std::min(ctx.ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float << <block_nums, block_dims, 0, stream >> > (ctx.src0_d, ctx.src1_d, ctx.dst_d, ctx.ne00, ctx.ne10, ctx.ne1);
}
