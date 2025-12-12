module;
#include <assert.h>
#include <bit>
#include <vector>
#include "common.h"
#include "op/cuda_func.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :cuda.op;

namespace op {

    void mul_mat_q(
        ggml_cuda_pool& pool, cudaStream_t stream, const ggml_tensor* ids, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

        const size_t ts_src0 = ggml_type_size(src0->type);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const size_t ts_dst = ggml_type_size(dst->type);

        GGML_ASSERT(src0->nb[0] == ts_src0);
        GGML_ASSERT(src1->nb[0] == ts_src1);
        GGML_ASSERT(dst->nb[0] == ts_dst);
        GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

        const char* src0_d = (const char*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        // If src0 is a temporary compute buffer, clear any potential padding.
        if (src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size_data = src0->nbytes();
            const size_t size_alloc = src0->buffer->get_alloc_size(src0);
            if (size_alloc > size_data) {
                GGML_ASSERT(ggml_is_contiguously_allocated(src0));
                GGML_ASSERT(!src0->view_src);
                CUDA_CHECK(cudaMemsetAsync((char*)src0->data + size_data, 0, size_alloc - size_data, stream));
            }
        }

        const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);

        const int64_t s01 = src0->nb[1] / ts_src0;
        const int64_t s1 = dst->nb[1] / ts_dst;
        const int64_t s02 = src0->nb[2] / ts_src0;
        const int64_t s2 = dst->nb[2] / ts_dst;
        const int64_t s03 = src0->nb[3] / ts_src0;
        const int64_t s3 = dst->nb[3] / ts_dst;

        const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
            || GGML_CUDA_CC_IS_CDNA(cc);

        if (!ids) {
            const size_t nbytes_src1_q8_1 = src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1 +
                get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
            ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);

            {
                const int64_t s11 = src1->nb[1] / ts_src1;
                const int64_t s12 = src1->nb[2] / ts_src1;
                const int64_t s13 = src1->nb[3] / ts_src1;
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), std::bit_cast<internal::ggml_type>(src0->type),
                    src1->ne[0], s11, s12, s13, ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);
                CUDA_CHECK(cudaGetLastError());
            }

            const int64_t s12 = src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
            const int64_t s13 = src1->ne[2] * s12;

            const mmq_args args = {
                src0_d, std::bit_cast<internal::ggml_type>(src0->type), (const int*)src1_q8_1.ptr, nullptr, nullptr, dst_d,
                src0->ne[0], src0->ne[1], dst->ne[1], s01, src1->ne[1], s1,
                src0->ne[2], src1->ne[2], s02, s12, s2,
                src0->ne[3], src1->ne[3], s03, s13, s3,
                use_stream_k, dst->ne[1] };
            ggml_cuda_mul_mat_q_switch_type(pool, args, stream);
            return;
        }

        GGML_ASSERT(src1->ne[3] == 1);
        GGML_ASSERT(src1->nb[2] % src1->nb[1] == 0);
        GGML_ASSERT(dst->nb[2] % dst->nb[1] == 0);

        const int64_t n_expert_used = ids->ne[0];
        const int64_t ne_get_rows = src1->ne[2] * n_expert_used;
        GGML_ASSERT(dst->ne[1] == n_expert_used);

        ggml_cuda_pool_alloc<int32_t> ids_src1(pool, ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(pool, ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> expert_bounds(pool, src0->ne[2] + 1);

        {
            GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
            const int si1 = ids->nb[1] / ggml_element_size(ids);
            const int sis1 = src1->nb[2] / src1->nb[1];

            ggml_cuda_launch_mm_ids_helper((const int32_t*)ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
                src0->ne[2], src1->ne[2], n_expert_used, src1->ne[1], si1, sis1, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        const size_t nbytes_src1_q8_1 = src1->ne[2] * n_expert_used * ne10_padded * sizeof(block_q8_1) / QK8_1 +
            get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);

        const int64_t ne11_flat = src1->ne[2] * n_expert_used;
        const int64_t ne12_flat = 1;
        const int64_t ne13_flat = 1;

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[2] / ts_src1;
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), std::bit_cast<internal::ggml_type>(src0->type),
                src1->ne[0], s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        const int64_t s12 = src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = src1->ne[2] * s12;

        // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
        const mmq_args args = {
            src0_d, std::bit_cast<internal::ggml_type>(src0->type), (const int*)src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
            src0->ne[0], src0->ne[1], ne_get_rows, s01, ne_get_rows, s1,
            src0->ne[2], src0->ne[2], s02, s12, s2,
            src0->ne[3], src1->ne[3], s03, s13, s3,
            use_stream_k, src1->ne[2] };

        ggml_cuda_mul_mat_q_switch_type(pool, args, stream);
    }

    void upscale(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int mode_flags = dst->op_params[0];
        const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

        float sf0 = (float)dst->ne[0] / src0->ne[0];
        float sf1 = (float)dst->ne[1] / src0->ne[1];
        const float sf2 = (float)dst->ne[2] / src0->ne[2];
        const float sf3 = (float)dst->ne[3] / src0->ne[3];

        float pixel_offset = 0.5f;
        if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
            sf0 = (dst->ne[0] > 1 && src0->ne[0] > 1) ? (float)(dst->ne[0] - 1) / (src0->ne[0] - 1) : sf0;
            sf1 = (dst->ne[1] > 1 && src0->ne[1] > 1) ? (float)(dst->ne[1] - 1) / (src0->ne[1] - 1) : sf1;
            pixel_offset = 0.0f;
        }

        upscale_context ctx{
            .src0_d = (const float*)src0->data,
            .dst_d = (float*)dst->data,
            .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
            .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] },
            .sf0 = sf0,
            .sf1 = sf1,
            .sf2 = sf2,
            .sf3 = sf3
        };

        if (mode == GGML_SCALE_MODE_NEAREST) {
            upscale_f32_cuda(ctx, stream);
        }
        else if (mode == GGML_SCALE_MODE_BILINEAR) {
            upscale_f32_bilinear_cuda(ctx, pixel_offset, mode_flags & GGML_SCALE_FLAG_ANTIALIAS, stream);
        }
        else if (mode == GGML_SCALE_MODE_BICUBIC) {
            upscale_f32_bicubic_cuda(ctx, pixel_offset, stream);
        }
    }

    void pad(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src0));

        const bool circular = dst->op_params[8];
        pad_context ctx{
            .src0_d = (const float*)src0->data,
            .dst_d = (float*)dst->data,
            .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
            .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] },
            .lp0 = ((const int32_t*)(dst->op_params))[0],
            .rp0 = ((const int32_t*)(dst->op_params))[1],
            .lp1 = ((const int32_t*)(dst->op_params))[2],
            .rp1 = ((const int32_t*)(dst->op_params))[3],
            .lp2 = ((const int32_t*)(dst->op_params))[4],
            .rp2 = ((const int32_t*)(dst->op_params))[5],
            .lp3 = ((const int32_t*)(dst->op_params))[6],
            .rp3 = ((const int32_t*)(dst->op_params))[7],
            .circular = circular
        };
        pad_f32_cuda(ctx, stream);
    }

    void solve_tri(ggml_cuda_pool& pool, cublasHandle_t cublas_handle, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];  // A (triangular n x x matrix)
        const ggml_tensor* src1 = dst->src[1];  // B (right hand side of n x k equation columns)

        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));

        const int64_t n = src0->ne[0];
        const int64_t k = src1->ne[0];
        const int64_t ne02 = src0->ne[2];
        const int64_t ne03 = src0->ne[3];

        static constexpr int64_t MAX_N_FAST = 64;
        static constexpr int64_t MAX_K_FAST = 32;

        if (n <= MAX_N_FAST && k <= MAX_K_FAST) {
            solve_tri_f32_cuda((const float*)src0->data, (const float*)src1->data, (float*)dst->data, n, k,
                src0->ne[2], src0->ne[3], src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), dst->nb[2] / sizeof(float),
                dst->nb[3] / sizeof(float), stream);
        }
        else {
            solve_tri_f32_cublas(pool, cublas_handle, (const float*)src0->data, (const float*)src1->data, (float*)dst->data, n, k,
                ne02, ne03, src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), dst->nb[2] / sizeof(float),
                dst->nb[3] / sizeof(float), stream);
        }
    }
}