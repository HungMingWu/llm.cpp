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

        // TODO: tighter pool buffer size vs q8 path
        const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;

        if (!ids) {
            const size_t nbytes_src1_q8_1 = src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1 +
                get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
            ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);

            {
                const int64_t s11 = src1->nb[1] / ts_src1;
                const int64_t s12 = src1->nb[2] / ts_src1;
                const int64_t s13 = src1->nb[3] / ts_src1;
                if (use_native_mxfp4) {
                    static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                    quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(),
                        std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0], s11, s12, s13, ne10_padded,
                        src1->ne[1], src1->ne[2], src1->ne[3], stream);

                }
                else {
                    quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(),
                        std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0], s11, s12, s13, ne10_padded,
                        src1->ne[1], src1->ne[2], src1->ne[3], stream);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            // Stride depends on quantization format
            const int64_t s12 = use_native_mxfp4 ?
                src1->ne[1] * ne10_padded * sizeof(block_fp4_mmq) /
                (8 * block_mxfp4::block_size * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                :
                src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
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
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_mxfp4) {
                quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(),
                    std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0], s11, s12, s13,
                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
            else {
                quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(),
                    std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0], s11, s12, s13,
                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        const int64_t s12 = use_native_mxfp4 ? src1->ne[1] * ne10_padded * sizeof(block_fp4_mmq) / (8 * block_mxfp4::block_size * sizeof(int)) :
            src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
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

        solve_tri_context ctx {
            .A = (const float*)src0->data,
            .A_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .A_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .B = (const float*)src1->data,
            .B_ne = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
            .B_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] },
            .X = (float*)dst->data,
            .X_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
            .X_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] }
        };
        solve_tri_f32_cuda(ctx, pool, cublas_handle, stream);
    }

    void flash_attn_ext(int device, ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* Q = dst->src[0];
        const ggml_tensor* K = dst->src[1];
        const ggml_tensor* V = dst->src[2];
        const ggml_tensor* mask = dst->src[3];
        const ggml_tensor* sinks = dst->src[4];

        ggml_cuda_set_device(device);

        static constexpr int64_t FATTN_KQ_STRIDE = 256;
        float max_bias = std::bit_cast<float>(dst->op_params[1]);
        // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
        //     are put into the template specialization without GQA optimizations.
        auto use_gpa_opt = [=]() -> bool {
            for (const ggml_tensor* t : { Q, K, V, mask }) {
                if (t == nullptr) {
                    continue;
                }
                for (size_t i = 1; i < GGML_MAX_DIMS; ++i)
                    if (t->nb[i] % 16 != 0)
                        return false;
            }
            return mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
        }();
        flash_attn_ext_context ctx{
            .device = device,
            .main_stream = stream,
            .pool = &pool,
            .scale = std::bit_cast<float>(dst->op_params[0]),
            .max_bias = max_bias,
            .logit_softcap = std::bit_cast<float>(dst->op_params[2]),
            .precision = std::bit_cast<internal::ggml_prec>(dst->op_params[3]),
            .use_gqa_opt = use_gpa_opt,
            .Q = {
                .type = std::bit_cast<internal::ggml_type>(Q->type),
                .data = Q->data,
				.ne = { Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3] },
                .nb = { Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3] },
                .element_size = ggml_element_size(Q)
            },
            .K = {
                .type = std::bit_cast<internal::ggml_type>(K->type),
                .block_size = ggml_blck_size(K->type),
                .type_size = ggml_type_size(K->type),
                .data = K->data,
                .elements = K->nelements(),
                .ne0 = K->ne[0],
                .ne1 = K->ne[1],
                .ne2 = K->ne[2],
                .ne3 = K->ne[3],
                .nb0 = K->nb[0],
                .nb1 = K->nb[1],
                .nb2 = K->nb[2],
                .nb3 = K->nb[3],
                .bs = ggml_blck_size(K->type),
                .ts = ggml_type_size(K->type),
                .element_size = ggml_element_size(K)
            },
            .V = {
                .exist = V != nullptr,
                .type = std::bit_cast<internal::ggml_type>(V ? V->type : GGML_TYPE_F32),
                .block_size = V ? ggml_blck_size(V->type) : 0,
                .type_size = V ? ggml_type_size(V->type) : 0,
                .data = V ? V->data : nullptr,
                .elements = V ? V->nelements() : 0,
                .ne0 = V ? V->ne[0] : 0,
                .ne1 = V ? V->ne[1] : 0,
                .ne2 = V ? V->ne[2] : 0,
                .ne3 = V ? V->ne[3] : 0,
                .nb0 = V ? V->nb[0] : 0,
                .nb1 = V ? V->nb[1] : 0,
                .nb2 = V ? V->nb[2] : 0,
                .nb3 = V ? V->nb[3] : 00,
                .bs = V ? ggml_blck_size(V->type) : 0,
                .ts = V ? ggml_type_size(V->type) : 0,
                .element_size = V ? ggml_element_size(V) : 0
            },
            .mask = {
                .exist = mask != nullptr,
                .type = std::bit_cast<internal::ggml_type>(mask ? mask->type : GGML_TYPE_F32),
                .data = mask ? mask->data : nullptr,
                .ne0 = (mask) ? mask->ne[0] : 0,
                .ne1 = (mask) ? mask->ne[1] : 0,
                .ne2 = (mask) ? mask->ne[2] : 0,
                .ne3 = (mask) ? mask->ne[3] : 0,
                .nb0 = (mask) ? mask->nb[0] : 0,
                .nb1 = (mask) ? mask->nb[1] : 0,
                .nb2 = (mask) ? mask->nb[2] : 0,
                .nb3 = (mask) ? mask->nb[3] : 0
            },
            .sinks = {
                .data = sinks ? sinks->data : nullptr
            },
            .KQV = {
                .type = std::bit_cast<internal::ggml_type>(dst->type),
                .data = dst->data,
                .elements = dst->nelements(),
                .nrows = ggml_nrows(dst),
                .ne0 = dst->ne[0],
                .ne1 = dst->ne[1],
                .ne2 = dst->ne[2],
                .ne3 = dst->ne[3]
            }
        };

        switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx);
            break;
        }
    }

    void count_equal(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == src1->type);
        GGML_ASSERT(dst->type == GGML_TYPE_I64);

        GGML_ASSERT(ggml_are_same_shape(src0, src1));
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(src0->type == GGML_TYPE_I32);

        const int64_t ne = src0->nelements();
        GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
        count_equal_context context{
            .src0_d = (const int*)src0->data,
            .src1_d = (const int*)src1->data,
            .dst_d = (int64_t*)dst->data,
            .dst_size = dst->nbytes(),
            .ne = ne,
        };
        count_equal_cuda(context, stream);
    }

    void top_k(ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        // are these asserts truly necessary?
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src0));

        top_k_context ctx{
            .pool = pool,
            .src0_d = (const float*)src0->data,
            .dst_d = (int*)dst->data,
            .nrows = ggml_nrows(src0),
            .ncols = src0->ne[0],
            .k = dst->ne[0]
        };
        top_k_cuda(ctx, stream);
    }

    void cumsum(ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(src0->type == dst->type);
        cumsum_context ctx{
            .pool = pool,
            .src0_type = std::bit_cast<internal::ggml_type>(src0->type),
            .src0_d = src0->data,
            .dst_d = dst->data,
            .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] }
        };
        cumsum_cuda(ctx, stream);
    }

    void soft_max(ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const ggml_tensor* src2 = dst->src[2];

        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

        const int64_t nrows_x = ggml_nrows(src0);
        const int64_t nrows_y = src0->ne[1];

        const int64_t ne00 = src0->ne[0];

        float scale = std::bit_cast<float>(dst->op_params[0]);
        float max_bias = std::bit_cast<float>(dst->op_params[1]);

        const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

        const uint32_t n_head = src0->ne[2];
        const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));

        const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

        softmax_context ctx {
            .pool = pool,
            .src0_d = src0_d,
            .dst_d = dst_d,
            .ne00 = ne00,
            .nrows_x = nrows_x,
            .nrows_y = nrows_y,
            .scale = scale,
            .max_bias = max_bias,
            .use_f16 = use_f16,
            .params = {
                .nheads = src0->ne[2],
                .n_head_log2 = n_head_log2,
                .ncols = ne00,
                .nrows_x = nrows_x,
                .nrows_y = nrows_y,
                .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]	},
                .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]	},
                .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]	},
                .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]	},
                .scale = scale,
                .max_bias = max_bias,
                .m0 = m0,
                .m1 = m1
            }
        };
        if (src1) {
            ctx.src1_d = src1->data;
            ctx.params.src1_ne[0] = src1->ne[0];
            ctx.params.src1_ne[1] = src1->ne[1];
            ctx.params.src1_ne[2] = src1->ne[2];
            ctx.params.src1_ne[3] = src1->ne[3];
            ctx.params.src1_nb[0] = src1->nb[0];
            ctx.params.src1_nb[1] = src1->nb[1];
            ctx.params.src1_nb[2] = src1->nb[2];
            ctx.params.src1_nb[3] = src1->nb[3];
        }
        else {
            ctx.src1_d = nullptr;
        }
        if (src2) {
            ctx.src2_d = (const float*)src2->data;
            ctx.params.src2_ne[0] = src2->ne[0];
            ctx.params.src2_ne[1] = src2->ne[1];
            ctx.params.src2_ne[2] = src2->ne[2];
            ctx.params.src2_ne[3] = src2->ne[3];
        }
        else {
            ctx.src2_d = nullptr;
            ctx.params.src2_ne[0] = ctx.params.src2_ne[1] = ctx.params.src2_ne[2] = ctx.params.src2_ne[3] = 0;
        }
        soft_max_f32_cuda(ctx, stream);
    }

}