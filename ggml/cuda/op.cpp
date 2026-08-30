module;
#include <assert.h>
#include <bit>
#include <vector>
#include "common.h"
#include "op/cuda_func.h"

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

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

        const int64_t ne10_padded = GGML_PAD1(src1->ne[0], MATRIX_ROW_PADDING);

        const int64_t s01 = src0->nb[1] / ts_src0;
        const int64_t s1 = dst->nb[1] / ts_dst;
        const int64_t s02 = src0->nb[2] / ts_src0;
        const int64_t s2 = dst->nb[2] / ts_dst;
        const int64_t s03 = src0->nb[3] / ts_src0;
        const int64_t s3 = dst->nb[3] / ts_dst;

        const bool fallback = src0->ne[1] % 128 != 0;

        const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
        const size_t y_block_size = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
        const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ : QK8_1_MMQ;

        if (!ids) {
            const size_t nbytes_src1_q8_1 = src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * y_block_size / y_values_per_block +
                ggml_cuda_mmq_get_J_max(std::bit_cast<internal::ggml_type>(src0->type), fallback, cc, src1->ne[1]) * sizeof(block_q8_1_mmq);
            ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);
            ggml_cuda_pool_alloc<float> src1_scale(pool);
            if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
                src1_scale.alloc(src1->ne[3] * src1->ne[2] * src1->ne[1]);
            }

            {
                const int64_t s11 = src1->nb[1] / ts_src1;
                const int64_t s12 = src1->nb[2] / ts_src1;
                const int64_t s13 = src1->nb[3] / ts_src1;
                if (use_native_fp4) {
                    static constexpr size_t align_float8 = 32;
                    const bool use_aligned_float8 = utils::ggml_cuda_is_aligned(src1, align_float8);
                    static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                    quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, std::bit_cast<internal::ggml_type>(src0->type), use_aligned_float8, src1->ne[0], s11, s12, s13, ne10_padded,
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
            const int64_t s12 = use_native_fp4 ?
                src1->ne[1] * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (block_q8_1::block_size * sizeof(int));
            const int64_t s13 = src1->ne[2] * s12;

            const mmq_args args = {
                src0_d, std::bit_cast<internal::ggml_type>(src0->type), (const int*)src1_q8_1.ptr, nullptr, nullptr, dst_d,
                src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
                src0->ne[0], src0->ne[1], dst->ne[1], s01, src1->ne[1], s1,
                src0->ne[2], src1->ne[2], s02, s12, s2,
                src0->ne[3], src1->ne[3], s03, s13, s3,
                dst->ne[1] };
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

        // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
        // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
        const bool dedup_bcast = src1->ne[1] == 1 && n_expert_used > 1;

        {
            GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
            const int si1 = ids->nb[1] / ggml_element_size(ids);
            const int sis1 = src1->nb[2] / src1->nb[1];

            ggml_cuda_launch_mm_ids_helper((const int32_t*)ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
                src0->ne[2], src1->ne[2], n_expert_used, src1->ne[1], si1, sis1,  /*write_inverse =*/ dedup_bcast, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        const size_t nbytes_src1_q8_1 = src1->ne[2] * n_expert_used * ne10_padded * y_block_size / y_values_per_block +
            ggml_cuda_mmq_get_J_max(std::bit_cast<internal::ggml_type>(src0->type), fallback, cc, src1->ne[1]) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(pool);
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(src1->ne[2] * n_expert_used);
        }

        const int64_t ne11_flat = src1->ne[2] * n_expert_used;
        const int64_t ne12_flat = 1;
        const int64_t ne13_flat = 1;

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = utils::ggml_cuda_is_aligned(src1, align_float8);
                if (dedup_bcast) {
                    quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, std::bit_cast<internal::ggml_type>(src0->type), use_aligned_float8, src1->ne[0],
                        /*stride_token=*/s12, ne10_padded, src1->ne[2], ne11_flat, n_expert_used, stream);
                }
                else {
                    quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, std::bit_cast<internal::ggml_type>(src0->type), use_aligned_float8, src1->ne[0], s11, s12, s13,
                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
                }
            }
            else if (dedup_bcast) {
                quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0],
                    /*stride_token=*/s12, ne10_padded, src1->ne[2], ne11_flat, n_expert_used, stream);
            }
            else {
                quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), std::bit_cast<internal::ggml_type>(src0->type), src1->ne[0], s11, s12, s13,
                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        static_assert(QK_FP4_MMQ == 8 * block_mxfp4::block_size, "QK_FP4_MMQ needs to be 8 * block_mxfp4::block_size");
        const int64_t s12 = use_native_fp4 ? src1->ne[1] * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
            src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (block_q8_1::block_size * sizeof(int));
        const int64_t s13 = src1->ne[2] * s12;

        // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
        const mmq_args args = {
            src0_d, std::bit_cast<internal::ggml_type>(src0->type), (const int*)src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
            src1_scale.ptr,
            src0->ne[0], src0->ne[1], ne_get_rows, s01, ne_get_rows, s1,
            src0->ne[2], src0->ne[2], s02, s12, s2,
            src0->ne[3], src1->ne[3], s03, s13, s3,
            src1->ne[2] };

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

        solve_tri_context ctx{
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
                if (t == nullptr || ggml_is_quantized(t->type)) {
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
            .V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs)),
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
                .type = std::bit_cast<internal::ggml_type>(V->type),
                .block_size = ggml_blck_size(V->type),
                .type_size = ggml_type_size(V->type),
                .data = V->data,
                .elements = V->nelements(),
                .ne0 = V->ne[0],
                .ne1 = V->ne[1],
                .ne2 = V->ne[2],
                .ne3 = V->ne[3],
                .nb0 = V->nb[0],
                .nb1 = V->nb[1],
                .nb2 = V->nb[2],
                .nb3 = V->nb[3],
                .bs = ggml_blck_size(V->type),
                .ts = ggml_type_size(V->type),
                .element_size = ggml_element_size(V)
            },
            .mask = {
                .exist = mask != nullptr,
                .type = std::bit_cast<internal::ggml_type>(mask ? mask->type : GGML_TYPE_F32),
            },
            .sinks = {
                .data = sinks ? sinks->data : nullptr
            },
            .KQV = {
                .type = std::bit_cast<internal::ggml_type>(dst->type),
                .data = dst->data,
                .elements = dst->nelements(),
                .nbytes = dst->nbytes(),
                .nrows = ggml_nrows(dst),
                .ne0 = dst->ne[0],
                .ne1 = dst->ne[1],
                .ne2 = dst->ne[2],
                .ne3 = dst->ne[3]
            }
        };

        if (mask) {
            ctx.mask.data = mask->data;
            ctx.mask.ne[0] = mask->ne[0]; ctx.mask.ne[1] = mask->ne[1]; ctx.mask.ne[2] = mask->ne[2]; ctx.mask.ne[3] = mask->ne[3];
            ctx.mask.nb[0] = mask->nb[0]; ctx.mask.nb[1] = mask->nb[1]; ctx.mask.nb[2] = mask->nb[2]; ctx.mask.nb[3] = mask->nb[3];
        }

        switch (utils::ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case utils::BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case utils::BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx);
            break;
        case utils::BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx);
            break;
        case utils::BEST_FATTN_KERNEL_MMA_F16:
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
            .k = dst->ne[0],
            .nb01 = src0->nb[1]
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

        softmax_context ctx{
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

    void mean(ggml_cuda_pool& pool, cudaStream_t stream, bool any_cuda_graph_has_instance, bool any_cuda_graph_enabled, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src0));
        mean_context ctx{
            .pool = pool,
            .src0_d = (const float*)src0->data,
            .dst_d = (float*)dst->data,
            .ncols = src0->ne[0],
            .nrows = ggml_nrows(src0),
            .any_cuda_graph_has_instance = any_cuda_graph_has_instance,
            .any_cuda_graph_enabled = any_cuda_graph_enabled
        };
        mean_cuda(ctx, stream);
    }

    void rope(cudaStream_t stream, ggml_tensor* dst, bool forward, const ggml_tensor* set_rows) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const ggml_tensor* src2 = dst->src[2];

        GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        GGML_ASSERT(src0->type == dst->type);

        const int mode = std::bit_cast<int>(dst->op_params[2]);

        void* dst_d = dst->data;
        const int64_t* row_indices = nullptr;
        ggml_type dst_type = dst->type;

        if (set_rows != nullptr) {
            GGML_ASSERT(forward);
            dst_d = set_rows->data;
            row_indices = (const int64_t*)set_rows->src[1]->data;
            dst_type = set_rows->type;
        }

        rope_context ctx{
            .forward = forward,
            .is_neox = static_cast<bool>(mode & GGML_ROPE_TYPE_NEOX),
            .is_mrope = static_cast<bool>(mode & GGML_ROPE_TYPE_MROPE),
            .is_imrope = static_cast<bool>(mode == GGML_ROPE_TYPE_IMROPE),
            .is_vision = static_cast<bool>(mode == GGML_ROPE_TYPE_VISION),
            .src_type = std::bit_cast<internal::ggml_type>(src0->type),
            .dst_type = std::bit_cast<internal::ggml_type>(dst_type),
            .src_d = src0->data,
            .dst_d = dst_d,
            .src_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .src_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .dst_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] },
            .n_dims = std::bit_cast<int>(dst->op_params[1]),
            .n_offs = std::bit_cast<int>(dst->op_params[15]),
            .n_ctx_orig = std::bit_cast<int>(dst->op_params[4]),
            .pos = (const int32_t*)src1->data,
            // RoPE alteration for extended context
            .freq_base = std::bit_cast<float>(dst->op_params[5]),
            .freq_scale = std::bit_cast<float>(dst->op_params[6]),
            .ext_factor = std::bit_cast<float>(dst->op_params[7]),
            .attn_factor = std::bit_cast<float>(dst->op_params[8]),
            .beta_fast = std::bit_cast<float>(dst->op_params[9]),
            .beta_slow = std::bit_cast<float>(dst->op_params[10]),
            .freq_factors = (src2 != nullptr) ? (const float*)src2->data : nullptr,
            .row_indices = row_indices,
        };
        memcpy(&ctx.sections.v, (int32_t*)dst->op_params + 11, sizeof(int) * 4);

        GGML_ASSERT(ctx.n_offs >= 0);
        GGML_ASSERT(ctx.n_offs % 2 == 0);
        GGML_ASSERT(ctx.n_offs + ctx.n_dims <= ctx.src_ne[0]);

        if (ctx.is_mrope) {
            GGML_ASSERT(ctx.sections.v[0] > 0 || ctx.sections.v[1] > 0 || ctx.sections.v[2] > 0);
        }
        if (ctx.is_vision) {
            GGML_ASSERT(ctx.n_dims == ctx.src_ne[0] / 2);
            GGML_ASSERT(ctx.n_offs == 0);
        }

        rope_cuda(ctx, stream);
    }

    void gated_delta_net(cudaStream_t stream, ggml_tensor* dst) {
        const gated_delta_net_context ctx = utils::build_gated_delta_net_context(dst);

        gated_delta_net_cuda(ctx, stream);
    }

    void unary_mul(cudaStream_t stream, ggml_tensor* unary_node, ggml_tensor* mul_node) {
        // unary_node: UNARY op applied to unary_node->src[0]
        // mul_node:   MUL(a, b) where one of a/b is unary_node
        // Output goes to mul_node->data

        const ggml_tensor* unary_src = unary_node->src[0];  // input to the unary op
        const ggml_tensor* other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

        GGML_ASSERT(ggml_is_contiguous_1(unary_src));
        GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
        GGML_ASSERT(ggml_is_contiguous_1(other_src));
        GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
        GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

        GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16);
        GGML_ASSERT(unary_src->type == other_src->type);
        GGML_ASSERT(unary_src->type == mul_node->type);
        unary_mul_context ctx{
            .op = std::bit_cast<internal::ggml_unary_op>(ggml_get_unary_op(unary_node)),
            .unary_src_type = std::bit_cast<internal::ggml_type>(unary_src->type),
            .k = mul_node->nelements(),
            .nc = unary_src->ne[0],
            .unary_stride = static_cast<int64_t>(unary_src->nb[1]),
            .other_stride = static_cast<int64_t>(other_src->nb[1]),
            .unary_src_data = unary_src->data,
            .other_src_data = other_src->data,
            .mul_node_data = mul_node->data
        };
        unary_mul_cuda(ctx, stream);
    }

    void ssm_conv(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* bias_add_node, ggml_tensor* silu_dst) {
        const ggml_tensor* src0 = dst->src[0];  // conv_x
        const ggml_tensor* src1 = dst->src[1];  // conv1d.weight
        const bool fuse_bias = bias_add_node != nullptr;
        const bool fuse_silu = silu_dst != nullptr;

        // bias always comes with silu.
        GGML_ASSERT(!fuse_bias || fuse_silu);

        // The bias (when fused) is the non-conv operand of the ADD node.
        const ggml_tensor* bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

        // When fusing, write to silu_dst (the node downstream references).
        const ggml_tensor* out = fuse_silu ? silu_dst : dst;

        const int64_t nr = src0->ne[1];                // d_inner

        GGML_ASSERT(out->ne[0] == src0->ne[1]);
        GGML_ASSERT(src0->nb[0] == sizeof(float));
        GGML_ASSERT(src1->nb[0] == sizeof(float));
        GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(out->type == GGML_TYPE_F32);
        if (fuse_bias) {
            GGML_ASSERT(bias->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(bias));
            GGML_ASSERT(bias->nelements() == nr);
        }

        ssm_conv_context ctx{
            .fuse_silu = fuse_silu,
            .src0_d = (const float*)src0->data,
            .src1_d = (const float*)src1->data,
            .bias_d = fuse_bias ? (const float*)bias->data : nullptr,
            .out_d = (float*)out->data,
            .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .src1_ne = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
            .src1_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] },
            .out_ne = { out->ne[0], out->ne[1], out->ne[2], out->ne[3] },
            .out_nb = { out->nb[0], out->nb[1], out->nb[2], out->nb[3] },
            .nc = src1->ne[0],                // d_conv
            .nr = nr,
            .n_t = out->ne[1],                // tokens per sequence
            .n_s = out->ne[2]                 // number of sequences in the batch
        };
        ssm_conv_f32_cuda(ctx, stream);
    }

    void relu_sqr(cudaStream_t stream, ggml_tensor* relu_node, ggml_tensor* sqr_node) {
        const ggml_tensor* src = relu_node->src[0];

        GGML_ASSERT(ggml_is_contiguous(src));

        unary_context ctx = create(src, sqr_node, stream);
        relu_sqr_cuda(ctx);
    }

    void out_prod(ggml_cuda_pool& pool, cudaStream_t stream, cublasHandle_t handle, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(src0->ne[1] == src1->ne[1]);
        GGML_ASSERT(dst->ne[0] == src0->ne[0]);
        GGML_ASSERT(dst->ne[1] == src1->ne[0]);

        GGML_ASSERT(dst->ne[2] % src0->ne[2] == 0);
        GGML_ASSERT(dst->ne[3] % src0->ne[3] == 0);

        GGML_ASSERT(dst->ne[2] == src1->ne[2]);
        GGML_ASSERT(dst->ne[3] == src1->ne[3]);

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const int64_t lda = src0->nb[1] / sizeof(float);
        const int64_t ldc = dst->nb[1] / sizeof(float);

        const bool src1_T = ggml_is_transposed(src1);
        const cublasOperation_t src1_cublas_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
        const int64_t           ldb = (src1_T ? src1->nb[0] : src1->nb[1]) / sizeof(float);
        GGML_ASSERT((src1_T ? src1->nb[1] : src1->nb[0]) == sizeof(float));

        // data strides in dimensions 2/3
        const size_t s02 = src0->nb[2] / sizeof(float);
        const size_t s03 = src0->nb[3] / sizeof(float);
        const size_t s12 = src1->nb[2] / sizeof(float);
        const size_t s13 = src1->nb[3] / sizeof(float);
        const size_t s2 = dst->nb[2] / sizeof(float);
        const size_t s3 = dst->nb[3] / sizeof(float);

        // dps == dst per src0, used for group query attention
        const int64_t dps2 = dst->ne[2] / src0->ne[2];
        const int64_t dps3 = dst->ne[3] / src0->ne[3];

        if (dps2 == 1 && dst->ne[2] > 1) {
            // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
            GGML_ASSERT(dst->ne[2] <= std::numeric_limits<int>::max());
            const int batch_count = (int)dst->ne[2];
            for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
                CUBLAS_CHECK(
                    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                        dst->ne[0], dst->ne[1], src0->ne[1],
                        &alpha, src0_d + (i3 / dps3) * s03, lda, s02,
                        src1_d + i3 * s13, ldb, s12,
                        &beta, dst_d + i3 * s3, ldc, s2,
                        batch_count));
            }
        }
        else if (dst->ne[2] > 1 || dst->ne[3] > 1) {
            // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
            // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
            GGML_ASSERT(dst->ne[3] > 0);
            GGML_ASSERT(dst->ne[2] <= (int64_t)std::numeric_limits<int>::max() / dst->ne[3]);

            const int batch_count = (int)(dst->ne[2] * dst->ne[3]);
            ggml_cuda_pool_alloc<const float*> ptrs_a(pool, batch_count);
            ggml_cuda_pool_alloc<const float*> ptrs_b(pool, batch_count);
            ggml_cuda_pool_alloc<      float*> ptrs_c(pool, batch_count);

            k_compute_out_prod_ptrs(
                src0_d, src1_d, dst_d,
                ptrs_a.get(), ptrs_b.get(), ptrs_c.get(),
                dst->ne[2], dst->ne[3], dps2, dps3, s02, s03, s12, s13, s2, s3, stream);
            CUBLAS_CHECK(
                cublasSgemmBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                    dst->ne[0], dst->ne[1], src0->ne[1],
                    &alpha, ptrs_a.get(), lda,
                    ptrs_b.get(), ldb,
                    &beta, ptrs_c.get(), ldc,
                    batch_count));

        }
        else {
            // ne2 == 1 && ne3 == 1: single GEMM
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                    dst->ne[0], dst->ne[1], src0->ne[1],
                    &alpha, src0_d, lda,
                    src1_d, ldb,
                    &beta, dst_d, ldc));
        }
    }

    void snake_fused(cudaStream_t stream,
        const ggml_tensor* x,
        const ggml_tensor* a,
        const ggml_tensor* inv_b,
        ggml_tensor* dst)
    {
        snake_context ctx{
            .x_type = std::bit_cast<internal::ggml_type>(x->type),
            .x_d = x->data,
            .a_d = (const float*)a->data,
            .inv_b_d = (const float*)inv_b->data,
            .dst_d = dst->data,
            .T = (int)x->ne[0],
            .C = (int)x->ne[1]
        };
        snake_cuda(ctx, stream);
    }

    bool fwht(cudaStream_t stream, const ggml_tensor* src, ggml_tensor* dst) {
        GGML_ASSERT(ggml_are_same_shape(src, dst));
        if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
            return false;
        }
        const int     n = src->ne[0];
        const int64_t rows = ggml_nrows(src);

        const float* src_d = (const float*)src->data;
        float* dst_d = (float*)dst->data;
        const float scale = 1 / sqrtf(n);

        return fwht_cuda(n, src_d, dst_d, rows, scale, stream);
    }

    void concat(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == src1->type);
        GGML_ASSERT(dst->type == src0->type);

        const int32_t dim = ((int32_t*)dst->op_params)[0];
        concat_context ctx {
            .src0_type = std::bit_cast<internal::ggml_type>(src0->type),
            .dim = dim,
            .src0_d = src0->data,
            .src1_d = src1->data,
            .dst_d = dst->data,
            .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .src1_ne = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
            .src1_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] },
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] }
        };
        if (ggml_is_quantized(src0->type)) {
            ctx.src0_ne[0] /= ggml_blck_size(src0->type);
            ctx.src1_ne[0] /= ggml_blck_size(src1->type);
            ctx.dst_ne[0] /= ggml_blck_size(dst->type);
            if (dim == 3) {
                GGML_ASSERT(ggml_is_contiguous(src0));
                GGML_ASSERT(ggml_is_contiguous(src1));
            }
            else {
                GGML_ASSERT(ggml_is_contiguous_to_3(src0));
                GGML_ASSERT(ggml_is_contiguous_to_3(src1));
            }
            GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
            GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

            // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
            concat_cuda(ctx, stream);
        }
        else {
            concat_cuda(ctx, stream);
        }
    }

    void col2im_1d(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));

        col2im_1d_context ctx{
            .src0_type = std::bit_cast<internal::ggml_type>(src0->type),
            .src0_d = src0->data,
            .dst_d = dst->data,
            .s0 = std::bit_cast<int32_t>(dst->op_params[0]),
            .OC = std::bit_cast<int32_t>(dst->op_params[1]),
            .p0 = std::bit_cast<int32_t>(dst->op_params[2]),
            .K_OC = (int)src0->ne[0],
            .T_in = (int)src0->ne[1],
            .T_out = (int)dst->ne[0]
        };
        col2im_1d_cuda(ctx, stream);
    }

    void lightning_indexer(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* w = dst->src[2]; // weights
        const ggml_tensor* m = dst->src[3]; // mask

        GGML_ASSERT(dst->type == internal::GGML_TYPE_F32);
        GGML_ASSERT(q->type == internal::GGML_TYPE_F32);
        GGML_ASSERT(w->type == internal::GGML_TYPE_F32);
        GGML_ASSERT(m->type == internal::GGML_TYPE_F16);

        // input tensor rows must be contiguous
        GGML_ASSERT(q->nb[0] == ggml_type_size(q->type));
        GGML_ASSERT(k->nb[0] == ggml_type_size(k->type));
        GGML_ASSERT(w->nb[0] == ggml_type_size(w->type));
        GGML_ASSERT(m->nb[0] == ggml_type_size(m->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(dst->nb[0] == sizeof(float));
        GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
        GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
        GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

        const int n_embd = q->ne[0];
        const int n_head = q->ne[1];
        const int n_batch = q->ne[2];
        const int n_stream = q->ne[3];
        const int n_kv = k->ne[2];

        const int device = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[device].cc;
        bool use_wmma_kernel = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16)
            use_wmma_kernel = true;
#endif
        lightning_indexer_context ctx {
            .n_embd = n_embd,
            .n_head = n_head,
            .n_batch = n_batch,
            .n_stream = n_stream,
            .n_kv = n_kv,
            .use_wmma_kernel = use_wmma_kernel,
			.k_type = std::bit_cast<internal::ggml_type>(k->type),
            .q_d = (const float*)q->data,
            .k_d = (const char*)k->data,
            .w_d = (const float*)w->data,
            .m_d = (const half*)m->data,
            .dst_d = (float*)dst->data,
			.m_ne = { m->ne[0], m->ne[1], m->ne[2], m->ne[3] },
			.m_nb = { m->nb[0], m->nb[1], m->nb[2], m->nb[3] },
			.k_nb = { k->nb[0], k->nb[1], k->nb[2], k->nb[3] },
			.w_nb = { w->nb[0], w->nb[1], w->nb[2], w->nb[3] },
			.q_nb = { q->nb[0], q->nb[1], q->nb[2], q->nb[3] },
			.dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] }
        };
        lightning_indexer_cuda(ctx, stream);
    }

    void dsv4_hc_comb(cudaStream_t stream, ggml_tensor* dst) {
        static constexpr int DSV4_HC = 4;
        const ggml_tensor* mixes = dst->src[0];
        const ggml_tensor* scale = dst->src[1];
        const ggml_tensor* base = dst->src[2];

        GGML_ASSERT(mixes->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);
        GGML_ASSERT(base->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        constexpr int64_t hc_mix_dim = (2 + DSV4_HC) * DSV4_HC;

        GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
        GGML_ASSERT(dst->ne[0] == DSV4_HC);
        GGML_ASSERT(dst->ne[1] == DSV4_HC);
        GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
        GGML_ASSERT(scale->ne[0] >= 3);
        GGML_ASSERT(base->ne[0] == hc_mix_dim);

        dsv4_hc_comb_context ctx {
            .n_tokens = mixes->ne[1],
			.mixes_data = (const float*)mixes->data,
            .scale_data = (const float*)scale->data,
            .base_data = (const float*)base->data,
            .dst_data = (float*)dst->data,
            .eps = std::bit_cast<float>(dst->op_params[0]),
			.n_iter = std::bit_cast<int32_t>(dst->op_params[1]),
			.nbm0 = mixes->nb[0],
			.nbm1 = mixes->nb[1],
            .nbb0 = base->nb[0],
			.nbs0 = scale->nb[0],
			.nbs1 = scale->nb[1],
			.nbd0 = dst->nb[0],
			.nbd1 = dst->nb[1],
			.nbd2 = dst->nb[2]
        };
        dsv4_hc_comb_cuda(ctx, stream);
    }

    void dsv4_hc_pre(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* x = dst->src[0];
        const ggml_tensor* weights = dst->src[1];

        GGML_ASSERT(x->type == GGML_TYPE_F32);
        GGML_ASSERT(weights->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        dsv4_hc_pre_context ctx {
            .n_embd = x->ne[0],
            .hc = x->ne[1],
            .n_tokens = x->ne[2],
			.x_data = (const float*)x->data,
			.weights_data = (const float*)weights->data,
			.dst_data = (float*)dst->data,
			.nbx0 = x->nb[0],
			.nbx1 = x->nb[1],
			.nbx2 = x->nb[2],
            .nbw0 = weights->nb[0],
			.nbw1 = weights->nb[1],
			.nbd0 = dst->nb[0],
            .nbd1 = dst->nb[1]
        };
        dsv4_hc_pre_cuda(ctx, stream);
    }

    void dsv4_hc_post(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* x = dst->src[0];
        const ggml_tensor* residual = dst->src[1];
        const ggml_tensor* post = dst->src[2];
        const ggml_tensor* comb = dst->src[3];

        GGML_ASSERT(x->type == GGML_TYPE_F32);
        GGML_ASSERT(residual->type == GGML_TYPE_F32);
        GGML_ASSERT(post->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        dsv4_hc_post_context ctx {
            .n_embd = x->ne[0],
            .n_tokens = x->ne[1],
            .hc = residual->ne[1],
			.x_data = (const float*)x->data,
			.residual_data = (const float*)residual->data,
			.post_data = (const float*)post->data,
			.comb_data = (const float*)comb->data,
			.dst_data = (float*)dst->data,
			.nbx0 = x->nb[0],
			.nbx1 = x->nb[1],
			.nbr0 = residual->nb[0],
			.nbr1 = residual->nb[1],
			.nbr2 = residual->nb[2],
			.nbp0 = post->nb[0],
			.nbp1 = post->nb[1],
			.nbc0 = comb->nb[0],
			.nbc1 = comb->nb[1],
			.nbc2 = comb->nb[2],
			.nbd0 = dst->nb[0],
			.nbd1 = dst->nb[1],
			.nbd2 = dst->nb[2]
        };
        dsv4_hc_post_cuda(ctx, stream);
    }

    void conv2d(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* kernel = dst->src[0];
        const ggml_tensor* input = dst->src[1];

        GGML_ASSERT(ggml_is_contiguous(input));
        GGML_ASSERT(ggml_is_contiguous(kernel));
        GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

        // same number of input channels
        GGML_ASSERT(input->ne[2] == kernel->ne[2]);

        conv2d_context ctx{
            .kernel_type = std::bit_cast<internal::ggml_type>(kernel->type),
            .N = input->ne[3],   // n_batches
            .CIn = input->ne[2],   // input_channels
            .IH = input->ne[1],   // input_h
            .IW = input->ne[0],  // input_w
            .COut = kernel->ne[3],  // ouptut_chanles
            .OH = dst->ne[1],     // output_h
            .OW = dst->ne[0],     // output_w
            .KH = kernel->ne[1],  // kernel_h
            .KW = kernel->ne[0],  // kernel_w
            .input_d = (const float*)input->data,
            .kernel_d = kernel->data,
            .output_d = (float*)dst->data,
            .stride_w = dst->op_params[0],
            .stride_h = dst->op_params[1],
            .pad_w = dst->op_params[2],
            .pad_h = dst->op_params[3],
            .dilation_w = dst->op_params[4],
            .dilation_h = dst->op_params[5]
        };

        conv2d_cuda(ctx, stream);
    }

    void ssm_scan(ggml_cuda_pool& pool, cublasHandle_t handle, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];  // s
        const ggml_tensor* src1 = dst->src[1];  // x
        const ggml_tensor* src2 = dst->src[2];  // dt
        const ggml_tensor* src3 = dst->src[3];  // A
        const ggml_tensor* src4 = dst->src[4];  // B
        const ggml_tensor* src5 = dst->src[5];  // C
        const ggml_tensor* src6 = dst->src[6];  // ids

        const int64_t nc = src0->ne[0];  // d_state
        const int64_t nr = src0->ne[1];  // head_dim or 1
        const int64_t nh = src1->ne[1];  // n_head
        const int64_t ng = src4->ne[1];  // n_group
        const int64_t n_t = src1->ne[2];  // number of tokens per sequence
        const int64_t n_s = src1->ne[3];  // number of sequences in the batch
        const int32_t K_param = std::bit_cast<int32_t>(dst->op_params[0]);
        const int64_t K = K_param > 0 ? K_param : 1;

        const int64_t s_off = src1->nelements() * sizeof(float);

        GGML_ASSERT(src1->nelements() + K * nc * nr * nh * n_s == dst->nelements());
        GGML_ASSERT(src0->nb[0] == sizeof(float));
        GGML_ASSERT(src1->nb[0] == sizeof(float));
        GGML_ASSERT(src2->nb[0] == sizeof(float));
        GGML_ASSERT(src3->nb[0] == sizeof(float));
        GGML_ASSERT(src4->nb[0] == sizeof(float));
        GGML_ASSERT(src5->nb[0] == sizeof(float));
        GGML_ASSERT(src6->nb[0] == sizeof(int32_t));
        GGML_ASSERT(src3->ne[0] == 1 || K == 1);

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        const float* src2_d = (const float*)src2->data;
        const float* src3_d = (const float*)src3->data;
        const float* src4_d = (const float*)src4->data;
        const float* src5_d = (const float*)src5->data;
        const int32_t* src6_d = (const int32_t*)src6->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src6->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        // Byte strides are narrowed to int for both scan and SSD paths.
        GGML_ASSERT(src0->nb[2] <= (size_t)INT_MAX);
        GGML_ASSERT(src0->nb[3] <= (size_t)INT_MAX);
        GGML_ASSERT(src1->nb[2] <= (size_t)INT_MAX);
        GGML_ASSERT(src1->nb[3] <= (size_t)INT_MAX);
        GGML_ASSERT(src2->nb[1] <= (size_t)INT_MAX);
        GGML_ASSERT(src2->nb[2] <= (size_t)INT_MAX);
        GGML_ASSERT(src3->nb[1] <= (size_t)INT_MAX);
        GGML_ASSERT(src4->nb[2] <= (size_t)INT_MAX);
        GGML_ASSERT(src4->nb[3] <= (size_t)INT_MAX);
        GGML_ASSERT(src5->nb[2] <= (size_t)INT_MAX);
        GGML_ASSERT(src5->nb[3] <= (size_t)INT_MAX);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define SSM_SSD_MIN_TOKENS 128
#define SSM_SSD_DT_BLOCK     256
#define SSM_SSD_DT_MAX_ITEMS  32

        // Maximum tokens the SSD path supports, derived from the prepare_dt kernel block capacity.
#define SSM_SSD_MAX_TOKENS (SSM_SSD_DT_BLOCK * SSM_SSD_DT_MAX_ITEMS)
        // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
        // Requires NVIDIA Turing+ otherwise fallback to scan.
        const bool is_mamba2 = (src3->nb[1] == sizeof(float));
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const bool use_ssd = is_mamba2 && n_t > SSM_SSD_MIN_TOKENS
            && K == 1
            && n_t <= SSM_SSD_MAX_TOKENS
            && GGML_CUDA_CC_IS_NVIDIA(cc)
            && cc >= GGML_CUDA_CC_TURING
            && nr % 8 == 0;  // cuBLAS requires 8-element (16-byte) alignment

        if (use_ssd) {
            // ssm_ssd_init_state_kernel uses flat linear indexing within each sequence,
            // so src0 must be fully contiguous across all inner dimensions.
            // The scan path handles non-contiguous nb[2] via src0_nb2 but does not handle nb[1].
            GGML_ASSERT(src0->nb[1] == nc * sizeof(float));
            GGML_ASSERT(src0->nb[2] == nc * nr * sizeof(float));

            ssm_scan_ssd_f32_cuda(pool, handle, stream,
                src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                (int64_t)(src0->nb[3] / sizeof(float)),
                (int)(src1->nb[2] / sizeof(float)), (int)(src1->nb[3] / sizeof(float)),
                (int)(src2->nb[1] / sizeof(float)), (int)(src2->nb[2] / sizeof(float)),
                (int)(src3->nb[1] / sizeof(float)),
                (int)(src4->nb[2] / sizeof(float)), (int)(src4->nb[3] / sizeof(float)),
                (int)(src5->nb[2] / sizeof(float)), (int)(src5->nb[3] / sizeof(float)),
                s_off, nc, nr, nh, ng, n_t, n_s);
            return;
        }
#endif
        ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
            src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
            s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
    }

    void rwkv_wkv7(cudaStream_t stream, ggml_tensor* dst) {
        const int64_t C = dst->ne[0];
        const int64_t HEADS = dst->src[0]->ne[1];
        static constexpr size_t CUDA_WKV_BLOCK_SIZE = 64;

        GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
        GGML_ASSERT(C % HEADS == 0);
        GGML_ASSERT(C / HEADS == CUDA_WKV_BLOCK_SIZE || C / HEADS == CUDA_WKV_BLOCK_SIZE * 2);

        rwkv_wkv7_context ctx{
            .n_seqs = dst->src[6]->ne[1],
            .T = dst->src[0]->ne[2],
            .C = C,
            .B = dst->src[6]->ne[1],
            .HEADS = HEADS,
            .r = (const float*)dst->src[0]->data,
            .w = (const float*)dst->src[1]->data,
            .k = (const float*)dst->src[2]->data,
            .v = (const float*)dst->src[3]->data,
            .a = (const float*)dst->src[4]->data,
            .b = (const float*)dst->src[5]->data,
            .s = (const float*)dst->src[6]->data,
            .dst = (float*)dst->data
        };
        rwkv_wkv7_cuda(ctx, stream);
    }

    void rms_norm_mul_rope_fused(cudaStream_t stream,
        ggml_tensor* rms_norm, ggml_tensor* mul, ggml_tensor* rope, ggml_tensor* set_rows) {
        const ggml_tensor* x = rms_norm->src[0];
        const ggml_tensor* mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

		const float eps = std::bit_cast<float>(rms_norm->op_params[0]);
        GGML_ASSERT(eps >= 0.0f);

        GGML_ASSERT(x->type == GGML_TYPE_F32);
        GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
        GGML_ASSERT(rope->type == GGML_TYPE_F32);

        void* dst_d = rope->data;
        ggml_type       dst_type = rope->type;
        const int64_t* row_indices = nullptr;
        int             set_rows_stride = 0;

        if (set_rows != nullptr) {
            dst_d = set_rows->data;
            dst_type = set_rows->type;
            row_indices = (const int64_t*)set_rows->src[1]->data;
            set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
        }
  
        const int mode = std::bit_cast<int>(rope->op_params[2]);

        const int64_t ts0 = ggml_type_size(x->type);
        GGML_ASSERT(x->nb[0] == ts0);

        const int64_t ts_mul = ggml_type_size(mul_src->type);
        GGML_ASSERT(mul_src->nb[0] == ts_mul);

        const int64_t ts_dst = ggml_type_size(rope->type);

        rms_norm_mul_rope_fused_context ctx {
			.dst_type = std::bit_cast<internal::ggml_type>(dst_type),
			.x_data = (const float*)x->data,
            .dst_d = dst_d,
			.x_ne = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] },
            .s01 = (int64_t)x->nb[1] / ts0,
            .s02 = (int64_t)x->nb[2] / ts0,
            .s03 = (int64_t)x->nb[3] / ts0,
            .s1 = (int64_t)rope->nb[1] / ts_dst,
            .s2 = (int64_t)rope->nb[2] / ts_dst,
            .s3 = (int64_t)rope->nb[3] / ts_dst,
            .eps = eps,
			.mul_src_data = mul_src->data,
            .mul_s01 = (int64_t)mul_src->nb[1] / ts_mul,
			.mul_s02 = (int64_t)mul_src->nb[2] / ts_mul,
			.mul_s03 = (int64_t)mul_src->nb[3] / ts_mul,
			.mul_src_ne = { mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3] },
            .n_dims = std::bit_cast<int32_t>(rope->op_params[1]),
            .pos = (const int32_t*)rope->src[1]->data,
            .freq_factors = rope->src[2] != nullptr ? (const float*)rope->src[2]->data : nullptr,
            .n_ctx_orig = std::bit_cast<int>(rope->op_params[4]),
			.freq_base = std::bit_cast<float>(rope->op_params[5]),
            .freq_scale = std::bit_cast<float>(rope->op_params[6]),
            .ext_factor = std::bit_cast<float>(rope->op_params[7]),
            .attn_factor = std::bit_cast<float>(rope->op_params[8]),
            .beta_fast = std::bit_cast<float>(rope->op_params[9]),
            .beta_slow = std::bit_cast<float>(rope->op_params[10]),
            .row_indices = row_indices,
            .set_rows_stride = set_rows_stride,
            .is_neox = static_cast<bool>(mode & GGML_ROPE_TYPE_NEOX)
        };
        rms_norm_mul_rope_fused_cuda(ctx, stream);
    }

    void pool1d(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int32_t* opts = (const int32_t*)dst->op_params;
        enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
        const int k0 = opts[1];
        const int s0 = opts[2];
        const int p0 = opts[3];

        const int64_t IW = src0->ne[0];
        const int64_t OW = dst->ne[0];
        const int64_t nr = ggml_nrows(src0);

        const int parallel_elements = (int)(nr * OW);

        pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, std::bit_cast<internal::ggml_op_pool>(op), stream);
    }

    void pool2d(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int32_t* opts = (const int32_t*)dst->op_params;

        const int64_t N = dst->ne[3];
        const int64_t OC = dst->ne[2];
        const int64_t OH = dst->ne[1];
        const int64_t OW = dst->ne[0];

        pool2d_context ctx{
            .IH = src0->ne[1],
            .IW = src0->ne[0],
            .N = N,
            .OC = OC,
            .OH = OH,
            .OW = OW,
            .KH = opts[2],
            .KW = opts[1],
            .SH = opts[4],
            .SW = opts[3],
            .PH = opts[6],
            .PW = opts[5],
            .parallel_elements = N * OC * OH * OW,
            .src0_d = (const float*)src0->data,
            .dst_d = (float*)dst->data,
            .op = std::bit_cast<internal::ggml_op_pool>(opts[0])
        };

        pool2d_nchw_kernel_cuda(ctx, stream);
    }

    void get_rows_back(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0]; // gradients of forward pass output
        const ggml_tensor* src1 = dst->src[1]; // src1 in forward pass

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(src0->ne[2] * src0->ne[3] == 1);
        GGML_ASSERT(src1->ne[2] * src1->ne[3] == 1);
        GGML_ASSERT(dst->ne[2] * dst->ne[3] == 1);
        get_row_back_context context{
            .src0_d = (const float*)src0->data,
            .src1_d = (const int32_t*)src1->data,
            .dst_d = (float*)dst->data,
            .src0_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] },
            .src0_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] },
            .src1_ne = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
            .src1_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3] },
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] },
        };
        get_rows_back_cuda(context, stream);
    }
}
