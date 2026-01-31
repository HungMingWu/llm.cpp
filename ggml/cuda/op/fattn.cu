#include <assert.h>
#include "cuda_func.h"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-vec.cuh"

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, int ntiles, bool use_logit_softcap, bool mla>
__launch_bounds__(nwarps* WARP_SIZE, 1)
static __global__ void flash_attn_ext_f16(
    [[maybe_unused]] const char* __restrict__ Q,
    [[maybe_unused]] const char* __restrict__ K,
    [[maybe_unused]] const char* __restrict__ V,
    [[maybe_unused]] const char* __restrict__ mask,
    [[maybe_unused]] const char* __restrict__ sinks,
    [[maybe_unused]] const int* __restrict__ KV_max,
    [[maybe_unused]] float* __restrict__ dst,
    [[maybe_unused]] float2* __restrict__ dst_meta,
    [[maybe_unused]] const float scale,
    [[maybe_unused]] const float max_bias,
    [[maybe_unused]] const float m0,
    [[maybe_unused]] const float m1,
    [[maybe_unused]] const uint32_t n_head_log2,
    [[maybe_unused]] const float logit_softcap,
    [[maybe_unused]] const int32_t ne00, [[maybe_unused]] const uint3   ne01, [[maybe_unused]] const int32_t ne02, [[maybe_unused]] const int32_t ne03,
    [[maybe_unused]] const int32_t nb01, [[maybe_unused]] const int32_t nb02, [[maybe_unused]] const int32_t nb03,
    [[maybe_unused]] const int32_t ne10, [[maybe_unused]] const int32_t ne11, [[maybe_unused]] const int32_t ne12, [[maybe_unused]] const int32_t ne13,
    [[maybe_unused]] const int32_t nb11, [[maybe_unused]] const int32_t nb12, [[maybe_unused]] const int64_t nb13,
    [[maybe_unused]] const int32_t nb21, [[maybe_unused]] const int32_t nb22, [[maybe_unused]] const int64_t nb23,
    [[maybe_unused]] const int32_t ne31, [[maybe_unused]] const int32_t ne32, [[maybe_unused]] const int32_t ne33,
    [[maybe_unused]] const int32_t nb31, [[maybe_unused]] const int32_t nb32, [[maybe_unused]] const int64_t nb33) {
    if constexpr (flash_attn_available_v && turing_mma_available_v) {

        // Skip unused kernel variants for faster compilation:
        if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
            NO_DEVICE_CODE;
            return;
        }
#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
        if (ncols1 * ncols2 > 32) {
            NO_DEVICE_CODE;
            return;
        }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

        static_assert(!mla || DKQ >= DV, "MLA needs DKQ >= DV");

        constexpr int ncols = ncols1 * ncols2;
        constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
        constexpr int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
        constexpr int nwarps = nthreads / WARP_SIZE;

        const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

        const int stride_Q1 = nb01 / sizeof(float2);
        const int stride_Q2 = nb02 / sizeof(float2);
        const int stride_K = nb11 / sizeof(half2);
        const int stride_mask = nb31 / sizeof(half);

        const int stride_V = mla ? stride_K : nb21 / sizeof(half2);

        const int iter_k = (ne11 + (nbatch_fa - 1)) / nbatch_fa;
        const int iter_j = (ne01.z + (ncols1 - 1)) / ncols1;

        // kbc == k block continuous, current index in continuous ijk space.
        int       kbc = int64_t(blockIdx.x + 0) * (iter_k * iter_j * (ne02 / ncols2) * ne03) / gridDim.x;
        const int kbc_stop = int64_t(blockIdx.x + 1) * (iter_k * iter_j * (ne02 / ncols2) * ne03) / gridDim.x;

        // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
        // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
        // In the most general case >2 seams can fall into the same tile.

        // kb0 == k start index when in the output tile.
        int kb0_start = kbc % iter_k;
        int kb0_stop = min(iter_k, kb0_start + kbc_stop - kbc);

        while (kbc < kbc_stop && kb0_stop == iter_k) {
            const int sequence = kbc / (iter_k * iter_j * (ne02 / ncols2));
            const int zt = (kbc - iter_k * iter_j * (ne02 / ncols2) * sequence) / (iter_k * iter_j); // head in units of ncols2
            const int jt = (kbc - iter_k * iter_j * (ne02 / ncols2) * sequence - iter_k * iter_j * zt) / iter_k; // j index of current tile.

            const int head0 = zt * ncols2;

            const float2* Q_f2 = (const float2*)(Q + nb03 * sequence + nb02 * head0);
            const half2* K_h2 = (const half2*)(K + nb13 * sequence + nb12 * (head0 / gqa_ratio));
            const half* mask_h = ncols2 == 1 && !mask ? nullptr :
                (const half*)(mask + nb33 * (sequence % ne33));
            float2* dstk = ((float2*)dst) + (sequence * ne01.z * ne02 + head0) * (DV / 2);

            const half2* V_h2 = mla ? K_h2 + (DKQ / 2 - DV / 2) : (const half2*)(V + nb23 * sequence + nb22 * (head0 / gqa_ratio));
            const float* sinks_f = sinks ? (const float*)sinks + head0 : nullptr;

            const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

            if (KV_max) {
                kb0_stop = min(kb0_stop, KV_max[sequence * iter_j + jt] / nbatch_fa);
            }
            constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
            if (kb0_start == 0) {
                constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
                flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup>
                    (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                        ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop);
            }
            else {
                constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
                flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup>
                    (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                        ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop);
            }

            kbc += iter_k;
            kbc -= kbc % iter_k;

            kb0_start = 0;
            kb0_stop = min(iter_k, kbc_stop - kbc);
        }

        if (kbc >= kbc_stop) {
            return;
        }

        const int sequence = kbc / (iter_k * iter_j * (ne02 / ncols2));
        const int zt = (kbc - iter_k * iter_j * (ne02 / ncols2) * sequence) / (iter_k * iter_j); // head in units of ncols2
        const int jt = (kbc - iter_k * iter_j * (ne02 / ncols2) * sequence - iter_k * iter_j * zt) / iter_k; // j index of current tile.

        const int head0 = zt * ncols2;

        const float2* Q_f2 = (const float2*)(Q + nb03 * sequence + nb02 * head0);
        const half2* K_h2 = (const half2*)(K + nb13 * sequence + nb12 * (head0 / gqa_ratio));
        const half* mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half*)(mask + nb33 * (sequence % ne33));
        float2* dstk = ((float2*)dst) + (sequence * ne01.z * ne02 + head0) * (DV / 2);

        const half2* V_h2 = mla ? K_h2 + (DKQ / 2 - DV / 2) : (const half2*)(V + nb23 * sequence + nb22 * (head0 / gqa_ratio));
        const float* sinks_f = sinks ? (const float*)sinks + head0 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence * iter_j + jt] / nbatch_fa);
        }

        constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
        constexpr bool needs_fixup = false;
        flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup>
            (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop);
    }
    else {
        NO_DEVICE_CODE;
    }
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(const flash_attn_ext_context& ctx) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && ctx.Q.ne[1] <= 8 / ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8 / ncols2, ncols2>(ctx);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && ctx.Q.ne[1] <= 16 / ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16 / ncols2, ncols2>(ctx);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || ctx.Q.ne[1] <= 32 / ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32 / ncols2, ncols2>(ctx);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64 / ncols2, ncols2>(ctx);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(const flash_attn_ext_context& ctx) {
    GGML_ASSERT(ctx.Q.ne[2] % ctx.K.ne2 == 0);
    const int gqa_ratio = ctx.Q.ne[2] / ctx.K.ne2;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (ctx.use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx);
            return;
        }

        if (ctx.use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx);
            return;
        }

        if (ctx.use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx);
        return;
    }

    if (ctx.use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx);
        return;
    }

    if (ctx.use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx);
        return;
    }

    if (ctx.use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx);
}

void ggml_cuda_flash_attn_ext_mma_f16(const flash_attn_ext_context& ctx) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    switch (ctx.Q.ne[0]) {
    case 64:
        GGML_ASSERT(ctx.V.ne0 == 64);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64, 64>(ctx);
        break;
    case 80:
        GGML_ASSERT(ctx.V.ne0 == 80);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80, 80>(ctx);
        break;
    case 96:
        GGML_ASSERT(ctx.V.ne0 == 96);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96, 96>(ctx);
        break;
    case 112:
        GGML_ASSERT(ctx.V.ne0 == 112);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx);
        break;
    case 128:
        GGML_ASSERT(ctx.V.ne0 == 128);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx);
        break;
    case 256:
        GGML_ASSERT(ctx.V.ne0 == 256);
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx);
        break;
    case 576: {
        // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
        GGML_ASSERT(ctx.V.ne0 == 512);
        const bool use_gqa_opt = ctx.mask.exist && ctx.max_bias == 0.0f;
        GGML_ASSERT(use_gqa_opt);

        GGML_ASSERT(ctx.Q.ne[2] % ctx.K.ne2 == 0);
        const int gqa_ratio = ctx.Q.ne[2] / ctx.K.ne2;
        if (gqa_ratio == 20) { // GLM 4.7 Flash
            if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                if (ctx.Q.ne[1] <= 8) {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx);
                    break;
                }
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx);
                break;
            }
            if (cc >= GGML_CUDA_CC_BLACKWELL) {
                if (ctx.Q.ne[1] <= 4 && ctx.K.ne1 >= 65536) {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx);
                    break;
                }
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx);
                break;
            }
            if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                if (ctx.Q.ne[1] <= 4) {
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx);
                    break;
                }
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx);
                break;
            }
            if (cc >= GGML_CUDA_CC_TURING) {
                if (ctx.Q.ne[1] <= 4) {
                    if (ctx.K.ne1 <= 16384) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx);
                    break;
                }
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx);
                break;
            }
            // Volta:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx);
        }
        else if (gqa_ratio % 16 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx);
        }
        else {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx);
        }
    } break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = ctx.K.type == (type_K) || (ctx.K.type == internal::GGML_TYPE_F32 && (type_K) == internal::GGML_TYPE_F16); \
        const bool type_V_okay = ctx.V.type == (type_V) || (ctx.V.type == internal::GGML_TYPE_F32 && (type_V) == internal::GGML_TYPE_F16); \
        if (ctx.Q.ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx);                                           \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                  \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

void ggml_cuda_flash_attn_ext_vec(const flash_attn_ext_context& ctx)
{
    if constexpr (ggml_cuda_fa_all_quants_v) {
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_F16)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_F16)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_1, internal::GGML_TYPE_F16)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_0, internal::GGML_TYPE_F16)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_1, internal::GGML_TYPE_F16)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_F16)

        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_Q4_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_Q4_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_1, internal::GGML_TYPE_Q4_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_0, internal::GGML_TYPE_Q4_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_1, internal::GGML_TYPE_Q4_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_Q4_0)

        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_Q4_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_Q4_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_1, internal::GGML_TYPE_Q4_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_0, internal::GGML_TYPE_Q4_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_1, internal::GGML_TYPE_Q4_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_Q4_1)

        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_Q5_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_Q5_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_1, internal::GGML_TYPE_Q5_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_0, internal::GGML_TYPE_Q5_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_1, internal::GGML_TYPE_Q5_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_Q5_0)

        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_Q5_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_Q5_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_1, internal::GGML_TYPE_Q5_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_0, internal::GGML_TYPE_Q5_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_1, internal::GGML_TYPE_Q5_1)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_Q5_1)

        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_Q8_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_Q8_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_1, internal::GGML_TYPE_Q8_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_0, internal::GGML_TYPE_Q8_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q5_1, internal::GGML_TYPE_Q8_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_Q8_0)
    }
    else {
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_F16, internal::GGML_TYPE_F16)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q4_0, internal::GGML_TYPE_Q4_0)
        FATTN_VEC_CASES_ALL_D(internal::GGML_TYPE_Q8_0, internal::GGML_TYPE_Q8_0)
    }

    GGML_ABORT("fatal error");
}