#include <assert.h>
#include "cuda_func.h"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap>
__launch_bounds__(nwarps* WARP_SIZE, 2)
static __global__ void flash_attn_ext_f16(
    const char* __restrict__ Q,
    const char* __restrict__ K,
    const char* __restrict__ V,
    const char* __restrict__ mask,
    float* __restrict__ dst,
    float2* __restrict__ dst_meta,
    const float scale,
    const float max_bias,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const float logit_softcap,
    const int ne00,
    const int ne01,
    const int ne02,
    const int ne03,
    const int ne10,
    const int ne11,
    const int ne12,
    const int ne13,
    const int ne31,
    const int nb31,
    const int nb01,
    const int nb02,
    const int nb03,
    const int nb11,
    const int nb12,
    const int nb13,
    const int nb21,
    const int nb22,
    const int nb23,
    const int ne0,
    const int ne1,
    const int ne2,
    const int ne3) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(NEW_MMA_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(FATTN_KQ_STRIDE % KQ_per_iter == 0, "bad KQ_per_iter");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1 = nb01 / sizeof(float2);
    const int stride_Q2 = nb02 / sizeof(float2);
    const int stride_KV = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half2);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    constexpr int kb_niter = FATTN_KQ_STRIDE / KQ_per_iter; // Number of kernel iterations per assigned KQ slice.

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc = (blockIdx.x + 0) * iter_k * iter_j * (ne02 / ncols2) / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1) * iter_k * iter_j * (ne02 / ncols2) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop = min(iter_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int channel = kbc / (iter_k * iter_j);
        const int jt = (kbc - channel * iter_k * iter_j) / iter_k; // j index of current tile.

        const float2* Q_f2 = (const float2*)(Q + nb02 * channel * ncols2);
        const half2* K_h2 = (const half2*)(K + nb12 * (channel * ncols2 / gqa_ratio));
        const half2* V_h2 = (const half2*)(V + nb12 * (channel * ncols2 / gqa_ratio)); // K and V have same shape
        const half2* mask_h2 = ncols2 > 1 || mask ? (const half2*)mask + (nb31 / sizeof(half2)) * jt * ncols1 : nullptr;
        float2* dstk = ((float2*)dst) + channel * (ncols2 * D / 2);

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, channel, n_head_log2, m0, m1) : 1.0f;

        const int kb0_start_kernel = kb0_start * kb_niter;
        const int kb0_stop_kernel = kb0_stop * kb_niter;

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale, slope, logit_softcap,
                    ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        }
        else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale, slope, logit_softcap,
                    ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int channel = kbc / (iter_k * iter_j);
    const int jt = (kbc - channel * iter_k * iter_j) / iter_k; // j index of current tile.

    const float2* Q_f2 = (const float2*)(Q + nb02 * channel * ncols2);
    const half2* K_h2 = (const half2*)(K + nb12 * (channel * ncols2 / gqa_ratio));
    const half2* V_h2 = (const half2*)(V + nb12 * (channel * ncols2 / gqa_ratio)); // K and V have same shape
    const half2* mask_h2 = ncols2 > 1 || mask ? (const half2*)mask + (nb31 / sizeof(half2)) * jt * ncols1 : nullptr;
    float2* dstk = ((float2*)dst) + channel * (ncols2 * D / 2);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, channel, n_head_log2, m0, m1) : 1.0f;

    const int kb0_start_kernel = kb0_start * kb_niter;
    const int kb0_stop_kernel = kb0_stop * kb_niter;

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale, slope, logit_softcap,
            ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap); GGML_UNUSED(ne00);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03); GGML_UNUSED(ne10);
    GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
    GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13); GGML_UNUSED(nb21);
    GGML_UNUSED(nb22); GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
    GGML_UNUSED(ne2); GGML_UNUSED(ne3);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(NEW_MMA_AVAILABLE)
}

template <int D, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(const flash_attn_ext_context& ctx) {
    constexpr int ncols = ncols1 * ncols2;
    constexpr int KQ_per_iter = D <= 128 && ncols1 <= 64 ? 64 : 32;
    constexpr int nwarps = (KQ_per_iter == 32 && ncols <= 16) ? 2 : 4;
    constexpr int ntiles = ncols <= 8 ? 1 : (ncols <= 64 ? 2 : 4);
    constexpr int cols_per_warp = ntiles * tile_B::I;

    static_assert(D % tile_B::J == 0, "bad D");
    static_assert(ncols % cols_per_warp == 0, "bad ncols");

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    const int KQ_shared_rows = cp_async_available(cc) ? 2 * KQ_per_iter : KQ_per_iter;

    const size_t nbytes_shared_KV = KQ_shared_rows * (D + 8) * sizeof(half);
    const size_t nbytes_shared_mask = ncols1 * (KQ_per_iter + 8) * sizeof(half);
    const size_t nbytes_shared_combine = nwarps * cols_per_warp * (D + 8) * sizeof(half);

    const size_t nbytes_shared_total = std::max(nbytes_shared_KV + nbytes_shared_mask, nbytes_shared_combine);

    fattn_kernel_t fattn_kernel;
    if (ctx.logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap>;
    }
    else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap>;
    }

    launch_fattn<D, ncols1, ncols2, KQ_per_iter>
        (ctx, fattn_kernel, nwarps, nbytes_shared_total, FATTN_KQ_STRIDE, true, true, true);
}

template <int D, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(const flash_attn_ext_context& ctx) {

    if (ctx.Q.ne1 <= 8 / ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<D, 8 / ncols2, ncols2>(ctx);
        return;
    }

    if (ctx.Q.ne1 <= 16 / ncols2) {
        //ggml_cuda_flash_attn_ext_mma_f16_case<D, 16 / ncols2, ncols2>(ctx);
        return;
    }

    if (ctx.Q.ne1 <= 32 / ncols2) {
        //ggml_cuda_flash_attn_ext_mma_f16_case<D, 32 / ncols2, ncols2>(ctx);
        return;
    }

    //ggml_cuda_flash_attn_ext_mma_f16_case<D, 64 / ncols2, ncols2>(ctx);
}

template <int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_hs(const flash_attn_ext_context& ctx) {
    switch (ctx.Q.ne0) {
    case 64:
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<64, ncols2>(ctx);
        break;
    case 80:
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<80, ncols2>(ctx);
        break;
    case 96:
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<96, ncols2>(ctx);
        break;
    case 112:
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<112, ncols2>(ctx);
        break;
    case 128:
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<128, ncols2>(ctx);
        break;
    case 256:
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<256, ncols2>(ctx);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

void ggml_cuda_flash_attn_ext_mma_f16(const flash_attn_ext_context& ctx) {
    const float use_gqa_opt = ctx.mask.exist && ctx.max_bias == 0.0f;

    assert(ctx.Q.ne2 % ctx.K.ne2 == 0);
    const int gqa_ratio = ctx.Q.ne2 / ctx.K.ne2;

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<8>(ctx);
        return;
    }

    if (use_gqa_opt && gqa_ratio == 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<4>(ctx);
        return;
    }

    if (use_gqa_opt && gqa_ratio == 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<2>(ctx);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_hs<1>(ctx);
}