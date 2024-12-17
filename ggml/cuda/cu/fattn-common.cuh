#pragma once
#include <assert.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

static constexpr size_t FATTN_KQ_STRIDE_TILE_F32 = 32;
static constexpr int64_t FATTN_KQ_STRIDE = 256;
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

using fattn_kernel_t = void (*)(
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
    const int ne3);

// parallel_blocks == 0 is stream-k decomposition
template <int D, int ncols1, int ncols2, int parallel_blocks, int KQ_stride>
static void launch_fattn(
    const flash_attn_ext_context& ctx, fattn_kernel_t fattn_kernel,
    const int nwarps, const size_t nbytes_shared, const bool need_f16_K, const bool need_f16_V
) {
    constexpr int ncols = ncols1 * ncols2;

    GGML_ASSERT(ctx.Q.type == GGML_TYPE_F32);
    GGML_ASSERT(ctx.KQV.type == GGML_TYPE_F32);

    GGML_ASSERT(!ctx.mask.exist || ctx.mask.type == GGML_TYPE_F16);
    GGML_ASSERT(!ctx.mask.exist || ctx.mask.ne1 >= GGML_PAD1(ctx.Q.ne1, 16) &&
        "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(ctx.K.ne1 % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    GGML_ASSERT(ctx.Q.ne3 == 1);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    ggml_cuda_pool& pool = *ctx.pool;
    cudaStream_t main_stream = ctx.main_stream;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char* K_data = (const char*)ctx.K.data;
    size_t nb11 = ctx.K.nb1;
    size_t nb12 = ctx.K.nb2;
    size_t nb13 = ctx.K.nb3;

    const char* V_data = (const char*)ctx.V.data;
    size_t nb21 = ctx.V.nb1;
    size_t nb22 = ctx.V.nb2;
    size_t nb23 = ctx.V.nb3;
    if (need_f16_K && ctx.K.type != GGML_TYPE_F16) {
        K_f16.alloc(ctx.K.elements);
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(ctx.K.type);
        to_fp16(K_data, K_f16.ptr, ctx.K.elements, main_stream);
        K_data = (char*)K_f16.ptr;

        nb11 = nb11 * ctx.K.bs * sizeof(half) / ctx.K.ts;
        nb12 = nb12 * ctx.K.bs * sizeof(half) / ctx.K.ts;
        nb13 = nb13 * ctx.K.bs * sizeof(half) / ctx.K.ts;
    }

    if (need_f16_V && ctx.V.type != GGML_TYPE_F16) {
        V_f16.alloc(ctx.V.elements);
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(ctx.V.type);
        to_fp16(V_data, V_f16.ptr, ctx.V.elements, main_stream);
        V_data = (char*)V_f16.ptr;

        nb21 = nb21 * ctx.V.bs * sizeof(half) / ctx.V.ts;
        nb22 = nb22 * ctx.V.bs * sizeof(half) / ctx.V.ts;
        nb23 = nb23 * ctx.V.bs * sizeof(half) / ctx.V.ts;
    }

    const int ntiles_x = ((ctx.Q.ne1 + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (ctx.Q.ne2 / ncols2) * ctx.Q.ne3;
    const dim3 block_dim(warp_size, nwarps, 1);
    dim3 blocks_num;
    if (parallel_blocks == 0) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = 2 * nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks * tiles_nwaves);

        const int nblocks_stream_k = max_blocks;
        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;
        dst_tmp_meta.alloc(blocks_num.x * ncols * (2 * 2 + D) * sizeof(float));
    }
    else {
        blocks_num.x = parallel_blocks * ntiles_x;
        blocks_num.y = ctx.Q.ne2;
        blocks_num.z = ctx.Q.ne3;

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks * ctx.KQV.elements);
            dst_tmp_meta.alloc(parallel_blocks * ctx.KQV.nrows);
        }
    }

    float scale = ctx.scale;
    if (ctx.logit_softcap != 0.0f) {
        scale /= ctx.logit_softcap;
    }

    const uint32_t n_head = ctx.Q.ne2;
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(ctx.max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(ctx.max_bias / 2.0f) / n_head_log2);
    GGML_ASSERT(block_dim.x % warp_size == 0);
    GGML_ASSERT(!GGML_CUDA_CC_IS_AMD(cc) || block_dim.x * block_dim.y <= 4 * (unsigned int)warp_size);

    fattn_kernel << <blocks_num, block_dim, nbytes_shared, main_stream >> > (
        (const char*)ctx.Q.data,
        K_data,
        V_data,
        (const char*)ctx.mask.data,
        (parallel_blocks) > 1 ? dst_tmp.ptr : (float*)ctx.KQV.data, dst_tmp_meta.ptr,
        scale, ctx.max_bias, m0, m1, n_head_log2, ctx.logit_softcap,
        ctx.Q.ne0, ctx.Q.ne1, ctx.Q.ne2, ctx.Q.ne3,
        ctx.K.ne0, ctx.K.ne1, ctx.K.ne2, ctx.K.ne3,
        ctx.mask.ne1, ctx.mask.nb1,
        ctx.Q.nb1, ctx.Q.nb2, ctx.Q.nb3,
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        ctx.KQV.ne0, ctx.KQV.ne1, ctx.KQV.ne2, ctx.KQV.ne3
        );
    CUDA_CHECK(cudaGetLastError());
    if constexpr (parallel_blocks == 0) {
        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(D, 1, 1);
            const dim3 blocks_num_combine = { blocks_num.x, ncols1, ncols2 };

            flash_attn_stream_k_fixup<D, ncols1, ncols2, KQ_stride>
                << <blocks_num_combine, block_dim_combine, 0, main_stream >> >
                ((float*)ctx.KQV.data, dst_tmp_meta.ptr, ctx.Q.ne1, ctx.Q.ne2, ctx.K.ne1);

        }
    }
    else if constexpr (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(ctx.Q.ne1, blocks_num.y, blocks_num.z);

        flash_attn_combine_results<D, parallel_blocks>
            << <blocks_num_combine, block_dim_combine, 0, main_stream >> >
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float*)ctx.KQV.data);
    }
    CUDA_CHECK(cudaGetLastError());
}