#pragma once
#include <assert.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#include "common.cuh"
#include "convert.cuh"
#include "internal_ds.h"

static constexpr size_t FATTN_KQ_STRIDE_TILE_F32 = 32;
static constexpr int64_t FATTN_KQ_STRIDE = 256;
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

using fattn_kernel_t = void (*)(
    const char* __restrict__ Q,
    const char* __restrict__ K,
    const char* __restrict__ V,
    const char* __restrict__ mask,
    const char* __restrict__ sinks,
    const int* __restrict__ KV_max,
    float* __restrict__ dst,
    float2* __restrict__ dst_meta,
    const float scale,
    const float max_bias,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const float logit_softcap,
    const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
    const int32_t nb01, const int32_t nb02, const int32_t nb03,
    const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
    const int32_t nb11, const int32_t nb12, const int64_t nb13,
    const int32_t nb21, const int32_t nb22, const int64_t nb23,
    const int32_t ne31, const int32_t ne32, const int32_t ne33,
    const int32_t nb31, const int32_t nb32, const int64_t nb33);

// Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
static constexpr float SOFTMAX_FTZ_THRESHOLD = -20.0f;

template<int D> // D == head size
#if !defined(GGML_USE_HIP)
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP)
static __global__ void flash_attn_combine_results(
    const float* __restrict__ VKQ_parts,
    const float2* __restrict__ VKQ_meta,
    float* __restrict__ dst,
    const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col = blockIdx.x;
    const int head = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence * ne01 + col) * ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks * D;
    VKQ_meta += j_dst_unrolled * parallel_blocks;
    dst += j_dst_unrolled * D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2 * parallel_blocks; i += D) {
        ((float*)meta)[i] = ((const float*)VKQ_meta)[i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t*)&KQ_max_scale) &= ftz_mask;

        VKQ_numerator += KQ_max_scale * VKQ_parts[l * D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE / 2, 1)
static __global__ void flash_attn_mask_to_KV_max(
    const half2* __restrict__ mask, int* __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31 = gridDim.x;
    const int tid = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt = blockIdx.x;

    mask += sequence * s33 + jt * ncols1 * s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j * s31 + KV_max_sj / 2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence * ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup(
    float* __restrict__ dst, const float2* __restrict__ dst_fixup, const int ne01, const int ne02, const int ne03, const int ne11) {
    constexpr int ncols = ncols1 * ncols2;

    const int bidx0 = blockIdx.x;
    const int j = blockIdx.y;
    const int c = blockIdx.z;
    const int jc = j * ncols2 + c;
    const int tid = threadIdx.x;

    const float* dst_fixup_data = ((const float*)dst_fixup) + gridDim.x * (2 * 2 * ncols);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    const int kbc0 = (bidx0 + 0) * (iter_k * iter_j * (ne02 / ncols2) * ne03) / gridDim.x;
    const int kbc0_stop = (bidx0 + 1) * (iter_k * iter_j * (ne02 / ncols2) * ne03) / gridDim.x;

    const bool did_not_have_any_data = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last = kbc0 / iter_k == kbc0_stop / iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    const int sequence = kbc0 / (iter_k * iter_j * (ne02 / ncols2));
    const int head = (kbc0 - iter_k * iter_j * (ne02 / ncols2) * sequence) / (iter_k * iter_j);
    const int jt = (kbc0 - iter_k * iter_j * (ne02 / ncols2) * sequence - iter_k * iter_j * head) / iter_k; // j index of current tile.

    if (jt * ncols1 + j >= ne01) {
        return;
    }

    dst += sequence * ne02 * ne01 * D + jt * ne02 * (ncols1 * D) + head * (ncols2 * D) + (j * ne02 + c) * D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val = tmp.x;
        rowsum = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while (true) {
        const int kbc = bidx * (iter_k * iter_j * (ne02 / ncols2) * ne03) / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx * ncols * D + jc * D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val * dst_val + scale_add * dst_add;
        rowsum = scale_val * rowsum + scale_add * tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc / iter_k < kbc0 / iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    const flash_attn_ext_context& ctx, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int KQ_row_granularity, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const bool is_mla = DV == 512; // TODO better parameterization

    GGML_ASSERT(ctx.V.exist || is_mla);
    GGML_ASSERT(ctx.Q.type == GGML_TYPE_F32);
    GGML_ASSERT(ctx.KQV.type == GGML_TYPE_F32);

    GGML_ASSERT(ctx.Q.nb0 == ctx.Q.element_size);
    GGML_ASSERT(ctx.K.nb0 == ctx.K.element_size);
    GGML_ASSERT(!ctx.V.exist || ctx.V.nb0 == ctx.V.element_size);

    GGML_ASSERT(!ctx.mask.exist || ctx.mask.type == GGML_TYPE_F16);
    GGML_ASSERT(!ctx.mask.exist || ctx.mask.ne1 >= GGML_PAD1(ctx.Q.ne1, 16) &&
        "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(ctx.K.ne1 % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    ggml_cuda_pool& pool = *ctx.pool;
    cudaStream_t main_stream = ctx.main_stream;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char* K_data = (const char*)ctx.K.data;
    size_t nb11 = ctx.K.nb1;
    size_t nb12 = ctx.K.nb2;
    size_t nb13 = ctx.K.nb3;

    const char* V_data = (const char*)ctx.V.data;
    size_t nb21 = ctx.V.exist ? ctx.V.nb1 : nb11;
    size_t nb22 = ctx.V.exist ? ctx.V.nb2 : nb12;
    size_t nb23 = ctx.V.exist ? ctx.V.nb3 : nb13;

    if (need_f16_K && ctx.K.type != GGML_TYPE_F16) {
        const size_t bs = ctx.K.block_size;
        const size_t ts = ctx.K.type_size;

        K_f16.alloc(ctx.K.elements);
        if (ctx.K.contiguously_allocated) {
            to_fp16_cuda(ctx.K.type, K_data, K_f16.ptr, ctx.K.elements, main_stream);

            nb11 = nb11 * bs * sizeof(half) / ts;
            nb12 = nb12 * bs * sizeof(half) / ts;
            nb13 = nb13 * bs * sizeof(half) / ts;
        }
        else {
            GGML_ASSERT(ctx.K.nb0 == ts);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            convert_to_nc_cuda(ctx.K.type, K_data, K_f16.ptr, ctx.K.ne0, ctx.K.ne1, ctx.K.ne2, ctx.K.ne3, s01, s02, s03, main_stream);

            nb11 = ctx.K.ne0 * sizeof(half);
            nb12 = ctx.K.ne1 * nb11;
            nb13 = ctx.K.ne2 * nb12;
        }
        K_data = (char*)K_f16.ptr;
    }

    if (ctx.V.exist && need_f16_V && ctx.V.type != GGML_TYPE_F16) {
        const size_t bs = ctx.V.block_size;
        const size_t ts = ctx.V.type_size;

        V_f16.alloc(ctx.V.elements);
        if (ctx.V.contiguously_allocated) {
            to_fp16_cuda(ctx.V.type, V_data, V_f16.ptr, ctx.K.elements, main_stream);
            V_data = (char*)V_f16.ptr;

            nb21 = nb21 * bs * sizeof(half) / ts;
            nb22 = nb22 * bs * sizeof(half) / ts;
            nb23 = nb23 * bs * sizeof(half) / ts;
        }
        else {
            GGML_ASSERT(ctx.V.nb0 == ts);
            const int64_t s01 = nb21 / ts;
            const int64_t s02 = nb22 / ts;
            const int64_t s03 = nb23 / ts;
            convert_to_nc_cuda(ctx.V.type, V_data, V_f16.ptr, ctx.V.ne0, ctx.V.ne1, ctx.V.ne2, ctx.V.ne3, s01, s02, s03, main_stream);

            nb21 = ctx.V.ne0 * sizeof(half);
            nb22 = ctx.V.ne1 * nb21;
            nb23 = ctx.V.ne2 * nb22;
        }
        V_data = (char*)V_f16.ptr;
    }

    const int ntiles_x = ((ctx.Q.ne1 + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (ctx.Q.ne2 / ncols2) * ctx.Q.ne3;

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (ctx.mask.exist && (ctx.Q.ne1 >= 1024 || ctx.Q.ne3 > 1)) {
        const int s31 = ctx.mask.nb1 / sizeof(half2);
        const int s33 = ctx.mask.nb3 / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, ctx.Q.ne3, 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x * blocks_num_KV_max.y;
        const int iter_k = ctx.K.ne1 / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1> << <blocks_num_KV_max, block_dim_KV_max, 0, main_stream >> >
            ((const half2*)ctx.mask.data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    int parallel_blocks = 1;

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm * nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks * tiles_nwaves);

        const int nblocks_stream_k = max_blocks;

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        dst_tmp_meta.alloc(blocks_num.x * ncols * (2 * 2 + DV) * sizeof(float));
    }
    else {
        GGML_ASSERT(ctx.K.ne1 % KQ_row_granularity == 0);
        const int ntiles_KQ = ctx.K.ne1 / KQ_row_granularity; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks should be at least large enough to achieve max. occupancy for a single wave:
        parallel_blocks = std::max((nsm * max_blocks_per_sm) / ntiles_total, 1);

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_total * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves * blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 90 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ctx.Q.ne2 * ctx.Q.ne3;

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
    fattn_kernel << <blocks_num, block_dim, nbytes_shared, main_stream >> > (
        (const char*)ctx.Q.data,
        K_data,
        V_data,
        (const char*)ctx.mask.data,
		(const char*)ctx.sinks.data,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float*)ctx.KQV.data, dst_tmp_meta.ptr,
        scale, ctx.max_bias, m0, m1, n_head_log2, ctx.logit_softcap,
        ctx.Q.ne0, ctx.Q.ne1, ctx.Q.ne2, ctx.Q.ne3, ctx.Q.nb1, ctx.Q.nb2, ctx.Q.nb3,
        ctx.K.ne0, ctx.K.ne1, ctx.K.ne2, ctx.K.ne3, nb11, nb12, nb13,
        nb21, nb22, nb23,
        ctx.mask.ne1, ctx.mask.ne2, ctx.mask.ne3,
        ctx.mask.nb1, ctx.mask.nb2, ctx.mask.nb3
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = { blocks_num.x, ncols1, ncols2 };

            flash_attn_stream_k_fixup<DV, ncols1, ncols2>
                << <blocks_num_combine, block_dim_combine, 0, main_stream >> >
                ((float*)ctx.KQV.data, dst_tmp_meta.ptr, ctx.Q.ne1, ctx.Q.ne2, ctx.Q.ne3, ctx.K.ne1);
        }
    }
    else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(ctx.Q.ne1, ctx.Q.ne2, ctx.Q.ne3);
        const size_t nbytes_shared_combine = parallel_blocks * sizeof(float2);

        flash_attn_combine_results<DV>
            << <blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream >> >
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float*)ctx.KQV.data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}