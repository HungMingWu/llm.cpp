#include <assert.h>
#include "cuda_func.h"
#include "internal_ds.h"
#include "mmq.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mmq_ids_helper_store {
    uint32_t data;

    __device__ mmq_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mmq_ids_helper_store) == 4, "unexpected size for mmq_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mmq_ids_helper(
    const int32_t* __restrict__ ids, int32_t* __restrict__ ids_src1, int32_t* __restrict__ ids_dst, int32_t* __restrict__ expert_bounds,
    const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mmq_ids_helper[];
    mmq_ids_helper_store* store = (mmq_ids_helper_store*)data_mmq_ids_helper;

    int nex_prev = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it * si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mmq_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    }
    else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size / neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it * si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mmq_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mmq_ids_helper_store store_it = store[itc];
        const int it = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it * sis1 + iex_used % nchannels_y;
        ids_dst[nex_prev + itc] = it * n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mmq_ids_helper(
    const int32_t* __restrict__ ids, int32_t* __restrict__ ids_src1, int32_t* __restrict__ ids_dst, int32_t* __restrict__ expert_bounds,
    const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens < (1 << 22) && "too few bits in mmq_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mmq_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(reinterpret_cast<const void*>(mmq_ids_helper<n_expert_used_template>), smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens * sizeof(mmq_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mmq_ids_helper<n_expert_used_template> << <num_blocks, block_size, nbytes_shared, stream >> >
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

void launch_mmq_ids_helper(const mmq_ids_helper_context* ctx, cudaStream_t stream)
{
    switch (ctx->n_expert_used) {
    case  2:
        launch_mmq_ids_helper< 2>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    case  4:
        launch_mmq_ids_helper< 4>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    case  6:
        launch_mmq_ids_helper< 6>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    case  8:
        launch_mmq_ids_helper< 8>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    case 16:
        launch_mmq_ids_helper<16>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    case 32:
        launch_mmq_ids_helper<32>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    default:
        launch_mmq_ids_helper< 0>(ctx->ids, ctx->ids_src1, ctx->ids_dst, ctx->expert_bounds,
            ctx->n_experts, ctx->n_tokens, ctx->n_expert_used, ctx->nchannels_y, ctx->si1, ctx->sis1, stream);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_q_switch_type(ggml_cuda_pool& pool, const mmq_args& args, cudaStream_t stream) {
    switch (args.type_x) {
    case GGML_TYPE_Q4_0:
        mul_mat_q_case<GGML_TYPE_Q4_0, block_q4_0>(pool, args, stream);
        break;
    case GGML_TYPE_Q4_1:
        mul_mat_q_case<GGML_TYPE_Q4_1, block_q4_1>(pool, args, stream);
        break;
    case GGML_TYPE_Q5_0:
        mul_mat_q_case<GGML_TYPE_Q5_0, block_q5_0>(pool, args, stream);
        break;
    case GGML_TYPE_Q5_1:
        mul_mat_q_case<GGML_TYPE_Q5_1, block_q5_1>(pool, args, stream);
        break;
    case GGML_TYPE_Q8_0:
        mul_mat_q_case<GGML_TYPE_Q8_0, block_q8_0>(pool, args, stream);
        break;
    case GGML_TYPE_MXFP4:
        mul_mat_q_case<GGML_TYPE_MXFP4, block_mxfp4>(pool, args, stream);
        break;
    case GGML_TYPE_Q2_K:
        mul_mat_q_case<GGML_TYPE_Q2_K, block_q2_K>(pool, args, stream);
        break;
    case GGML_TYPE_Q3_K:
        mul_mat_q_case<GGML_TYPE_Q3_K, block_q3_K>(pool, args, stream);
        break;
    case GGML_TYPE_Q4_K:
        mul_mat_q_case<GGML_TYPE_Q4_K, block_q4_K>(pool, args, stream);
        break;
    case GGML_TYPE_Q5_K:
        mul_mat_q_case<GGML_TYPE_Q5_K, block_q5_K>(pool, args, stream);
        break;
    case GGML_TYPE_Q6_K:
        mul_mat_q_case<GGML_TYPE_Q6_K, block_q6_K>(pool, args, stream);
        break;
    case GGML_TYPE_IQ2_XXS:
        mul_mat_q_case<GGML_TYPE_IQ2_XXS, block_iq2_xxs>(pool, args, stream);
        break;
    case GGML_TYPE_IQ2_XS:
        mul_mat_q_case<GGML_TYPE_IQ2_XS, block_iq2_xs>(pool, args, stream);
        break;
    case GGML_TYPE_IQ2_S:
        mul_mat_q_case<GGML_TYPE_IQ2_S, block_iq2_s>(pool, args, stream);
        break;
    case GGML_TYPE_IQ3_XXS:
        mul_mat_q_case<GGML_TYPE_IQ3_XXS, block_iq3_xxs>(pool, args, stream);
        break;
    case GGML_TYPE_IQ3_S:
        mul_mat_q_case<GGML_TYPE_IQ3_S, block_iq3_s>(pool, args, stream);
        break;
    case GGML_TYPE_IQ1_S:
        mul_mat_q_case<GGML_TYPE_IQ1_S, block_iq1_s>(pool, args, stream);
        break;
    case GGML_TYPE_IQ4_XS:
        mul_mat_q_case<GGML_TYPE_IQ4_XS, block_iq4_xs>(pool, args, stream);
        break;
    case GGML_TYPE_IQ4_NL:
        mul_mat_q_case<GGML_TYPE_IQ4_NL, block_iq4_nl>(pool, args, stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}
