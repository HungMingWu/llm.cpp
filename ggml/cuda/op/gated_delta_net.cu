#include "cuda_func.h"
#include "common.cuh"
#include "reduce.cuh"
#include "mdspan_helper.h"

#define GGML_ABORT(...)

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(gated_delta_net_context ctx,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * ctx.H * ctx.n_tokens * ctx.n_seqs;

    auto attn_data = [=]() {
        std::mdspan attn_data(ctx.dst_d, ctx.n_seqs, ctx.n_tokens, ctx.H, S_v);
        return std::submdspan(attn_data, sequence, std::full_extent, h_idx, std::full_extent);
    }();

    auto state = [=]() {
        std::mdspan state(ctx.dst_d + attn_score_elems, ctx.n_seqs, ctx.H, S_v, S_v);
        return std::submdspan(state, sequence, h_idx, std::full_extent, std::full_extent);
    }();

    // input state layout (D, K, n_seqs) ¡X seq stride is K * D = K * H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) ¡X same per-(seq,head) offset as before.
	const auto curr_state = [=]() {
        std::mdspan curr_state(ctx.s_d, ctx.n_seqs, ctx.K, ctx.H, S_v, S_v);
        return std::submdspan(curr_state, sequence, 0, h_idx, std::full_extent, std::full_extent);
    }();

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<warp_size>(block);

    ggml_cuda_pdl_sync();
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state(col, i);
    }

    // slot mapping: target_slot = t - shift. When n_tokens < K only the last n_tokens slots
    // are written; earlier slots are left untouched (caller-owned).
    const int shift = (int)ctx.n_tokens - ctx.K;

    const auto q_t = [=]() {
        auto q_data = make_strided_mdspan(ctx.q_d, ctx.qk_ne, ctx.qk_nb);
        return std::submdspan(q_data, iq3, std::full_extent, iq1, std::full_extent);
    }();

    const auto k_t = [=]() {
        auto k_data = make_strided_mdspan(ctx.k_d, ctx.qk_ne, ctx.qk_nb);
        return std::submdspan(k_data, iq3, std::full_extent, iq1, std::full_extent);
    }();

    const auto v_t = [=]() {
        auto v_data = make_strided_mdspan(ctx.v_d, ctx.v_ne, ctx.v_nb);
        return std::submdspan(v_data, sequence, std::full_extent, h_idx, std::full_extent);
    }();

    const auto g_t = [=]() {
        auto g_data = make_strided_mdspan(ctx.g_d, ctx.g_ne, ctx.g_nb);
        return std::submdspan(g_data, sequence, std::full_extent, h_idx, std::full_extent);
    }();

    const auto beta_t = [=]() {
        auto beta_data = make_strided_mdspan(ctx.b_d, ctx.b_ne, ctx.b_nb);
        return std::submdspan(beta_data, sequence, std::full_extent, h_idx, 0);
    }();

    for (int t = 0; t < ctx.n_tokens; t++) {
        const float beta_val = beta_t(t);

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t(t, i);
            q_reg[r] = q_t(t, i);
        }

        if constexpr (!KDA) {
            const float g_val = expf(g_t(t, 0));

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = cooperative_groups::reduce(tile, kv_shard, cooperative_groups::plus<float>());

            // delta[col] = (v[t, col] - g * kv[col]) * beta
            float delta_col = (v_t(t, col) - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = cooperative_groups::reduce(tile, attn_partial, cooperative_groups::plus<float>());

            if (lane == 0) {
                attn_data(t, col) = attn_col * ctx.scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t(t, i)) * s_shard[r] * k_reg[r];
            }

            float kv_col = cooperative_groups::reduce(tile, kv_shard, cooperative_groups::plus<float>());

            // delta[col] = (v[t, col] - kv[col]) * beta
            float delta_col = (v_t(t, col) - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t(t, i)) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = cooperative_groups::reduce(tile, attn_partial, cooperative_groups::plus<float>());

            if (lane == 0) {
                attn_data(t, col) = attn_col * ctx.scale;
            }
        }

        if constexpr (keep_rs_t) {
            const int target_slot = t - shift;
            if (target_slot >= 0 && target_slot < ctx.K) {
                auto curr_state_o = [&]() {
                    std::mdspan curr_state_o(ctx.dst_d + attn_score_elems, ctx.K, ctx.n_seqs, ctx.H, S_v, S_v);
                    return std::submdspan(curr_state_o, target_slot, sequence, h_idx, std::full_extent, std::full_extent);
                }();
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state_o(col, i) = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state(col, i) = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(const gated_delta_net_context& ctx, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
	const int64_t S_v = ctx.S_v;
    dim3      grid_dims(ctx.H, ctx.n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(ctx.neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(ctx.rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                ctx, neqk1_magic, rq3_magic);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                ctx, neqk1_magic, rq3_magic);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                ctx, neqk1_magic, rq3_magic);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                ctx, neqk1_magic, rq3_magic);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void gated_delta_net_cuda(const gated_delta_net_context& ctx, cudaStream_t  stream)
{
    const bool keep_rs = ctx.K > 1;
    if (ctx.kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(ctx, stream);
        } else {
            launch_gated_delta_net<true, true>(ctx, stream);
        }
    }
    else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(ctx, stream);
        } else {
            launch_gated_delta_net<false, false>(ctx, stream);
        }
    }
}