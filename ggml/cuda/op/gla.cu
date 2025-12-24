#include "cuda_func.h"
#include "mdspan_helper.h"
#include "operator.cuh"

template <int head_size>
static __global__ void gated_linear_attn_f32(const int n_seqs, const int T, const int C, const int HEADS, const float scale,
    auto k, auto v, auto r, auto td, auto s, auto dst_data, auto dst_state) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int batch_i = bid / HEADS;
    const int head_i = bid % HEADS;
    const int state_size = C * head_size;
    const int64_t n_seq_tokens = T / n_seqs;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _td[head_size];

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s(batch_i, head_i, i, tid);
    }

    for (int t = batch_i * n_seq_tokens; t < (batch_i + 1) * n_seq_tokens; t++) {
        _k[tid] = k(t, head_i, tid);
        _r[tid] = r(t, head_i, tid);
        _td[tid] = td(t, head_i, tid);
        __syncthreads();

        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);

            s = s * td + k * v(t, head_i, tid);
            y += dot_product(r, s);
        }

        dst_data(t, head_i, tid) = y * scale;
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst_state(batch_i, head_i, i, tid) = state[i];
    }
}

template <size_t head_size>
void gated_linear_attn_cuda(const gla_context& ctx, cudaStream_t stream)
{
    std::mdspan s(ctx.s, ctx.n_seqs, ctx.HEADS, head_size, head_size);
    std::mdspan k(ctx.k, ctx.T, ctx.HEADS, head_size);
    std::mdspan r(ctx.r, ctx.T, ctx.HEADS, head_size);
    std::mdspan v(ctx.v, ctx.T, ctx.HEADS, head_size);
    std::mdspan td(ctx.td, ctx.T, ctx.HEADS, head_size);
    std::mdspan dst_data(ctx.dst, ctx.T, ctx.HEADS, head_size);
    std::mdspan dst_state(ctx.dst + ctx.T * ctx.C, ctx.n_seqs, ctx.HEADS, head_size, head_size);
    gated_linear_attn_f32<64> << <ctx.n_seqs * ctx.HEADS, head_size, 0, stream >> >
        (ctx.n_seqs, ctx.T, ctx.C, ctx.HEADS, ctx.scale, k, v, r, td, s, dst_data, dst_state);
}

void gated_linear_attn_cuda(const gla_context &ctx, cudaStream_t stream)
{
    const size_t head_size = ctx.C / ctx.HEADS;

    if (ctx.C / ctx.HEADS == 64) {
		gated_linear_attn_cuda<64>(ctx, stream);
    }
    else {
        gated_linear_attn_cuda<128>(ctx, stream);
    }
}