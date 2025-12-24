#include "cuda_func.h"
#include "mdspan_helper.h"
#include "operator.cuh"
#include "launch.cuh"

#define GGML_ASSERT(...)
static constexpr size_t CUDA_WKV_BLOCK_SIZE = 64;

template <int head_size>
static __global__ void rwkv_wkv6_f32(
    const int n_seqs, const int T, const int C, const int HEADS, auto k,
    auto v, auto r, auto tf, auto td, auto s, auto dst_data, auto dst_state) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int batch_i = bid / HEADS;
    const int head_i = bid % HEADS;
    const int n_seq_tokens = T / n_seqs;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _tf[head_size], _td[head_size];

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s(batch_i, head_i, i, tid);
    }

    _tf[tid] = tf(head_i, tid);
    __syncthreads();

    for (int t = batch_i * n_seq_tokens; t < (batch_i + 1) * n_seq_tokens; t++) {
        _k[tid] = k(t, head_i, tid);
        _r[tid] = r(t, head_i, tid);
        _td[tid] = td(t, head_i, tid);
        __syncthreads();

        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& tf = (float4&)(_tf[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            const float4 kv = k * v(t, head_i, tid);

            y += dot_product(r, tf * kv + s);
            s = s * td + kv;
            __syncthreads();
        }
        dst_data(t, head_i, tid) = y;
    }

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst_state(batch_i, head_i, i, tid) = state[i];
    }
}

template <int head_size>
static __global__ void rwkv_wkv7_f32(const int n_seqs,
    const int T, const int C, const int HEADS,
    auto r, auto w, auto k,
    auto v, auto a, auto b, auto s, auto dst_data, auto dst_state)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int batch_i = bid / HEADS;
    const int head_i = bid % HEADS;
    const int n_seq_tokens = T / n_seqs;

    float state[head_size];
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s(batch_i, head_i, tid, i);;
    }

    for (int t = batch_i * n_seq_tokens; t < (batch_i + 1) * n_seq_tokens; t++) {
        _r[tid] = r(t, head_i, tid);
        _w[tid] = w(t, head_i, tid);
        _k[tid] = k(t, head_i, tid);
        _a[tid] = a(t, head_i, tid);;
        _b[tid] = b(t, head_i, tid);;
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& a = (float4&)(_a[j]);
            const float4& s = (float4&)(state[j]);
            sa += dot_product(a, s);
        }

        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& r = (float4&)(_r[j]);
            const float4& w = (float4&)(_w[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& b = (float4&)(_b[j]);
            float4& s = (float4&)(state[j]);
            s = s * w + k * v(t, head_i, tid) + b * sa;
            y += dot_product(s, r);
        }
        dst_data(t, head_i, tid) = y;
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst_state(batch_i, head_i, tid, i) = state[i];
    }
}

template <size_t head_size>
void rwkv_wkv6_cuda(const rwkv_wkv6_context& ctx, cudaStream_t stream)
{
    std::mdspan s(ctx.s, ctx.n_seqs, ctx.HEADS, head_size, head_size);
    std::mdspan k(ctx.k, ctx.T, ctx.HEADS, head_size);
    std::mdspan r(ctx.r, ctx.T, ctx.HEADS, head_size);
    std::mdspan v(ctx.v, ctx.T, ctx.HEADS, head_size);
    std::mdspan td(ctx.td, ctx.T, ctx.HEADS, head_size);
    std::mdspan tf(ctx.tf, ctx.HEADS, head_size);
    std::mdspan dst_data(ctx.dst, ctx.T, ctx.HEADS, head_size);
    std::mdspan dst_state(ctx.dst + ctx.T * ctx.C, ctx.n_seqs, ctx.HEADS, head_size, head_size);
    rwkv_wkv6_f32<head_size> << <ctx.n_seqs * ctx.HEADS, head_size, 0, stream >> >
        (ctx.n_seqs, ctx.T, ctx.C, ctx.HEADS, k, v, r, tf, td, s, dst_data, dst_state);
}

void rwkv_wkv6_cuda(const rwkv_wkv6_context&ctx, cudaStream_t stream)
{
    const size_t head_size = ctx.C / ctx.HEADS;

    if (head_size == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv6_cuda<CUDA_WKV_BLOCK_SIZE>(ctx, stream);
    }
    else {
        rwkv_wkv6_cuda<CUDA_WKV_BLOCK_SIZE * 2>(ctx, stream);
    }
}

template <size_t head_size>
void rwkv_wkv7_cuda(const rwkv_wkv7_context& ctx, cudaStream_t stream)
{
    std::mdspan s(ctx.s, ctx.n_seqs, ctx.HEADS, head_size, head_size);
    std::mdspan r(ctx.r, ctx.T, ctx.HEADS, head_size);
    std::mdspan w(ctx.w, ctx.T, ctx.HEADS, head_size);
    std::mdspan k(ctx.k, ctx.T, ctx.HEADS, head_size);
    std::mdspan a(ctx.a, ctx.T, ctx.HEADS, head_size);
    std::mdspan b(ctx.b, ctx.T, ctx.HEADS, head_size);
    std::mdspan v(ctx.v, ctx.T, ctx.HEADS, head_size);
    std::mdspan dst_data(ctx.dst, ctx.T, ctx.HEADS, head_size);
    std::mdspan dst_state(ctx.dst + ctx.T * ctx.C, ctx.n_seqs, ctx.HEADS, head_size, head_size);
    rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE> << <ctx.n_seqs * ctx.HEADS, head_size, 0, stream >> >
        (ctx.n_seqs, ctx.T, ctx.C, ctx.HEADS, r, w, k, v, a, b, s, dst_data, dst_state);
}

void rwkv_wkv7_cuda(const rwkv_wkv7_context& ctx, cudaStream_t stream)
{
	const size_t head_size = ctx.C / ctx.HEADS;
    GGML_ASSERT(head_size == CUDA_WKV_BLOCK_SIZE || head_size == CUDA_WKV_BLOCK_SIZE * 2);

    if (head_size == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_cuda< CUDA_WKV_BLOCK_SIZE>(ctx, stream);
    }
    else {
        rwkv_wkv7_cuda< CUDA_WKV_BLOCK_SIZE * 2>(ctx, stream);
    }
}