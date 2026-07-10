#include "cuda_func.h"
#include "mdspan_helper.h"
#include "operator.cuh"
#include "launch.cuh"
#include "reduce.cuh"

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

template <int rows_per_block>
static __global__ void __launch_bounds__(WARP_SIZE * rows_per_block, 2)
rwkv_wkv7_f32_t1_warp_row(rwkv_wkv7_context ctx) {
    constexpr int head_size = CUDA_WKV_BLOCK_SIZE;
    constexpr int half_head = head_size / 2;

    const int lane = threadIdx.x;
    const int row  = blockIdx.y * rows_per_block + threadIdx.y;
    const int bid  = blockIdx.x;

    const int batch_i = bid / ctx.HEADS;
    const int head_i  = bid % ctx.HEADS;
    const int state_size = ctx.C * head_size;
    const int head_off = head_i * head_size;
    const int t = batch_i * ctx.C + head_off + row;
    const float* r = ctx.r;
    const float* w = ctx.w;
    const float* k = ctx.k;
    const float* v = ctx.v;
    const float* a = ctx.a;
    const float* b = ctx.b;
    const float* s = ctx.s;
    float* dst = ctx.dst;

    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

    if (threadIdx.y == 0) {
        _r[lane] = r[batch_i * ctx.C + head_off + lane];
        _w[lane] = w[batch_i * ctx.C + head_off + lane];
        _k[lane] = k[batch_i * ctx.C + head_off + lane];
        _a[lane] = a[batch_i * ctx.C + head_off + lane];
        _b[lane] = b[batch_i * ctx.C + head_off + lane];

        _r[lane + half_head] = r[batch_i * ctx.C + head_off + lane + half_head];
        _w[lane + half_head] = w[batch_i * ctx.C + head_off + lane + half_head];
        _k[lane + half_head] = k[batch_i * ctx.C + head_off + lane + half_head];
        _a[lane + half_head] = a[batch_i * ctx.C + head_off + lane + half_head];
        _b[lane + half_head] = b[batch_i * ctx.C + head_off + lane + half_head];
    }
    __syncthreads();

    const int64_t state_base = batch_i * state_size + head_i * head_size * head_size + row * head_size;
    const float s0 = s[state_base + lane];
    const float s1 = s[state_base + lane + half_head];

    auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    const float sa = cooperative_groups::reduce(tile, _a[lane] * s0 + _a[lane + half_head] * s1, cooperative_groups::plus<float>());

    const float vt  = v[t];
    const float st0 = s0 * _w[lane]             + _k[lane]             * vt + sa * _b[lane];
    const float st1 = s1 * _w[lane + half_head] + _k[lane + half_head] * vt + sa * _b[lane + half_head];
    const float y = cooperative_groups::reduce(tile, st0 * _r[lane] + st1 * _r[lane + half_head], cooperative_groups::plus<float>());

    dst[ctx.T * ctx.C + state_base + lane]             = st0;
    dst[ctx.T * ctx.C + state_base + lane + half_head] = st1;

    if (lane == 0) {
        dst[t] = y;
    }
}

void rwkv_wkv7_cuda(const rwkv_wkv7_context& ctx, cudaStream_t stream)
{
    if (ctx.T / ctx.B == 1 && ctx.C / ctx.HEADS == CUDA_WKV_BLOCK_SIZE) {
        constexpr int rows_per_block = 4;
        rwkv_wkv7_f32_t1_warp_row<rows_per_block><<<dim3(ctx.B * ctx.HEADS, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>(ctx);
    } else if (ctx.C / ctx.HEADS == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_cuda<CUDA_WKV_BLOCK_SIZE>(ctx, stream);
    }
    else {
        rwkv_wkv7_cuda<CUDA_WKV_BLOCK_SIZE * 2>(ctx, stream);
    }
}