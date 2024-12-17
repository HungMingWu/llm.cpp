#include "cuda_func.h"

template<int HEAD_SIZE>
static __global__ void gated_linear_attn_f32(const int B, const int T, const int C, const int H, const float scale,
    const float* k, const float* v, const float* r, const float* td, const float* s, float* dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = HEAD_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _td[head_size];

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;

            y += r.x * s.x;
            y += r.y * s.y;
            y += r.z * s.z;
            y += r.w * s.w;
        }
        dst[t] = y * scale;
    }

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

void gated_linear_attn_cuda(const gla_context* ctx, cudaStream_t stream)
{
    if (ctx->C / ctx->H == 64) {
        gated_linear_attn_f32<64> << <ctx->B * ctx->H, ctx->C / ctx->H, 0, stream >> > 
            (ctx->B, ctx->T, ctx->C, ctx->H, ctx->scale, ctx->k, ctx->v, ctx->r, ctx->td, ctx->s, ctx->dst);
    }
    else {
        gated_linear_attn_f32<128> << <ctx->B * ctx->H, ctx->C / ctx->H, 0, stream >> > 
            (ctx->B, ctx->T, ctx->C, ctx->H, ctx->scale, ctx->k, ctx->v, ctx->r, ctx->td, ctx->s, ctx->dst);
    }
}