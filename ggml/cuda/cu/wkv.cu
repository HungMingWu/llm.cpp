#define GGML_ASSERT(...)
static constexpr size_t CUDA_WKV_BLOCK_SIZE = 64;

template <int block_size>
static __global__ void rwkv_wkv_f32(const int B, const int T, const int C, const int H, const float* k, const float* v, const float* r, const float* tf, const float* td, const float* s, float* dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _tf[head_size], _td[head_size];

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    __syncthreads();
    _tf[tid] = tf[head_i * head_size + tid];
    __syncthreads();

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
            const float4& tf = (float4&)(_tf[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            y += r.x * (tf.x * kv.x + s.x);
            y += r.y * (tf.y * kv.y + s.y);
            y += r.z * (tf.z * kv.z + s.z);
            y += r.w * (tf.w * kv.w + s.w);

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;
        }
        dst[t] = y;
    }

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

void rwkv_wkv6_cuda(const int B, const int T, const int C,
	const int H, const float* k,
	const float* v, const float* r,
	const float* tf, const float* td, const float* s, float* dst, cudaStream_t stream)
{
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);
    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE> << <B * H, C / H, 0, stream >> > (B, T, C, H, k, v, r, tf, td, s, dst);
    }
    else {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE * 2> << <B * H, C / H, 0, stream >> > (B, T, C, H, k, v, r, tf, td, s, dst);
    }
}

template <int block_size>
static __global__ void rwkv_wkv7_f32(const int B,
    const int T, const int C, const int H,
    const float* r, const float* w, const float* k,
    const float* v, const float* a, const float* b, const float* s, float* dst)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

#ifndef GGML_USE_MUSA
#pragma unroll
#endif
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& a = (float4&)(_a[j]);
            const float4& s = (float4&)(state[j]);
            sa += a.x * s.x;
            sa += a.y * s.y;
            sa += a.z * s.z;
            sa += a.w * s.w;
        }

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& r = (float4&)(_r[j]);
            const float4& w = (float4&)(_w[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& b = (float4&)(_b[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * w.x + kv.x + sa * b.x;
            s.y = s.y * w.y + kv.y + sa * b.y;
            s.z = s.z * w.z + kv.z + sa * b.z;
            s.w = s.w * w.w + kv.w + sa * b.w;

            y += s.x * r.x;
            y += s.y * r.y;
            y += s.z * r.z;
            y += s.w * r.w;
        }
        dst[t] = y;
    }

#pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

void rwkv_wkv_cuda(const int B,
    const int T, const int C, const int H,
    const float* k, const float* v, const float* r,
    const float* tf, const float* td, const float* s, float* dst, cudaStream_t stream)
{
    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE> << <B * H, C / H, 0, stream >> > (B, T, C, H, k, v, r, tf, td, s, dst);
    }
    else {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE * 2> << <B * H, C / H, 0, stream >> > (B, T, C, H, k, v, r, tf, td, s, dst);
    }
}

void rwkv_wkv7_cuda(const int B,
    const int T, const int C, const int H,
    const float* r, const float* w, const float* k,
    const float* v, const float* a, const float* b, const float* s, float* dst, cudaStream_t stream)
{
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE> << <B * H, C / H, 0, stream >> > (B, T, C, H, r, w, k, v, a, b, s, dst);
    }
    else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2> << <B * H, C / H, 0, stream >> > (B, T, C, H, r, w, k, v, a, b, s, dst);
    }
}