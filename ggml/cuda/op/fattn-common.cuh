#pragma once
#include <assert.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#include "common.cuh"
#include "convert.cuh"
#include "cuda_func.h"
#include "vecdotq.cuh"

static constexpr size_t FATTN_KQ_STRIDE_TILE_F32 = 32;
static constexpr int64_t FATTN_KQ_STRIDE = 256;
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

using fattn_kernel_t = void (*)(
    flash_attn_ext_context ctx,
    const char* __restrict__ K,
    const char* __restrict__ V,
    const int* __restrict__ KV_max,
    float* __restrict__ dst,
    float2* __restrict__ dst_meta,
    const float scale,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
    const int32_t nb01, const int32_t nb02, const int32_t nb03,
    const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
    const int32_t nb11, const int32_t nb12, const int64_t nb13,
    const int32_t nb21, const int32_t nb22, const int64_t nb23,
    const int32_t ne31, const int32_t ne32, const int32_t ne33,
    const int32_t nb31, const int32_t nb32, const int64_t nb33);

// Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
static constexpr float SOFTMAX_FTZ_THRESHOLD = -20.0f;

// Remove later
static __device__ __forceinline__ float __low2float1(uint32_t value)
{
    const auto [x, _] = std::bit_cast<std::array<half, 2>>(value);
    return __half2float(x);
}

static __device__ __forceinline__ float __high2float1(uint32_t value)
{
    const auto [_, y] = std::bit_cast<std::array<half, 2>>(value);
    return __half2float(y);
}

static __device__ __forceinline__ half __low2half1(uint32_t value)
{
    const auto [x, _] = std::bit_cast<std::array<half, 2>>(value);
    return x;
}

static __device__ __forceinline__ half2 __tohalf21(uint32_t value)
{
    const auto [x, y] = std::bit_cast<std::array<half, 2>>(value);
    return half2(x, y);
}

using vec_dot_KQ_t = float (*)(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ /*Q_q8*/, const void* __restrict__ /*Q_ds_v*/) {

    const half2* K_h2 = (const half2*)K_c;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / 2; k_KQ_0 += nthreads * cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            if constexpr (v_dot2_f32_f16_available_v) {
                ggml_cuda_mad(sum, tmp[k_KQ_1], ((const half2*)Q_v)[k_KQ_0 / nthreads + k_KQ_1]);
            } else {
                ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2*)Q_v)[k_KQ_0 / nthreads + k_KQ_1]);
            }
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char* __restrict__ K_c, const void* __restrict__ /*Q_v*/, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q4_0* K_q4_0 = (const block_q4_0*)K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D / sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI4_0;
        const int shift = k_KQ & (QI8_1 / 2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int) * iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0 / nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2*)Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi * Q_ds.x - (8 / QI8_1) * Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char* __restrict__ K_c, const void* __restrict__ /*Q_v*/, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q4_1* K_q4_1 = (const block_q4_1*)K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D / sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI4_1;
        const int shift = k_KQ & (QI8_1 / 2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int) * iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0 / nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2*)Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x * Q_ds.x * sumi + K_dm.y * Q_ds.y / QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char* __restrict__ K_c, const void* __restrict__ /*Q_v*/, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q5_0* K_q5_0 = (const block_q5_0*)K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D / sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI5_0;
        const int iqs8 = k_KQ % QI8_1;
        const int shift = k_KQ & (QI8_1 / 2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int) * iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh << 4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0 / nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2*)Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi * Q_ds.x - (16 / QI8_1) * Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char* __restrict__ K_c, const void* __restrict__ /*Q_v*/, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q5_1* K_q5_1 = (const block_q5_1*)K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D / sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI5_1;
        const int iqs8 = k_KQ % QI8_1;
        const int shift = k_KQ & (QI8_1 / 2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int) * iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh << 4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0 / nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2*)Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x * Q_ds.x * sumi + K_dm.y * Q_ds.y / QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char* __restrict__ K_c, const void* __restrict__ /*Q_v*/, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q8_0* K_q8_0 = (const block_q8_0*)K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D / sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4 * iqs);

        const float2* Q_ds = (const float2*)Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0 / nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0 / nthreads], std::bit_cast<half>(K_q8_0[ib].d), Q_d);
    }

    return sum;
}

template <internal::ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == internal::GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    }
    else if constexpr (type_K == internal::GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    }
    else if constexpr (type_K == internal::GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    }
    else if constexpr (type_K == internal::GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    }
    else if constexpr (type_K == internal::GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    }
    else if constexpr (type_K == internal::GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    }
    else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

using dequantize_V_t = void (*)(const void*, void*, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void* __restrict__ vx, void* __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne * sizeof(half)>(dst, (const half*)vx + i0);
    }
    else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne / 2];
        ggml_cuda_memcpy_1<ne * sizeof(half)>(tmp, (const half*)vx + i0);
        float2* dst_f2 = (float2*)dst;
#pragma unroll
        for (int l = 0; l < ne / 2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    }
    else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void* __restrict__ vx, void* __restrict__ dst, const int64_t i0) {
    const block_q4_0* x = (const block_q4_0*)vx;

	constexpr size_t QK4_0 = block_q4_0::block_size;
    const int64_t ib = i0 / QK4_0;
    const int     iqs = i0 % (QK4_0 / 2);
    const int     shift = (i0 % QK4_0) / (QK4_0 / 2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4 * shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t* q8 = (const int8_t*)&q;

    if constexpr (fp16_available_v && std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2*)dst)[l0 / 2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const float d = __half2float(x[ib].d);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float*)dst)[l] = d * q8[l];
        }
    }
    else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void* __restrict__ vx, void* __restrict__ dst, const int64_t i0) {
    const block_q4_1* x = (const block_q4_1*)vx;

    constexpr size_t QK4_1 = block_q4_1::block_size;
    const int64_t ib = i0 / QK4_1;
    const int     iqs = i0 % (QK4_1 / 2);
    const int     shift = (i0 % QK4_1) / (QK4_1 / 2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4 * shift;
    q &= 0x0F0F0F0F;

    const int8_t* q8 = (const int8_t*)&q;

    if constexpr (fp16_available_v && std::is_same_v<T, half>) {
        const half2 dm = __tohalf2(x[ib].dm);
        const half2 d = __half2half2(__low2half(dm));
        const half2 m = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2*)dst)[l0 / 2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float*)dst)[l] = dm.x * q8[l] + dm.y;
        }
    }
    else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void* __restrict__ vx, void* __restrict__ dst, const int64_t i0) {
    const block_q5_0* x = (const block_q5_0*)vx;

    constexpr size_t QK5_0 = block_q5_0::block_size;
    const int64_t ib = i0 / QK5_0;
    const int     idq = i0 % QK5_0;
    const int     iqs = i0 % (QK5_0 / 2);
    const int     shift = (i0 % QK5_0) / (QK5_0 / 2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4 * shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8 * l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t* q8 = (const int8_t*)&q;

    if constexpr (fp16_available_v && std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2*)dst)[l0 / 2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    }
    else {
        if constexpr (std::is_same_v<T, float>) {
            const float d = __half2float(x[ib].d);

#pragma unroll
            for (int l = 0; l < ne; ++l) {
                ((float*)dst)[l] = d * q8[l];
            }
        }
        else {
            static_assert(std::is_same_v<T, void>, "bad type");
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void* __restrict__ vx, void* __restrict__ dst, const int64_t i0) {
    const block_q5_1* x = (const block_q5_1*)vx;

    constexpr size_t QK5_1 = block_q5_1::block_size;
    const int64_t ib = i0 / QK5_1;
    const int     idq = i0 % QK5_1;
    const int     iqs = i0 % (QK5_1 / 2);
    const int     shift = (i0 % QK5_1) / (QK5_1 / 2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4 * shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8 * l + 4);
        }
    }

    const int8_t* q8 = (const int8_t*)&q;

    if constexpr (fp16_available_v && std::is_same_v<T, half>) {
        const half2 dm = __tohalf2(x[ib].dm);
        const half2 d = __half2half2(__low2half(dm));
        const half2 m = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2*)dst)[l0 / 2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    }
    else {
        if constexpr (std::is_same_v<T, float>) {
            const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
            for (int l = 0; l < ne; ++l) {
                ((float*)dst)[l] = dm.x * q8[l] + dm.y;
            }
        }
        else {
            static_assert(std::is_same_v<T, void>, "bad type");
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void* __restrict__ vx, void* __restrict__ dst, const int64_t i0) {
    const block_q8_0* x = (const block_q8_0*)vx;

    constexpr size_t QK8_0 = block_q8_0::block_size;
    const int64_t ib = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

    if constexpr (fp16_available_v && std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2*)dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    }
    else if constexpr (std::is_same<T, float>::value) {
        const float d = std::bit_cast<half>(x[ib].d);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float*)dst)[l] = d * qs[l];
        }
    }
    else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <internal::ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == internal::GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    }
    else if constexpr (type_V == internal::GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    }
    else if constexpr (type_V == internal::GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    }
    else if constexpr (type_V == internal::GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    }
    else if constexpr (type_V == internal::GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    }
    else if constexpr (type_V == internal::GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    }
    else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float* __restrict__ x, const float scale, int* __restrict__ yq32, void* __restrict__ yds) {

    float vals[sizeof(int)] = { 0.0f };
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4 * threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1 / 2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t* q8 = (int8_t*)&q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2*)yds)[threadIdx.x / QI8_1] = make_half2(d, sum);
        }
        else {
            ((float2*)yds)[threadIdx.x / QI8_1] = make_float2(d, sum);
        }
    }
}

template<int D> // D == head size
__launch_bounds__(D, 1)
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
        const float KQ_max_scale = expf(meta[l].x - kqmax);

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
    float* __restrict__ dst, const float2* __restrict__ dst_fixup, const int ne01, const int ne02, const int ne03,
    const int ne11, const int ne12, const int nbatch_fa) {
    constexpr int ncols = ncols1 * ncols2;

    const int bidx0 = blockIdx.x;
    const int j = blockIdx.y;
    const int c = blockIdx.z;
    const int jc = j * ncols2 + c;
    const int tid = threadIdx.x;

    const float* dst_fixup_data = ((const float*)dst_fixup) + gridDim.x * (2 * 2 * ncols);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int iter_k = (ne11 + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2 - 1)) / ncols2;

    const int kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;

    const bool did_not_have_any_data = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last = kbc0 / iter_k == kbc0_stop / iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const int sequence = kbc0 / (iter_k * iter_j * iter_z_gqa * ne12);
    const int z_KV = (kbc0 - iter_k * iter_j * iter_z_gqa * ne12 * sequence) / (iter_k * iter_j * iter_z_gqa);
    const int zt_gqa = (kbc0 - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV) / (iter_k * iter_j);
    const int jt = (kbc0 - iter_k * iter_j * iter_z_gqa * ne12 * sequence - iter_k * iter_j * iter_z_gqa * z_KV - iter_k * iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV * gqa_ratio + zt_gqa * ncols2; // Global Q head start index.

    if (jt * ncols1 + j >= ne01 || zt_gqa * ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence * ne02 * ne01 * D + jt * ne02 * (ncols1 * D) + zt_Q * D + (j * ne02 + c) * D + tid;

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
        const int kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / gridDim.x;
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
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    GGML_ASSERT(ctx.Q.type == internal::GGML_TYPE_F32);
    GGML_ASSERT(ctx.KQV.type == internal::GGML_TYPE_F32);

    GGML_ASSERT(ctx.Q.nb[0] == ctx.Q.element_size);
    GGML_ASSERT(ctx.K.nb0 == ctx.K.element_size);
    GGML_ASSERT(ctx.V.nb0 == ctx.V.element_size);

    GGML_ASSERT(!ctx.mask.exist || ctx.mask.type == internal::GGML_TYPE_F16);

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
    size_t nb10 = ctx.K.nb0;
    size_t nb11 = ctx.K.nb1;
    size_t nb12 = ctx.K.nb2;
    size_t nb13 = ctx.K.nb3;

    const char* V_data = (const char*)ctx.V.data;
    size_t nb20 = ctx.V.nb0;
    size_t nb21 = ctx.V.nb1;
    size_t nb22 = ctx.V.nb2;
    size_t nb23 = ctx.V.nb3;

    if (need_f16_K && ctx.K.type != internal::GGML_TYPE_F16) {
        const size_t bs = ctx.K.block_size;
        const size_t ts = ctx.K.type_size;

        K_f16.alloc(ctx.K.elements);
        GGML_ASSERT(ctx.K.nb0 == ts);
        convert_context convert_ctx {
            .src_type = ctx.K.type,
            .src_ne = { ctx.K.ne0, ctx.K.ne1, ctx.K.ne2, ctx.K.ne3 },
            .src_nb = { nb10, nb11, nb12, nb13 },
        };
        convert_to_cuda(convert_ctx, K_data, K_f16.ptr, main_stream);

        nb11 = ctx.K.ne0 * sizeof(half);
        nb12 = ctx.K.ne1 * nb11;
        nb13 = ctx.K.ne2 * nb12;
        K_data = (char*)K_f16.ptr;
    }

    if (need_f16_V && ctx.V.type != internal::GGML_TYPE_F16) {
        if (ctx.V_is_K_view) {
            V_data = K_data;
            nb21 = nb11;
            nb22 = nb12;
            nb23 = nb13;
        }
        else {
            const size_t bs = ctx.V.block_size;
            const size_t ts = ctx.V.type_size;

            V_f16.alloc(ctx.V.elements);
            GGML_ASSERT(ctx.V.nb0 == ts);
            convert_context convert_ctx{
                .src_type = ctx.V.type,
                .src_ne = { ctx.V.ne0, ctx.V.ne1, ctx.V.ne2, ctx.V.ne3 },
                .src_nb = { nb20, nb21, nb22, nb23 },
            };
            convert_to_cuda(convert_ctx, V_data, V_f16.ptr, main_stream);

            nb21 = ctx.V.ne0 * sizeof(half);
            nb22 = ctx.V.ne1 * nb21;
            nb23 = ctx.V.ne2 * nb22;
            V_data = (char*)V_f16.ptr;
        }
    }

    const int ntiles_x = ((ctx.Q.ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio = ctx.Q.ne[2] / ctx.K.ne2;
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_total = ntiles_x * ntiles_z_gqa * ctx.K.ne2 * ctx.Q.ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (ctx.mask.exist && ctx.K.ne1 % FATTN_KQ_STRIDE == 0 && (ctx.Q.ne[1] >= 1024 || ctx.Q.ne[3] > 1)) {
        const int s31 = ctx.mask.nb1 / sizeof(half2);
        const int s33 = ctx.mask.nb3 / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, ctx.Q.ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x * blocks_num_KV_max.y;
        const int iter_k = ctx.K.ne1 / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1> << <blocks_num_KV_max, block_dim_KV_max, 0, main_stream >> >
            ((const half2*)ctx.mask.data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm * nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks * tiles_nwaves);

        const int nblocks_stream_k = max_blocks;

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV / 2)));
        }
    }
    else {
        const int ntiles_KQ = (ctx.K.ne1 + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size

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
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
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
        blocks_num.z = ntiles_z_gqa * ctx.K.ne2 * ctx.Q.ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks * ctx.KQV.elements);
            dst_tmp_meta.alloc(parallel_blocks * ctx.KQV.nrows);
        }
    }

    float scale = ctx.scale;
    if (ctx.logit_softcap != 0.0f) {
        scale /= ctx.logit_softcap;
    }

    const uint32_t n_head = ctx.Q.ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(ctx.max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(ctx.max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(ctx.Q.ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel << <blocks_num, block_dim, nbytes_shared, main_stream >> > (
        ctx,
        K_data,
        V_data,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float*)ctx.KQV.data, dst_tmp_meta.ptr,
        scale, m0, m1, n_head_log2,
        ctx.Q.ne[0], ne01, ctx.Q.ne[2], ctx.Q.ne[3], ctx.Q.nb[1], ctx.Q.nb[2], ctx.Q.nb[3],
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
                ((float*)ctx.KQV.data, dst_tmp_meta.ptr, ctx.Q.ne[1], ctx.Q.ne[2], ctx.Q.ne[3], ctx.K.ne1, ctx.K.ne2, nbatch_fa);
        }
    }
    else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(ctx.Q.ne[1], ctx.Q.ne[2], ctx.Q.ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks * sizeof(float2);

        flash_attn_combine_results<DV>
            << <blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream >> >
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float*)ctx.KQV.data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}