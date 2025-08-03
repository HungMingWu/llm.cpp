#include <assert.h>
#include <float.h>
#include "cuda_func.h"
#include "common.cuh"
#include "vecdotq.cuh"
#include "fattn-common.cuh"
#include "../vendor_constant.h"
#include <bit>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_UNUSED(x) (void)(x)

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

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f16(
    const char* __restrict__ Q,
    const char* __restrict__ K,
    const char* __restrict__ V,
    const char* __restrict__ mask,
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
    const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f16_t vec_dot_KQ = get_vec_dot_KQ_f16<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f16_t dequantize_1_v = get_dequantize_1_f16(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03 * sequence + nb02 * head + nb01 * ic0;
    K += nb13 * sequence + nb12 * (head / gqa_ratio);
    V += nb23 * sequence + nb22 * (head / gqa_ratio);

    const half* maskh = (const half*)(mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half KQ[ncols * D];
    half2* KQ2 = (half2*)KQ;

    half kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
    }
    half kqsum[ncols] = { 0.0f };

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ half maskh_shared[ncols * D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskh_shared[j * D + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][D / (2 * WARP_SIZE)];
    int   Q_i32[ncols][D / (sizeof(int) * QK8_1) == 0 ? 1 : D / (sizeof(int) * QK8_1)];
    half2  Q_ds[ncols][D / QK8_1 == 0 ? 1 : D / QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int* tmp_q_i32 = (int*)&KQ[j * D];
            half2* tmp_q_ds = (half2*)(tmp_q_i32 + D / sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D / sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D / QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float* Q_f = (const float*)(Q + j * nb01);
#pragma unroll
            for (int i0 = 0; i0 < D / sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4 * i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int* tmp_q_i32 = (int*)&KQ[j * D];
            half2* tmp_q_ds = (half2*)(tmp_q_i32 + D / sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D / sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0 / WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0 / WARP_SIZE] = tmp_q_ds[i / QI8_1];
            }
        }

        __syncthreads();
    }
    else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2* Q_f2_j = (const float2*)(Q + j * nb01);

#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0 / WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }


#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j * D + tid] = -HALF_MAX_HALF;
    }
    __syncthreads();

    half2 VKQ[ncols] = { {0.0f, 0.0f} };

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;
    K += blockIdx.y * D * nb11;
    V += blockIdx.y * D * nb21;
    maskh += blockIdx.y * D;
    for (int k_VKQ_0 = blockIdx.y * D; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y * D,
        // Increment pointers after each loop:
        K += gridDim.y * D * nb11, V += gridDim.y * D * nb21, maskh += gridDim.y * D) {

        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskh_shared[j * D + tid] = slopeh * maskh[j * ne11 + tid];
            }
            __syncthreads();
        }

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + i_KQ * nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap * tanhf(sum);
                }

                sum += maskh_shared[j * D + i_KQ];

                if (ncols == 1) {
                    kqmax_new = ggml_cuda_hmax(kqmax_new, sum);
                }
                else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j * D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j * D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j] * KQ_max_scale + val;
            KQ[j * D + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k0 + 0) * nb21, tid);
            reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k0 + 1) * nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k * KQ2[j * (D / 2) + k0 / 2];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum((float)kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum((float)kqsum[j_VKQ]);

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (gridDim.y == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        dst[(((sequence * ne01 + ic0 + j_VKQ) * ne02 + head) * gridDim.y + blockIdx.y) * D + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence * ne01 + ic0 + tid) * ne02 + head) * gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta);
    GGML_UNUSED(scale); GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13);
    GGML_UNUSED(nb21); GGML_UNUSED(nb22); GGML_UNUSED(nb23);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
    GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

using vec_dot_KQ_f32_t = float (*)(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds);

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_0(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q4_0* K_q4_0 = (const block_q4_0*)K_c;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / sizeof(int); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI4_0;
        const int shift = k_KQ & (QI8_1 / 2);

        const int v = (get_int_b2(K_q4_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0 / warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2* Q_ds = (const half2*)Q_ds_v;

            const half2 sum2 = __half2half2(K_q4_0[ib].d) * Q_ds[k_KQ_0 / warp_size];
            sum += (T)(((half)sumi) * __low2half(sum2) - __high2half(sum2) /* *8/QI8_1 == 1 */);
        }
        else
#endif // FP16_AVAILABLE
        {
            const float2* Q_ds = (const float2*)Q_ds_v;

            sum += (T)(__half2float(K_q4_0[ib].d) * (sumi * Q_ds[k_KQ_0 / warp_size].x - (8 / QI8_1) * Q_ds[k_KQ_0 / warp_size].y));
        }
    }

    return sum;
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_1(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q4_1* K_q4_1 = (const block_q4_1*)K_c;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / sizeof(int); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI4_1;
        const int shift = k_KQ & (QI8_1 / 2);

        const int v = (get_int_b4(K_q4_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0 / warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2* Q_ds = (const half2*)Q_ds_v;

            const half2 d4d8_m4s8 = K_q4_1[ib].dm * Q_ds[k_KQ_0 / warp_size];
            const half2 sumid4d8_m4s8scaled = d4d8_m4s8 * make_half2(sumi, 1.0f / QI8_1);
            sum += (T)(__low2half(sumid4d8_m4s8scaled) + __high2half(sumid4d8_m4s8scaled));
        }
        else
#endif // FP16_AVAILABLE
        {
            const float2* Q_ds = (const float2*)Q_ds_v;

            const float sumid4d8 = __low2float1(K_q4_1[ib].dm) * Q_ds[k_KQ_0 / warp_size].x * sumi;
            const float m4s8scaled = __high2float1(K_q4_1[ib].dm) * Q_ds[k_KQ_0 / warp_size].y / QI8_1;

            sum += (T)(sumid4d8 + m4s8scaled);
        }
    }

    return sum;
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_0(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q5_0* K_q5_0 = (const block_q5_0*)K_c;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / sizeof(int); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI5_0;
        const int iqs8 = k_KQ % QI8_1;
        const int shift = k_KQ & (QI8_1 / 2);

        int v = (get_int_b2(K_q5_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_b2(K_q5_0[ib].qh, 0) >> (iqs8 * QI5_0);
        v |= (vh << 4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0 / warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2* Q_ds = (const half2*)Q_ds_v;

            const half2 sum2 = __half2half2(K_q5_0[ib].d) * Q_ds[k_KQ_0 / warp_size];
            sum += (T)(((half)sumi) * __low2half(sum2) - __high2half(sum2) * __float2half(2.0f)) /* *16/QI8_1 == 2 */;
        }
        else
#endif // FP16_AVAILABLE
        {
            const float2* Q_ds = (const float2*)Q_ds_v;

            sum += (T)(__half2float(K_q5_0[ib].d) * (sumi * Q_ds[k_KQ_0 / warp_size].x - (16 / QI8_1) * Q_ds[k_KQ_0 / warp_size].y));
        }
    }

    return sum;
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_1(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q5_1* K_q5_1 = (const block_q5_1*)K_c;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / sizeof(int); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib = k_KQ / QI8_1;
        const int iqs4 = k_KQ % QI5_1;
        const int iqs8 = k_KQ % QI8_1;
        const int shift = k_KQ & (QI8_1 / 2);

        int v = (get_int_b2(K_q5_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_b2(K_q5_1[ib].qh, 0) >> (iqs8 * QI5_1);
        v |= (vh << 4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0 / warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2* Q_ds = (const half2*)Q_ds_v;

            const half2 d5d8_m5s8 = K_q5_1[ib].dm * Q_ds[k_KQ_0 / warp_size];
            const half2 sumid5d8_m5s8scaled = d5d8_m5s8 * make_half2(sumi, 1.0f / QI8_1);
            sum += (T)(__low2half(sumid5d8_m5s8scaled) + __high2half(sumid5d8_m5s8scaled));
        }
        else
#endif // FP16_AVAILABLE
        {
            const float2* Q_ds = (const float2*)Q_ds_v;

            const float sumid5d8 = __low2float1(K_q5_1[ib].dm) * Q_ds[k_KQ_0 / warp_size].x * sumi;
            const float m5s8scaled = __high2float1(K_q5_1[ib].dm) * Q_ds[k_KQ_0 / warp_size].y / QI8_1;

            sum += (T)(sumid5d8 + m5s8scaled);
        }
    }

    return sum;
}

template <typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q8_0(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const block_q8_0* K_q8_0 = (const block_q8_0*)K_c;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / sizeof(int); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        const int v = get_int_b2(K_q8_0[ib].qs, iqs);

        T Q_d;
        if (std::is_same<T, half>::value) {
            const half2* Q_ds = (const half2*)Q_ds_v;
            Q_d = __low2half(Q_ds[k_KQ_0 / warp_size]);
        }
        else {
            const float2* Q_ds = (const float2*)Q_ds_v;
            Q_d = Q_ds[k_KQ_0 / warp_size].x;
        }

        sum += vec_dot_q8_0_q8_1_impl<T, 1>(&v, &Q_q8[k_KQ_0 / warp_size], std::bit_cast<half>(K_q8_0[ib].d), Q_d);
    }

    return sum;
}

template <typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_f16(
    const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds_v) {

    const half2* K_h2 = (const half2*)K_c;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        const half2* Q_h2 = (const half2*)Q_v;

        half2 sum2 = make_half2(0.0f, 0.0f);

#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D / 2; k_KQ_0 += warp_size) {
            const int k_KQ = k_KQ_0 + threadIdx.x;

            const half2 K_ik = K_h2[k_KQ];
            sum2 += K_ik * Q_h2[k_KQ_0 / warp_size];
        }

        return __low2half(sum2) + __high2half(sum2);
    }
#endif // FP16_AVAILABLE

    const float2* Q_f2 = (const float2*)Q_v;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D / 2; k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const half2 K_ik = K_h2[k_KQ];
        sum += __low2float(K_ik) * Q_f2[k_KQ_0 / warp_size].x;
        sum += __high2float(K_ik) * Q_f2[k_KQ_0 / warp_size].y;
    }

    return sum;
}

template <int D>
constexpr __device__ vec_dot_KQ_f32_t get_vec_dot_KQ_f32(ggml_type type_K) {
    return type_K == GGML_TYPE_Q4_0 ? vec_dot_fattn_vec_KQ_q4_0<float, D> :
        type_K == GGML_TYPE_Q4_1 ? vec_dot_fattn_vec_KQ_q4_1<float, D> :
        type_K == GGML_TYPE_Q5_0 ? vec_dot_fattn_vec_KQ_q5_0<float, D> :
        type_K == GGML_TYPE_Q5_1 ? vec_dot_fattn_vec_KQ_q5_1<float, D> :
        type_K == GGML_TYPE_Q8_0 ? vec_dot_fattn_vec_KQ_q8_0<float, D> :
        type_K == GGML_TYPE_F16 ? vec_dot_fattn_vec_KQ_f16<float, D> :
        nullptr;
}

using dequantize_1_f32_t = float (*)(const void*, const int64_t);

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_0(const void* __restrict__ vx, const int64_t i) {
    const block_q4_0* x = (const block_q4_0*)vx;

    const int64_t ib = i / block_q4_0::block_size;
    const int     iqs = i % (block_q4_0::block_size / 2);
    const int     shift = (i % block_q4_0::block_size) / (block_q4_0::block_size / 2);

    const T   d = std::bit_cast<half>(x[ib].d);
    const int q0 = x[ib].qs[iqs];
    const int q = ((q0 >> (4 * shift)) & 0x0F) - 8;

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half)d) * ((half)q);
    }
#endif // FP16_AVAILABLE

    return ((float)d) * ((float)q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_1(const void* __restrict__ vx, const int64_t i) {
    const block_q4_1* x = (const block_q4_1*)vx;

    const int64_t ib = i / block_q4_1::block_size;
    const int     iqs = i % (block_q4_1::block_size / 2);
    const int     shift = (i % block_q4_1::block_size) / (block_q4_1::block_size / 2);

    const half2 dm = __tohalf21(x[ib].dm);
    const int   q0 = x[ib].qs[iqs];
    const int   q = ((q0 >> (4 * shift)) & 0x0F);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm) * ((half)q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm) * ((float)q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_0(const void* __restrict__ vx, const int64_t i) {
    const block_q5_0* x = (const block_q5_0*)vx;

    const int64_t ib = i / block_q5_0::block_size;
    const int     idq = i % block_q5_0::block_size;
    const int     iqs = i % (block_q5_0::block_size / 2);
    const int     shift = (i % block_q5_0::block_size) / (block_q5_0::block_size / 2);

    const T   d = std::bit_cast<half>(x[ib].d);
    const int ql0 = x[ib].qs[iqs];
    const int qh0 = get_int_b2(x[ib].qh, 0);
    const int ql = ((ql0 >> (4 * shift)) & 0x0F);
    const int qh = ((qh0 >> idq) << 4) & 0x10;
    const int q = (ql | qh) - 16;

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half)d) * ((half)q);
    }
#endif // FP16_AVAILABLE

    return ((float)d) * ((float)q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_1(const void* __restrict__ vx, const int64_t i) {
    const block_q5_1* x = (const block_q5_1*)vx;

    const int64_t ib = i / block_q5_1::block_size;
    const int     idq = i % block_q5_1::block_size;
    const int     iqs = i % (block_q5_1::block_size / 2);
    const int     shift = (i % block_q5_1::block_size) / (block_q5_1::block_size / 2);

    const half2 dm = __tohalf21(x[ib].dm);
    const int   ql0 = x[ib].qs[iqs];
    const int   qh0 = get_int_b4(x[ib].qh, 0);
    const int   ql = ((ql0 >> (4 * shift)) & 0x0F);
    const int   qh = ((qh0 >> idq) << 4) & 0x10;
    const int   q = (ql | qh);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm) * ((half)q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm) * ((float)q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q8_0(const void* __restrict__ vx, const int64_t i) {
    const block_q8_0* x = (const block_q8_0*)vx;

    const int64_t ib = i / block_q8_0::block_size;
    const int     iqs = i % block_q8_0::block_size;;

    const T   d = std::bit_cast<half>(x[ib].d);
    const int q = x[ib].qs[iqs];

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half)d) * ((half)q);
    }
#endif // FP16_AVAILABLE

    return ((float)d) * ((float)q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_f16(const void* __restrict__ vx, const int64_t i) {
    const half* x = (const half*)vx;

    return x[i];
}

constexpr __device__ dequantize_1_f32_t get_dequantize_1_f32(ggml_type type_V) {
    return type_V == GGML_TYPE_Q4_0 ? dequantize_1_q4_0<float> :
        type_V == GGML_TYPE_Q4_1 ? dequantize_1_q4_1<float> :
        type_V == GGML_TYPE_Q5_0 ? dequantize_1_q5_0<float> :
        type_V == GGML_TYPE_Q5_1 ? dequantize_1_q5_1<float> :
        type_V == GGML_TYPE_Q8_0 ? dequantize_1_q8_0<float> :
        type_V == GGML_TYPE_F16 ? dequantize_1_f16<float> :
        nullptr;
}

template <typename Tds>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float* __restrict__ x, const float scale, int* __restrict__ yq32, void* __restrict__ yds) {

    float vals[sizeof(int)] = { 0.0f };
#pragma unroll
    for (int l = 0; l < sizeof(int); ++l) {
        vals[l] = scale * x[4 * threadIdx.x + l];
    }

    float amax = fabsf(vals[0]);
    float sum = vals[0];
#pragma unroll
    for (int l = 1; l < sizeof(int); ++l) {
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
        for (int l = 0; l < sizeof(int); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0) {
        if (std::is_same<Tds, half2>::value) {
            ((half2*)yds)[threadIdx.x / QI8_1] = make_half2(d, sum);
        }
        else {
            ((float2*)yds)[threadIdx.x / QI8_1] = make_float2(d, sum);
        }
    }
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f32(
    const char* __restrict__ Q,
    const char* __restrict__ K,
    const char* __restrict__ V,
    const char* __restrict__ mask,
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
    const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
        GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
        GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
        GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
        GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02);
        GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne11);
        GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
        GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33); GGML_UNUSED(nb01); GGML_UNUSED(nb02);
        GGML_UNUSED(nb03); GGML_UNUSED(nb11); GGML_UNUSED(nb12);
        GGML_UNUSED(nb13); GGML_UNUSED(nb21); GGML_UNUSED(nb22);
        GGML_UNUSED(nb23);
        NO_DEVICE_CODE;
        return;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f32_t vec_dot_KQ = get_vec_dot_KQ_f32<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f32_t dequantize_1_v = get_dequantize_1_f32(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03 * sequence + nb02 * head + nb01 * ic0;
    K += nb13 * sequence + nb12 * (head / gqa_ratio);
    V += nb23 * sequence + nb22 * (head / gqa_ratio);

    const half* maskh = (const half*)(mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float KQ[ncols * D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j * D + tid] = -FLT_MAX / 2.0f;
    }

    float kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX / 2.0f;
    }
    float kqsum[ncols] = { 0.0f };

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX / 2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ float maskf_shared[ncols * D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskf_shared[j * D + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    float2  Q_f2[ncols][D / (2 * WARP_SIZE)];
    int    Q_i32[ncols][D / (sizeof(int) * QK8_1) == 0 ? 1 : D >= D / (sizeof(int) * QK8_1)];
    float2  Q_ds[ncols][D / QK8_1 == 0 ? 1 : D / QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int* tmp_q_i32 = (int*)&KQ[j * D];
            float2* tmp_q_ds = (float2*)(tmp_q_i32 + D / sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < int(D / sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D / QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
                continue;
            }

            const float* Q_f = (const float*)(Q + j * nb01);
#pragma unroll
            for (int i0 = 0; i0 < int(D / sizeof(int)); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2>(Q_f + 4 * i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int* tmp_q_i32 = (int*)&KQ[j * D];
            float2* tmp_q_ds = (float2*)(tmp_q_i32 + D / sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D / sizeof(int)); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0 / WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0 / WARP_SIZE] = tmp_q_ds[i / QI8_1];
            }
        }

        __syncthreads();
    }
    else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2* Q_f2_j = (const float2*)(Q + j * nb01);
#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_f2[j][i0 / WARP_SIZE] = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_f2[j][i0 / WARP_SIZE].x *= scale;
                Q_f2[j][i0 / WARP_SIZE].y *= scale;
            }
        }
    }

    float VKQ[ncols] = { 0.0f };

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;
    K += blockIdx.y * D * nb11;
    V += blockIdx.y * D * nb21;
    maskh += blockIdx.y * D;
    for (int k_VKQ_0 = blockIdx.y * D; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y * D,
        // Increment pointers after each loop:
        K += gridDim.y * D * nb11, V += gridDim.y * D * nb21, maskh += gridDim.y * D) {

        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskf_shared[j * D + tid] = slope * __half2float(maskh[j * ne11 + tid]);
            }
            __syncthreads();
        }

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ * nb11, Q_f2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap * tanhf(sum);
                }

                sum += maskf_shared[j * D + i_KQ];

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x == 0) {
                    KQ[j * D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(KQ[j * D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j] * KQ_max_scale + val;
            KQ[j * D + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < D; ++k) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            const float V_ki = dequantize_1_v(V + k * nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_ki * KQ[j * D + k];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        float dst_val = VKQ[j_VKQ];
        if (gridDim.y == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        dst[(((sequence * ne01 + ic0 + j_VKQ) * ne02 + head) * gridDim.y + blockIdx.y) * D + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence * ne01 + ic0 + tid) * ne02 + head) * gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
    GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33);
    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13);
    GGML_UNUSED(nb21); GGML_UNUSED(nb22); GGML_UNUSED(nb23);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f16_case_impl(const flash_attn_ext_context& ctx) {
    constexpr int nwarps = D / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f32_case_impl(const flash_attn_ext_context& ctx) {
    constexpr int nwarps = D / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = D != 128;
    constexpr bool need_f16_V = D != 128 && D != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f16_case(const flash_attn_ext_context& ctx) {

    GGML_ASSERT(ctx.precision == GGML_PREC_DEFAULT);

    GGML_ASSERT(ctx.K.type == type_K);
    GGML_ASSERT(ctx.V.type == type_V);

    if (ctx.Q.ne1 == 1) {
        constexpr int cols_per_block = 1;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    if (ctx.Q.ne1 == 2) {
        constexpr int cols_per_block = 2;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    if (ctx.Q.ne1 <= 4) {
        constexpr int cols_per_block = 4;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (ctx.logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
    }
    else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
    }
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_case(const flash_attn_ext_context& ctx) {
    GGML_ASSERT(ctx.K.type == type_K);
    GGML_ASSERT(ctx.V.type == type_V);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (ctx.Q.ne1 == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    if (ctx.Q.ne1 == 2) {
        constexpr int cols_per_block = 2;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    if (ctx.Q.ne1 <= 4) {
        constexpr int cols_per_block = 4;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (ctx.logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
    }
    else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
    }
}

static void on_no_fattn_vec_case(const int D) {
    if (D == 64) {
        fprintf(stderr, "Unsupported KV type combination for head_size 64.\n");
        fprintf(stderr, "By default only f16 KV cache is supported.\n");
        fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for V cache quantization support.\n");
        GGML_ABORT("fatal error");
    }
    else if (D == 128) {
        fprintf(stderr, "Unsupported KV type combination for head_size 128.\n");
        fprintf(stderr, "Supported combinations:\n");
        fprintf(stderr, "  - K == q4_0, V == q4_0,  4.50 BPV\n");
        fprintf(stderr, "  - K == q8_0, V == q8_0,  8.50 BPV\n");
        fprintf(stderr, "  - K == f16,  V == f16,  16.00 BPV\n");
        fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.\n");
        GGML_ABORT("fatal error");
    }
    else {
        fprintf(stderr, "Unsupported KV type combination for head_size 256.\n");
        fprintf(stderr, "Only f16 is supported.\n");
        GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_F16_CASE(D, type_K, type_V)                                      \
    if (ctx.Q.ne0 == (D) && ctx.K.type == (type_K) && ctx.V.type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<D, type_K, type_V>(ctx);             \
        return;                                                                    \
    }

void ggml_cuda_flash_attn_ext_vec_f16(const flash_attn_ext_context& ctx)
{
#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(ctx.Q.ne0);
}

#define FATTN_VEC_F32_CASE(D, type_K, type_V)                                      \
    if (ctx.Q.ne0 == (D) && ctx.K.type == (type_K) && ctx.V.type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<D, type_K, type_V>(ctx);             \
        return;                                                                    \
    }  

void ggml_cuda_flash_attn_ext_vec_f32(const flash_attn_ext_context& ctx)
{
#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q4_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q5_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(ctx.Q.ne0);
}

template<int D, int ncols, int nwarps, bool use_logit_softcap> // D == head size
#if !defined(GGML_USE_HIP)
__launch_bounds__(nwarps* WARP_SIZE, 2)
#endif // !defined(GGML_USE_HIP)
static __global__ void flash_attn_tile_ext_f32(
    const char* __restrict__ Q,
    const char* __restrict__ K,
    const char* __restrict__ V,
    const char* __restrict__ mask,
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
    const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
        GGML_UNUSED(dst); GGML_UNUSED(dst_meta);
        GGML_UNUSED(scale); GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
        GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
        GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
        GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
        GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
        GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13);
        GGML_UNUSED(nb21); GGML_UNUSED(nb22); GGML_UNUSED(nb23);
        GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
        GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33);
        NO_DEVICE_CODE;
        return;
    }

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2* Q_f2 = (const float2*)(Q + nb03 * sequence + nb02 * head + nb01 * ic0);
    const half2* K_h2 = (const half2*)(K + nb13 * sequence + nb12 * (head / gqa_ratio));
    const half2* V_h2 = (const half2*)(V + nb13 * sequence + nb12 * (head / gqa_ratio)); // K and V have same shape
    const half* maskh = (const half*)(mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ float KQ[ncols * FATTN_KQ_STRIDE_TILE_F32];

    __shared__ float KV_tmp[FATTN_KQ_STRIDE_TILE_F32][D + 1]; // Pad D to avoid memory bank conflicts.
    float2* KV_tmp2 = (float2*)KV_tmp;

    float kqmax[ncols / nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0 / nwarps] = -FLT_MAX / 2.0f;
    }
    float kqsum[ncols / nwarps] = { 0.0f };

    float2 VKQ[ncols / nwarps][(D / 2) / WARP_SIZE] = { {{0.0f, 0.0f}} };

    // Convert Q to half2 and store in registers:
    __shared__ float Q_f[ncols][D];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += 2 * WARP_SIZE) {
            float2 tmp = ic0 + j < ne01 ? Q_f2[j * (nb01 / sizeof(float2)) + i0 / 2 + threadIdx.x] : make_float2(0.0f, 0.0f);
            Q_f[j][i0 + 0 * WARP_SIZE + threadIdx.x] = tmp.x * scale;
            Q_f[j][i0 + 1 * WARP_SIZE + threadIdx.x] = tmp.y * scale;
        }
    }

    __syncthreads();

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;
    for (int k_VKQ_0 = blockIdx.y * FATTN_KQ_STRIDE_TILE_F32; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y * FATTN_KQ_STRIDE_TILE_F32) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols / nwarps];
#pragma unroll
        for (int j = 0; j < ncols / nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 2 * WARP_SIZE) {
                const half2 tmp = K_h2[int64_t(k_VKQ_0 + i_KQ) * stride_KV2 + k_KQ_0 / 2 + threadIdx.x];
                KV_tmp[i_KQ][k_KQ_0 + 0 * WARP_SIZE + threadIdx.x] = __low2float(tmp);
                KV_tmp[i_KQ][k_KQ_0 + 1 * WARP_SIZE + threadIdx.x] = __high2float(tmp);
            }
        }

        __syncthreads();

        float sum[FATTN_KQ_STRIDE_TILE_F32 / WARP_SIZE][ncols / nwarps] = { {0.0f} };

#pragma unroll
        for (int k_KQ = 0; k_KQ < D; ++k_KQ) {
            float K_k[FATTN_KQ_STRIDE_TILE_F32 / WARP_SIZE];
            float Q_k[ncols / nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0 / WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0 / nwarps] = Q_f[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    sum[i_KQ_0 / WARP_SIZE][j_KQ_0 / nwarps] += K_k[i_KQ_0 / WARP_SIZE] * Q_k[j_KQ_0 / nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                if (use_logit_softcap) {
                    sum[i_KQ_0 / WARP_SIZE][j_KQ_0 / nwarps] = logit_softcap * tanhf(sum[i_KQ_0 / WARP_SIZE][j_KQ_0 / nwarps]);
                }

                sum[i_KQ_0 / WARP_SIZE][j_KQ_0 / nwarps] += mask ? slope * __half2float(maskh[j_KQ * ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0 / nwarps] = fmaxf(kqmax_new[j_KQ_0 / nwarps], sum[i_KQ_0 / WARP_SIZE][j_KQ_0 / nwarps]);

                KQ[j_KQ * FATTN_KQ_STRIDE_TILE_F32 + i_KQ] = sum[i_KQ_0 / WARP_SIZE][j_KQ_0 / nwarps];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0 / nwarps] = warp_reduce_max(kqmax_new[j0 / nwarps]);
            const float KQ_max_scale = expf(kqmax[j0 / nwarps] - kqmax_new[j0 / nwarps]);
            kqmax[j0 / nwarps] = kqmax_new[j0 / nwarps];

            float kqsum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F32; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float diff = KQ[j * FATTN_KQ_STRIDE_TILE_F32 + i] - kqmax[j0 / nwarps];
                const float val = expf(diff);
                kqsum_add += val;
                KQ[j * FATTN_KQ_STRIDE_TILE_F32 + i] = val;
            }
            kqsum[j0 / nwarps] = kqsum[j0 / nwarps] * KQ_max_scale + kqsum_add;

#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += WARP_SIZE) {
                VKQ[j0 / nwarps][i0 / WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0 / nwarps][i0 / WARP_SIZE].y *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F32; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const half2 tmp = V_h2[int64_t(k_VKQ_0 + k) * stride_KV2 + i];
                KV_tmp2[k * (D / 2) + i].x = __low2float(tmp);
                KV_tmp2[k * (D / 2) + i].y = __high2float(tmp);
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < FATTN_KQ_STRIDE_TILE_F32; ++k) {
            float2 V_k[(D / 2) / WARP_SIZE];
            float  KQ_k[ncols / nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0 / WARP_SIZE] = KV_tmp2[k * (D / 2) + i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0 / nwarps] = KQ[j * FATTN_KQ_STRIDE_TILE_F32 + k];
            }

#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0 / nwarps][i0 / WARP_SIZE].x += V_k[i0 / WARP_SIZE].x * KQ_k[j0 / nwarps];
                    VKQ[j0 / nwarps][i0 / WARP_SIZE].y += V_k[i0 / WARP_SIZE].y * KQ_k[j0 / nwarps];
                }
            }
        }

        __syncthreads();
    }

    float2* dst2 = (float2*)dst;

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float kqsum_j = kqsum[j_VKQ_0 / nwarps];
        kqsum_j = warp_reduce_sum(kqsum_j);

        const int j_dst_unrolled = ((sequence * ne01 + ic0 + j_VKQ) * ne02 + head) * gridDim.y + blockIdx.y;

#pragma unroll
        for (int i00 = 0; i00 < D / 2; i00 += WARP_SIZE) {
            const int i0 = i00 + threadIdx.x;

            float2 dst_val = VKQ[j_VKQ_0 / nwarps][i0 / WARP_SIZE];
            if (gridDim.y == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            dst2[j_dst_unrolled * (D / 2) + i0] = dst_val;
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(kqmax[j_VKQ_0 / nwarps], kqsum_j);
        }
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta);
    GGML_UNUSED(scale); GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13);
    GGML_UNUSED(nb21); GGML_UNUSED(nb22); GGML_UNUSED(nb23);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
    GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int cols_per_block, bool use_logit_softcap>
void launch_fattn_tile_f32_64_128(const flash_attn_ext_context& ctx) {
    switch (ctx.Q.ne0) {
    case  64: {
        constexpr int    D = 64;
        constexpr int    nwarps = 8;
        constexpr size_t nbytes_shared = 0;
        fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, use_logit_softcap>;
        launch_fattn<D, cols_per_block, 1>
            (ctx, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F32, true, true, false);
    } break;
    case 128: {
        constexpr int    D = 128;
        constexpr int    nwarps = 8;
        constexpr size_t nbytes_shared = 0;
        fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, use_logit_softcap>;
        launch_fattn<D, cols_per_block, 1>
            (ctx, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F32, true, true, false);
    } break;
    default: {
        GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
    } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f32(const flash_attn_ext_context& ctx) {
    if (ctx.Q.ne1 <= 16) {
        constexpr int cols_per_block = 16;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    if (ctx.logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx);
    }
    else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx);
    }
}