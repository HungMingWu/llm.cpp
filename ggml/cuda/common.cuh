#pragma once
#include <assert.h>
#include <stdio.h>
#include <bit>

#include "common.h"
#include "block.h"

[[noreturn]]
static __device__ void no_device_code(
    const char* file_name, const int line, const char* function_name, const int arch, [[maybe_unused]] const char* arch_list)
{

#if defined(GGML_USE_HIP)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
        file_name, line, function_name, arch);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
        file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIP)
    __trap();

#if defined(GGML_USE_MUSA)
    __builtin_unreachable();
#endif // defined(GGML_USE_MUSA)
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    }
    else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __any_sync(0xffffffff, x);
    }
    else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, [[maybe_unused]] const half b) {
#ifdef FP16_AVAILABLE

#if !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    return __hmax(a, b);
#endif // !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2([[maybe_unused]] const half2 a, [[maybe_unused]] const half2 b) {
#if defined(GGML_USE_HIP) && HIP_VERSION >= 50700000
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
#elif !defined(GGML_USE_HIP) && CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#elif !defined(GGML_USE_HIP)
    half2 ret;
    reinterpret_cast<half&>(ret.x) = __float2half(fmaxf(__low2float(a), __low2float(b)));
    reinterpret_cast<half&>(ret.y) = __float2half(fmaxf(__high2float(a), __high2float(b)));
    return ret;
#else
    NO_DEVICE_CODE;
#endif
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    assert(d_64 != 0);
    assert(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t)((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI4_0 (block_q4_0::block_size / (4 * QR4_0))
#define QR4_0 2

#define QI4_1 (block_q4_1::block_size / (4 * QR4_1))
#define QR4_1 2

#define QI_MXFP4 (block_mxfp4::block_size / (4 * QR_MXFP4))
#define QR_MXFP4 2

#define QI5_0 (block_q5_0::block_size / (4 * QR5_0))
#define QR5_0 2

#define QI5_1 (block_q5_1::block_size / (4 * QR5_1))
#define QR5_1 2

#define QI8_0 (block_q8_0::block_size / (4 * QR8_0))
#define QR8_0 1

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2

#define QI5_K (QK_K / (4*QR5_K))
#define QR5_K 2

#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2

#define QI2_XXS (QK_K / (4*QR2_XXS))
#define QR2_XXS 4

#define QI2_XS (QK_K / (4*QR2_XS))
#define QR2_XS 4

#define QI2_S (QK_K / (4*QR2_S))
#define QR2_S 4

#define QI3_XXS (QK_K / (4*QR3_XXS))
#define QR3_XXS 4

#define QI3_XS (QK_K / (4*QR3_XS))
#define QR3_XS 4

#define QI1_S (QK_K / (4*QR1_S))
#define QR1_S 8

#define QI1_M (QK_K / (4*QR1_M))
#define QR1_M 8

#define QI4_NL (block_iq4_nl::block_size / (4*QR4_NL))
#define QR4_NL 2

#define QI4_XS (QK_K / (4*QR4_XS))
#define QR4_XS 2

#define QI3_S (QK_K / (4*QR3_S))
#define QR3_S 4

template <typename type>
struct ggml_cuda_type_traits;

static constexpr int VDR_Q4_0_Q8_1_MMVQ = 2;
static constexpr int VDR_Q4_1_Q8_1_MMVQ = 2;
static constexpr int VDR_Q5_0_Q8_1_MMVQ = 2;
static constexpr int VDR_Q5_1_Q8_1_MMVQ = 2;
static constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;
static constexpr int VDR_MXFP4_Q8_1_MMVQ = 2;
static constexpr int VDR_Q2_K_Q8_1_MMVQ = 1;
static constexpr int VDR_Q3_K_Q8_1_MMVQ = 1;
static constexpr int VDR_Q4_K_Q8_1_MMVQ = 2;
static constexpr int VDR_Q5_K_Q8_1_MMVQ = 2;
static constexpr int VDR_Q6_K_Q8_1_MMVQ = 1;
static constexpr int VDR_IQ1_S_Q8_1_MMVQ = 1;
static constexpr int VDR_IQ1_M_Q8_1_MMVQ = 1;
static constexpr int VDR_IQ2_S_Q8_1_MMVQ = 2;
static constexpr int VDR_IQ2_XS_Q8_1_MMVQ = 2;
static constexpr int VDR_IQ2_XXS_Q8_1_MMVQ = 2;
static constexpr int VDR_IQ3_S_Q8_1_MMVQ = 2;
static constexpr int VDR_IQ3_XXS_Q8_1_MMVQ = 2;
static constexpr int VDR_IQ4_NL_Q8_1_MMVQ = 2;
static constexpr int VDR_IQ4_XS_Q8_1_MMVQ = 4;

template<>
struct ggml_cuda_type_traits<block_q4_0> {
    static constexpr int qk = block_q4_0::block_size;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
    static constexpr int mmvq = VDR_Q4_0_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q4_1> {
    static constexpr int qk = block_q4_1::block_size;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
    static constexpr int mmvq = VDR_Q4_1_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q5_0> {
    static constexpr int qk = block_q5_0::block_size;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
    static constexpr int mmvq = VDR_Q5_0_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q5_1> {
    static constexpr int qk = block_q5_1::block_size;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
    static constexpr int mmvq = VDR_Q5_1_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q8_0> {
    static constexpr int qk = block_q8_0::block_size;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
    static constexpr int mmvq = VDR_Q8_0_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_mxfp4> {
    static constexpr int qk = block_mxfp4::block_size;;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
    static constexpr int mmvq = VDR_MXFP4_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
    static constexpr int mmvq = VDR_Q2_K_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
    static constexpr int mmvq = VDR_Q3_K_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
    static constexpr int mmvq = VDR_Q4_K_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int mmvq = VDR_Q5_K_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int mmvq = VDR_Q6_K_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq2_xxs> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
    static constexpr int mmvq = VDR_IQ2_XXS_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq2_xs> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
    static constexpr int mmvq = VDR_IQ2_XS_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq2_s> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
    static constexpr int mmvq = VDR_IQ2_S_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq3_xxs> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
    static constexpr int mmvq = VDR_IQ3_XXS_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq1_s> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
    static constexpr int mmvq = VDR_IQ1_S_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq1_m> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
    static constexpr int mmvq = VDR_IQ1_M_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq4_nl> {
    static constexpr int qk = block_iq4_nl::block_size;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
    static constexpr int mmvq = VDR_IQ4_NL_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq4_xs> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
    static constexpr int mmvq = VDR_IQ4_XS_Q8_1_MMVQ;
};

template<>
struct ggml_cuda_type_traits<block_iq3_s> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
    static constexpr int mmvq = VDR_IQ3_S_Q8_1_MMVQ;
};

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else // defined(GGML_USE_HIP)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t* a8 = (const int8_t*)&a;
    const int8_t* b8 = (const int8_t*)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)

#endif // defined(GGML_USE_HIP)
}

static bool cp_async_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

// may change later

static __device__ __forceinline__ float __half2float(uint16_t value)
{
    return __half2float(std::bit_cast<half>(value));
}

static __device__ __forceinline__ float __low2float(uint32_t value)
{
    const auto [x, _] = std::bit_cast<std::array<half, 2>>(value);
    return __half2float(x);
}

static __device__ __forceinline__ half __low2half(uint32_t value)
{
    const auto [x, _] = std::bit_cast<std::array<half, 2>>(value);
    return x;
}

static __device__ __forceinline__ half __high2half(uint32_t value)
{
    const auto [_, y] = std::bit_cast<std::array<half, 2>>(value);
    return y;
}

static __device__ __forceinline__ half2 __tohalf2(uint32_t value)
{
    const auto [x, y] = std::bit_cast<std::array<half, 2>>(value);
    return half2(x, y);
}

static __device__ __forceinline__ float2 __half22float2(uint32_t value)
{
    return __half22float2(__tohalf2(value));
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float)e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    }
    else {
        bits = (uint32_t)x << 23;
    }

    return std::bit_cast<float>(bits);
#endif // CUDART_VERSION >= 12050
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#ifdef GGML_USE_HIP
    return 16;
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // GGML_USE_HIP
}

static __device__ __forceinline__ void ggml_cuda_mad(half2& acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v * u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void* __restrict__ dst, const void* __restrict__ src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char*)dst)[i] = ((const char*)src)[i];
        }
        else if constexpr (nb_per_cpy == 2) {
            ((short*)dst)[i] = ((const short*)src)[i];
        }
        else if constexpr (nb_per_cpy == 4) {
            ((int*)dst)[i] = ((const int*)src)[i];
        }
        else if constexpr (nb_per_cpy == 8) {
            ((int2*)dst)[i] = ((const int2*)src)[i];
        }
        else if constexpr (nb_per_cpy == 16) {
            ((int4*)dst)[i] = ((const int4*)src)[i];
        }
        else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ void ggml_cuda_mad(float& acc, const float v, const float u) {
    acc += v * u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float& acc, const float2 v, const float2 u) {
    acc += v.x * u.x;
    acc += v.y * u.y;
}

static __device__ __forceinline__ void ggml_cuda_mad(float& acc, const half2 v, const half2 u) {
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v * u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2)  || defined(RDNA3) || defined(RDNA4) || defined(GCN5) || defined(CDNA))
}
