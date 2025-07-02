#pragma once
#include <stdio.h>
#include <bit>

#include "../../internal_ds.h"
#include "../common.h"
#include "block.h"

[[noreturn]]
static __device__ void no_device_code(
    const char* file_name, const int line, const char* function_name, const int arch, const char* arch_list)
{
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
        file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
        file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    __trap();

    //GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

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

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template<bool norm>
static __global__ void reduce_rows_f32(const float* x, float* dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}

// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI4_0 (block_q4_0::block_size / (4 * QR4_0))
#define QR4_0 2

#define QI4_1 (block_q4_1::block_size / (4 * QR4_1))
#define QR4_1 2

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
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(RDNA2)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx1010__) || defined(__gfx900__)
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

#else // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    const int8_t* a8 = (const int8_t*)&a;
    const int8_t* b8 = (const int8_t*)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A

#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
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