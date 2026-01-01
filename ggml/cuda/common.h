#pragma once
#include <array>
#include <stdlib.h>
#include "vendors/cuda.h"
#include "cuda_config.h"

#define GGML_ABORT(...)

// AMD
// GCN/CDNA, wave size is 64
#define GGML_CUDA_CC_GCN4       (GGML_CUDA_CC_OFFSET_AMD + 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA       (GGML_CUDA_CC_OFFSET_AMD + 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20     (GGML_CUDA_CC_OFFSET_AMD + 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_CDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_CUDA_CC_CDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x910)  // MI210, minimum acc register renameing
#define GGML_CUDA_CC_CDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300

// RDNA removes MFMA, dp4a, xnack, acc registers, wave size is 32
#define GGML_CUDA_CC_RDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x1010) // RX 5000
#define GGML_CUDA_CC_RDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x1030) // RX 6000, minimum for dp4a
#define GGML_CUDA_CC_RDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x1100) // RX 7000, minimum for WMMA
#define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.
#define GGML_CUDA_CC_RDNA4      (GGML_CUDA_CC_OFFSET_AMD + 0x1200) // RX 9000

#define GGML_CUDA_CC_IS_AMD(cc)     (cc >= GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_RDNA(cc)    (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc)   (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA2(cc)   (cc >= GGML_CUDA_CC_RDNA2 && cc < GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) (cc >= GGML_CUDA_CC_RDNA3 && cc < GGML_CUDA_CC_RDNA3_5)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
#define GGML_CUDA_CC_IS_RDNA4(cc)   (cc >= GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_GCN(cc)     (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_CDNA1)
#define GGML_CUDA_CC_IS_CDNA(cc)    (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_CDNA1(cc)   (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_CDNA2)
#define GGML_CUDA_CC_IS_CDNA2(cc)   (cc >= GGML_CUDA_CC_CDNA2 && cc < GGML_CUDA_CC_CDNA3)
#define GGML_CUDA_CC_IS_CDNA3(cc)   (cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_RDNA1)

// Moore Threads
#define MUSART_HMASK 40300 // MUSA rc4.3, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_QY1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x210) // MTT S80, MTT S3000
#define GGML_CUDA_CC_QY2 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x220) // MTT S4000
#define GGML_CUDA_CC_PH1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x310) // MTT S5000

#define GGML_CUDA_CC_IS_MTHREADS(cc) (cc >= GGML_CUDA_CC_OFFSET_MTHREADS && cc < GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_QY1(cc)      (cc >= GGML_CUDA_CC_QY1 && cc < GGML_CUDA_CC_QY2)
#define GGML_CUDA_CC_IS_QY2(cc)      (cc >= GGML_CUDA_CC_QY2 && cc < GGML_CUDA_CC_PH1)
#define GGML_CUDA_CC_IS_PH1(cc)      (cc >= GGML_CUDA_CC_PH1)

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
#define MUL_MAT_SRC1_COL_STRIDE 128
#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor co

static constexpr size_t GGML_CUDA_MAX_DEVICES = 16;
static constexpr size_t GGML_CUDA_MAX_STREAMS = 8;

static constexpr int WARP_SIZE = 32;

[[noreturn]]
void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg);

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        size_t  smpbo;              // max. shared memory per block (with opt-in)
        bool    integrated;         // Device is integrated as opposed to discrete
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;          // Number of threads in a dispatch
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info& ggml_cuda_info();
int ggml_backend_cuda_get_device_count();

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

#if CUDART_VERSION >= 12000 || defined(GGML_USE_MUSA)
static const char* cublas_get_error_str(const cublasStatus_t err) {
    return cublasGetStatusString(err);
}
#else
static const char* cublas_get_error_str(const cublasStatus_t err) {
    switch (err) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "unknown error";
    }
}
#endif // CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
static const char* cu_get_error_str(CUresult err) {
    const char* err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) {
    return false;
}

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
    if (cur == 0) {
        return -1;
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    }
    else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

#if (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || \
    (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low = 0x0000FFFF * (float(__low2half(a)) > float(__low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)

static constexpr bool fp16_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_PASCAL;
}

static constexpr bool fast_fp16_available(const int cc) {
    return GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_NVIDIA(cc) && fp16_available(cc) && ggml_cuda_highest_compiled_arch(cc) != 610);
}

bool amd_wmma_available(const int cc);
bool volta_mma_available(const int cc);
bool turing_mma_available(const int cc);
bool ampere_mma_available(const int cc);
bool blackwell_mma_available(const int cc);

constexpr int get_mmq_y_host(const int cc) {
    return cc >= GGML_CUDA_CC_OFFSET_AMD ? (cc == GGML_CUDA_CC_RDNA1 ? 64 : 128) : (cc >= GGML_CUDA_CC_VOLTA ? 128 : 64);
}

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;

    return powf(base, exph);
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}

static constexpr bool int8_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && cc >= GGML_CUDA_CC_TURING;
}

bool bf16_mma_hardware_available(const int cc);

// To be used for feature selection of external libraries, e.g. cuBLAS.
bool fast_fp16_hardware_available(const int cc);

void CUDA_SET_SHARED_MEMORY_LIMIT(const void* kernel, size_t nbytes);

bool amd_mfma_available(const int cc);

bool fp32_mma_hardware_available(const int cc);

// To be used for feature selection of external libraries, e.g. cuBLAS.
bool fp16_mma_hardware_available(const int cc);

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func& f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func& f, Args... args) const {
        f(0, args...);
    }
};
