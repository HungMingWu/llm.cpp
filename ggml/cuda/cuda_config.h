#pragma once
#include "config.h"
#include "vendors/constant.h"

static constexpr bool use_cuda_graph_v = ggml_cuda_use_graphs_v | ggml_hip_graphs_v;

#ifdef GGML_CUDA_FORCE_CUBLAS
constexpr bool force_enable_cuda_blas_v = true;
#else
constexpr bool force_enable_cuda_blas_v = false;
#endif

#ifdef GGML_CUDA_FORCE_MMQ
constexpr bool force_enable_cuda_mmq_v = true;
#else
constexpr bool force_enable_cuda_mmq_v = false;
#endif

#ifdef GGML_CUDA_USE_CUB
constexpr bool enable_cuda_cub_v = true;
#else
constexpr bool enable_cuda_cub_v = false;
#endif

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
static constexpr bool new_mma_available_v = true;
#else
static constexpr bool new_mma_available_v = false;
#endif 

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
static constexpr bool v_dot2_f32_f16_available_v = true;
#define V_DOT2_F32_F16_AVAILABLE // Specical case, may remove later
#else
static constexpr bool v_dot2_f32_f16_available_v = false;
#endif

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
static constexpr bool fp16_available_v = true;
static constexpr bool fast_fp16_available_v = (__CUDA_ARCH__ != 610);
#else
static constexpr bool fp16_available_v = false;
static constexpr bool fast_fp16_available_v = false;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
static constexpr bool amd_mfma_available_v = true;
#define AMD_MFMA_AVAILABLE
#else
static constexpr bool amd_mfma_available_v = false;
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
static constexpr bool amd_wmma_available_v = true;
#define AMD_WMMA_AVAILABLE
#else
static constexpr bool amd_wmma_available_v = false;
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
static constexpr bool volta_mma_available_v = true;
#define VOLTA_MMA_AVAILABLE
#else
static constexpr bool volta_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
static constexpr bool turing_mma_available_v = true;
#define TURING_MMA_AVAILABLE
#else
static constexpr bool turing_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
static constexpr bool ampere_mma_available_v = true;
#define AMPERE_MMA_AVAILABLE
#else
static constexpr bool ampere_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE