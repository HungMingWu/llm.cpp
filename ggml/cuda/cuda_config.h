#pragma once
#include "config.h"
#include "vendors/constant.h"

static constexpr bool use_cuda_graph_v = ggml_cuda_use_graphs_v | ggml_hip_graphs_v;

// GGML_USE_HIP comes from CMake definition
#ifdef GGML_USE_HIP
static constexpr bool use_hip_v = true;
#else
static constexpr bool use_hip_v = false;
#endif

// GGML_USE_MUSA comes from CMake definition
#ifdef GGML_USE_MUSA
static constexpr bool use_musa_v = true;
#else
static constexpr bool use_musa_v = false;
#endif

#if defined(GGML_CUDA_FORCE_CUBLAS)
static constexpr bool force_enable_cuda_blas_v = true;
#else
static constexpr bool force_enable_cuda_blas_v = false;
#endif

#if defined(GGML_CUDA_FORCE_MMQ)
static constexpr bool ggml_cuda_force_mmq_v = true;
#else
static constexpr bool ggml_cuda_force_mmq_v = false;
#endif

#if defined(GGML_CUDA_NO_PEER_COPY)
static constexpr bool ggml_cuda_no_peer_copy_v = true;
#else
static constexpr bool ggml_cuda_no_peer_copy_v = false;
#endif

static constexpr int ggml_cuda_peer_max_batch_size_v = GGML_CUDA_PEER_MAX_BATCH_SIZE;

#if defined(GGML_CUDA_FA_ALL_QUANTS)
static constexpr bool ggml_cuda_fa_all_quants_v = true;
#else
static constexpr bool ggml_cuda_fa_all_quants_v = false;
#endif

#ifdef GGML_CUDA_USE_CUB
static constexpr bool enable_cuda_cub_v = true;
#else
static constexpr bool enable_cuda_cub_v = false;
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
#else
static constexpr bool amd_mfma_available_v = false;
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
static constexpr bool amd_wmma_available_v = true;
#else
static constexpr bool amd_wmma_available_v = false;
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
static constexpr bool volta_mma_available_v = true;
#else
static constexpr bool volta_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
static constexpr bool turing_mma_available_v = true;
#else
static constexpr bool turing_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
static constexpr bool ampere_mma_available_v = true;
#else
static constexpr bool ampere_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
static constexpr bool blackwell_mma_available_v = true;
#else
static constexpr bool blackwell_mma_available_v = false;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL

#if !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)
static constexpr bool flash_attn_available_v = true;
#define FLASH_ATTN_AVAILABLE
#else
static constexpr bool flash_attn_available_v = false;
#endif // !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)

#if (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))
static constexpr bool ggml_use_vmm_v = true;
#else
static constexpr bool ggml_use_vmm_v = false;
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))