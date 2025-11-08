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