#pragma once
#include "vendors/cuda.h"

#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_ADA_LOVELACE    890
// While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core instructions available to 1200 family, see
// https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
#define GGML_CUDA_CC_BLACKWELL       1200
#define GGML_CUDA_CC_DGX_SPARK       1210
#define GGML_CUDA_CC_RUBIN           1300
#define GGML_CUDA_CC_OFFSET_AMD      0x1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)
