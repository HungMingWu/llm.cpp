#pragma once
#include <bit>
#include "common.h"

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float)e;
}