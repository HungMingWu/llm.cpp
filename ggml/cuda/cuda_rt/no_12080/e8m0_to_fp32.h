#pragma once
#include "common.h"
#include <bit>
static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    }
    else {
        bits = (uint32_t)x << 23;
    }

    return std::bit_cast<float>(bits);
}