#pragma once
#include "mma.cuh"

using namespace ggml_cuda_mma;

using tile_A = tile<16, 8, half2>;
using tile_B = tile< 8, 8, half2>;
using tile_B_16 = tile<16, 8, half2>;
using tile_C_KQ = tile<16, 8, float>;
using tile_C_KQ_16 = tile<16, 16, float>;
using tile_C_VKQ = tile<16, 4, half2>;
using tile_C_VKQ_16 = tile<16, 8, half2>;