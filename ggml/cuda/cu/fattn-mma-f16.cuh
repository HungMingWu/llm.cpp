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

// Config options for specific head sizes.
// Should not affect results, only speed/register pressure/shared memory use.
//
// nbatch_fa:      number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
// nwarps_max:     maximum number of warps per CUDA block, up to 8 warps in total can run per SM (given enough shared memory).
// Q_in_reg:       whether the Q values should be kept permanently in registers.
// nstages_target: targeted number of pipeline stages for cp_async (if available), 0 means synchronous data loading.
// nbatch_K2:      number of K half2 values in direction of DKQ to load in parallel.
// nbatch_V2:      number of V half2 values in direction of DV to load in parallel.
// nbatch_combine: number of VKQ half2 values in direction of DV to combine in parallel.

template <int DKQ, int DV>
struct fattn_mma_f16_config;

template <>
struct fattn_mma_f16_config< 64, 64> {
    static constexpr int  nbatch_fa = 64;
    static constexpr int  nwarps_max = 4;
    static constexpr bool Q_in_reg = true;
    static constexpr int  nstages_target = 2;
    static constexpr int  nbatch_K2 = 32;
    static constexpr int  nbatch_V2 = 32;
    static constexpr int  nbatch_combine = 32;
};

template <>
struct fattn_mma_f16_config< 80, 80> {
    static constexpr int  nbatch_fa = 64;
    static constexpr int  nwarps_max = 4;
    static constexpr bool Q_in_reg = true;
    static constexpr int  nstages_target = 2;
    static constexpr int  nbatch_K2 = 40;
    static constexpr int  nbatch_V2 = 40;
    static constexpr int  nbatch_combine = 40;
};

template <>
struct fattn_mma_f16_config< 96, 96> {
    static constexpr int  nbatch_fa = 64;
    static constexpr int  nwarps_max = 4;
    static constexpr bool Q_in_reg = true;
    static constexpr int  nstages_target = 2;
    static constexpr int  nbatch_K2 = 48;
    static constexpr int  nbatch_V2 = 48;
    static constexpr int  nbatch_combine = 48;
};

template <>
struct fattn_mma_f16_config<112, 112> {
    static constexpr int  nbatch_fa = 64;
    static constexpr int  nwarps_max = 4;
    static constexpr bool Q_in_reg = true;
    static constexpr int  nstages_target = 2;
    static constexpr int  nbatch_K2 = 56;
    static constexpr int  nbatch_V2 = 56;
    static constexpr int  nbatch_combine = 56;
};

template <>
struct fattn_mma_f16_config<128, 128> {
    static constexpr int  nbatch_fa = 64;
    static constexpr int  nwarps_max = 4;
    static constexpr bool Q_in_reg = true;
    static constexpr int  nstages_target = 2;
    static constexpr int  nbatch_K2 = 64;
    static constexpr int  nbatch_V2 = 64;
    static constexpr int  nbatch_combine = 64;
};

template <>
struct fattn_mma_f16_config<256, 256> {
    static constexpr int  nbatch_fa = 32;
    static constexpr int  nwarps_max = 4;
    static constexpr bool Q_in_reg = true;
    static constexpr int  nstages_target = 2;
    static constexpr int  nbatch_K2 = 128;
    static constexpr int  nbatch_V2 = 128;
    static constexpr int  nbatch_combine = 128;
};

template <>
struct fattn_mma_f16_config<576, 512> {
    static constexpr int  nbatch_fa = 32;
    static constexpr int  nwarps_max = 8;
    static constexpr bool Q_in_reg = false;
    static constexpr int  nstages_target = 1;
    static constexpr int  nbatch_K2 = 160;
    static constexpr int  nbatch_V2 = 128;
    static constexpr int  nbatch_combine = 128;
};
