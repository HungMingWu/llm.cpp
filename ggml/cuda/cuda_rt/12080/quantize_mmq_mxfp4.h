#pragma once

#include "common.cuh"

static __device__ __forceinline__
void quantize_mmq_mxfp4_helper(const int lane_in_group, const float xi, const float inv_s, const int base, char2* dst)
{
    const float scaled_val = xi * inv_s;

    const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
    const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
    const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
    const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

    if (lane_in_group == 0) {
        __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

        *dst = *(char2*)&fp4_packed;
    }
}