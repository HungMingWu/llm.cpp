#pragma once
#include "common.cuh"

// Fallback: manual FP4 conversion using LUT
static __device__ __forceinline__
void quantize_mmq_mxfp4_helper(const int lane_in_group, const float xi, const float inv_s, const int base, char2* dst)
{
	const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

	const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base, WARP_SIZE);
	const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1, WARP_SIZE);
	const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
	const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

	if (lane_in_group == 0)
	{
		char2 q;
		q.x = (q_hi_0 << 4) | q_lo_0;
		q.y = (q_hi_1 << 4) | q_lo_1;
		*dst = q;
	}
}