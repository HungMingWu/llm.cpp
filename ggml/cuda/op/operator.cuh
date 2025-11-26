#pragma once
#include "cuda_func.h"

static __forceinline__ __device__ float4 operator+(const float4& lhs, const float4& rhs)
{
	return float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

static __forceinline__ __device__ float4 operator*(const float4& lhs, const float4& rhs)
{
	return float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

static __forceinline__ __device__ float4 operator*(const float4& v, float scale)
{
	return float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
}

static __forceinline__ __device__ float dot_product(const float4& lhs, const float4& rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}
