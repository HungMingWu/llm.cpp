#pragma once

static __device__ __forceinline__ float2 operator+(float2 lhs, float2 rhs) {
    return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

static __device__ __forceinline__ float2 operator*(float2 lhs, float2 rhs) {
    return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
}