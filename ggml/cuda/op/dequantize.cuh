#pragma once

static __device__ __forceinline__ void dequantize(const block_q4_0* x, const int iqs, float2& v) {
    const float d = std::bit_cast<half>(x->d);

    const int vui = x->qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize(const block_q4_1* x, const int iqs, float2& v) {
    const auto dm = std::bit_cast<std::array<half, 2>>(x->dm);
    const float d = dm[0];
    const float m = dm[1];

    const int vui = x->qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
}

static __device__ __forceinline__ void dequantize(const block_q5_0* x, const int iqs, float2& v) {
    const float d = std::bit_cast<half>(x->d);

    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x->qs[iqs] & 0xf) | xh_0);
    v.y = ((x->qs[iqs] >> 4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize(const block_q5_1* x, const int iqs, float2& v) {
    const auto dm = std::bit_cast<std::array<half, 2>>(x->dm);
    const float d = dm[0];
    const float m = dm[1];

    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x->qs[iqs] & 0xf) | xh_0);
    v.y = ((x->qs[iqs] >> 4) | xh_1);

    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
}

static __device__ __forceinline__ void dequantize(const block_q8_0* x,  const int iqs, float2& v) {
    const float d = std::bit_cast<half>(x->d);

    v.x = x->qs[iqs + 0];
    v.y = x->qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}