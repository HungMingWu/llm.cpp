#pragma once

static __device__ __forceinline__ void dequantize(const block_q4_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = std::bit_cast<half>(x[ib].d);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, { 8.0f, 8.0f });
    v = __hmul2(v, { d, d });
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q4_1* x, const int64_t ib, const int iqs, dfloat2& v) {
    const auto dm = std::bit_cast<std::array<half, 2>>(x[ib].dm);
    const dfloat d = dm[0];
    const dfloat m = dm[1];

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
    v = __hadd2(v, { m, m });
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q5_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = std::bit_cast<half>(x[ib].d);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, { 16.0f, 16.0f });
    v = __hmul2(v, { d, d });
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q5_1* x, const int64_t ib, const int iqs, dfloat2& v) {
    const auto dm = std::bit_cast<std::array<half, 2>>(x[ib].dm);
    const dfloat d = dm[0];
    const dfloat m = dm[1];

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
    v = __hadd2(v, { m, m });
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}