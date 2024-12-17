module;
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <bit>
#include "block.h"

module ggml:quants;
import :types;

void dequantize_row(const block_q4_0* x, float* y, int64_t k)
{
    static const int qk = block_q4_0::block_size;

    assert(k% qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;

            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

void dequantize_row_q4_0(const block_q4_0* x, float* y, int64_t k) {
    static const int qk = block_q4_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;

            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

void dequantize_row(const block_q4_1* x, float* y, int64_t k) {
    static const int qk = block_q4_1::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float m = toFloat32(dm[1]);

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >> 4);

            y[i * qk + j + 0] = x0 * d + m;
            y[i * qk + j + qk / 2] = x1 * d + m;
        }
    }
}

void dequantize_row_q4_1(const block_q4_1* x, float* y, int64_t k) {
    static const int qk = block_q4_1::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float m = toFloat32(dm[1]);

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >> 4);

            y[i * qk + j + 0] = x0 * d + m;
            y[i * qk + j + qk / 2] = x1 * d + m;
        }
    }
}

void dequantize_row(const block_q5_0* x, float* y, int64_t k) {
    static const int qk = block_q5_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

void dequantize_row_q5_0(const block_q5_0* x, float* y, int64_t k) {
    static const int qk = block_q5_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

void dequantize_row(const block_q5_1* x, float* y, int64_t k) {
    static const int qk = block_q5_1::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float m = toFloat32(dm[1]);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >> 4) | xh_1;

            y[i * qk + j + 0] = x0 * d + m;
            y[i * qk + j + qk / 2] = x1 * d + m;
        }
    }
}

void dequantize_row_q5_1(const block_q5_1* x, float* y, int64_t k) {
    static const int qk = block_q5_1::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float m = toFloat32(dm[1]);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >> 4) | xh_1;

            y[i * qk + j + 0] = x0 * d + m;
            y[i * qk + j + qk / 2] = x1 * d + m;
        }
    }
}

void dequantize_row(const block_q8_0* x, float* y, int64_t k) {
    static const int qk = block_q8_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int j = 0; j < qk; ++j) {
            y[i * qk + j] = x[i].qs[j] * d;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k) {
    static const int qk = block_q8_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int j = 0; j < qk; ++j) {
            y[i * qk + j] = x[i].qs[j] * d;
        }
    }
}

void quantize_row_q4_0_ref(const float* x, block_q4_0* y, int64_t k);
void quantize_row_q4_1_ref(const float* x, block_q4_1* y, int64_t k);
void quantize_row_q5_0_ref(const float* x, block_q5_0* y, int64_t k);
void quantize_row_q5_1_ref(const float* x, block_q5_1* y, int64_t k);
void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int64_t k);
void quantize_row_q2_K_ref(const float* x, block_q2_K* y, int64_t k);
void quantize_row_q3_K_ref(const float* x, block_q3_K* y, int64_t k);
void quantize_row_q4_K_ref(const float* x, block_q4_K* y, int64_t k);
void quantize_row_q5_K_ref(const float* x, block_q5_K* y, int64_t k);
void quantize_row_q6_K_ref(const float* x, block_q6_K* y, int64_t k);
void quantize_row_tq1_0_ref(const float* x, block_tq1_0* y, int64_t k);
void quantize_row_tq2_0_ref(const float* x, block_tq2_0* y, int64_t k);

template <typename T>
constexpr bool is_quant_type_v = false;

template <>
constexpr bool is_quant_type_v<block_q4_0> = true;

template <>
constexpr bool is_quant_type_v<block_q4_1> = true;

template <>
constexpr bool is_quant_type_v<block_q5_0> = true;

template <>
constexpr bool is_quant_type_v<block_q5_1> = true;

template <>
constexpr bool is_quant_type_v<block_q8_0> = true;

template <>
constexpr bool is_quant_type_v<block_iq4_nl> = true;

void quantize_row_q4_0(const float* x, void* y, int64_t k) {
    quantize_row_q4_0_ref(x, (block_q4_0*)y, k);
}

void quantize_row(const float* x, block_q4_0* y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1(const float* x, void* y, int64_t k) {
    quantize_row_q4_1_ref(x, (block_q4_1*)y, k);
}

void quantize_row(const float* x, block_q4_1* y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0(const float* x, void* y, int64_t k) {
    quantize_row_q5_0_ref(x, (block_q5_0*)y, k);
}

void quantize_row(const float* x, block_q5_0* y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1(const float* x, void* y, int64_t k) {
    quantize_row_q5_1_ref(x, (block_q5_1*)y, k);
}

void quantize_row(const float* x, block_q5_1* y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q6_K(const float* x, void* y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_q6_K_ref(x, static_cast<block_q6_K*>(y), k);
}

void quantize_row_q8_0(const float* x, void* vy, int64_t k) {
#if 0
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0* y = vy;
#endif
#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv[8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = wasm_v128_load(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = wasm_f32x4_max(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = wasm_f32x4_max(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = wasm_f32x4_max(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
            wasm_f32x4_extract_lane(amaxv[0], 1)),
            MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const v128_t v = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4 * j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4 * j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4 * j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4 * j + 3] = wasm_i32x4_extract_lane(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Apply the multiplier
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32(i0, i1);	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32(i2, i3);	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
        // Convert int16 to int8
        i0 = _mm256_packs_epi16(i0, i2);	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256((__m256i*)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128(i0);
        __m128i ni1 = _mm256_extractf128_si256(i0, 1);
        __m128i ni2 = _mm256_castsi256_si128(i1);
        __m128i ni3 = _mm256_extractf128_si256(i1, 1);
        __m128i ni4 = _mm256_castsi256_si128(i2);
        __m128i ni5 = _mm256_extractf128_si256(i2, 1);
        __m128i ni6 = _mm256_castsi256_si128(i3);
        __m128i ni7 = _mm256_extractf128_si256(i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32(ni0, ni1);
        ni2 = _mm_packs_epi32(ni2, ni3);
        ni4 = _mm_packs_epi32(ni4, ni5);
        ni6 = _mm_packs_epi32(ni6, ni7);
        // Convert int16 to int8
        ni0 = _mm_packs_epi16(ni0, ni2);
        ni4 = _mm_packs_epi16(ni4, ni6);

        _mm_storeu_si128((__m128i*)(y[i].qs + 0), ni0);
        _mm_storeu_si128((__m128i*)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_0);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x = __riscv_vle32_v_f32m4(x + i * QK8_0, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs, vs, vl);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv[8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vec_max(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vec_max(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = vec_max(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
            vec_extract(amaxv[0], 1)),
            MAX(vec_extract(amaxv[0], 2),
                vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const vector float v = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])), 0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        __m256 v0 = (__m256)__lasx_xvld(x, 0);
        __m256 v1 = (__m256)__lasx_xvld(x, 32);
        __m256 v2 = (__m256)__lasx_xvld(x, 64);
        __m256 v3 = (__m256)__lasx_xvld(x, 96);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s(-0.0f);
        __m256 max_abs = (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v0);
        max_abs = __lasx_xvfmax_s(max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v1));
        max_abs = __lasx_xvfmax_s(max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v2));
        max_abs = __lasx_xvfmax_s(max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v3));

        __m128 max4 = __lsx_vfmax_s(lasx_extractf128(max_abs, 1), lasx_extractf128(max_abs, 0));
        max4 = __lsx_vfmax_s(max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4));
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s(max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w(max4, 1), 0));
        const float max_scalar = ((v4f32)max4)[0];

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = (__m256)__lasx_xvreplfr2vr_s(id);

        // Apply the multiplier
        v0 = __lasx_xvfmul_s(v0, mul);
        v1 = __lasx_xvfmul_s(v1, mul);
        v2 = __lasx_xvfmul_s(v2, mul);
        v3 = __lasx_xvfmul_s(v3, mul);

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s(v0);
        __m256i i1 = __lasx_xvftintrne_w_s(v1);
        __m256i i2 = __lasx_xvftintrne_w_s(v2);
        __m256i i3 = __lasx_xvftintrne_w_s(v3);

        __m128i ni0 = lasx_extracti128(i0, 0);
        __m128i ni1 = lasx_extracti128(i0, 1);
        __m128i ni2 = lasx_extracti128(i1, 0);
        __m128i ni3 = lasx_extracti128(i1, 1);
        __m128i ni4 = lasx_extracti128(i2, 0);
        __m128i ni5 = lasx_extracti128(i2, 1);
        __m128i ni6 = lasx_extracti128(i3, 0);
        __m128i ni7 = lasx_extracti128(i3, 1);

        // Convert int32 to int16
        ni0 = lsx_packs_w(ni0, ni1);
        ni2 = lsx_packs_w(ni2, ni3);
        ni4 = lsx_packs_w(ni4, ni5);
        ni6 = lsx_packs_w(ni6, ni7);
        // Convert int16 to int8
        ni0 = lsx_packs_h(ni0, ni2);
        ni4 = lsx_packs_h(ni4, ni6);

        __lsx_vst(ni0, (__m128i*)(y[i].qs + 0), 0);
        __lsx_vst(ni4, (__m128i*)(y[i].qs + 16), 0);

    }
#else
    // scalar
    quantize_row_q8_0_ref(x, (block_q8_0*)vy, k);
#endif
}
void quantize_row(const float* x, block_q8_0* y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

// ============================ 4-bit non-linear quants
static constexpr double GROUP_MAX_EPS = 1e-15f;

static inline int best_index_int8(int n, const int8_t* val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n - 1]) return n - 1;
    int ml = 0, mu = n - 1;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float* x,
    uint16_t* dh, uint8_t* q4, uint16_t* scales_h, uint8_t* scales_l,
    float* scales, float* weight, uint8_t* L,
    const int8_t* values,
    const float* quant_weights,
    const int ntry) {

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j] * x[j];
    sigma2 *= 2.f / super_block_size;

    memset(q4, 0, super_block_size / 2);
    dh[0] = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size / block_size; ++ib) {
        const float* xb = x + ib * block_size;
        uint8_t* Lb = L + ib * block_size;
        if (quant_weights) {
            const float* qw = quant_weights + ib * block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
        }
        else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j] * xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max / values[0] : max / values[0];
        float id = 1 / d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id * xb[j];
            int l = best_index_int8(16, values, al);
            Lb[j] = l;
            float q = values[l];
            float w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumqx / sumq2;
        float best = d * sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0]) / max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id * xb[j];
                int l = best_index_int8(16, values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                d = sumqx / sumq2; best = d * sumqx;
            }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size / block_size > 1) {
        int nb = super_block_size / block_size;
        memset(scales_h, 0, ((nb + 7) / 8) * sizeof(uint16_t));
        float d = -max_scale / 32;
        dh[0] = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d));
        float id = d ? 1 / d : 0.f;
        for (int ib = 0; ib < super_block_size / block_size; ++ib) {
            int l = nearest_int(id * scales[ib]);
            l = std::max(-32, std::min(31, l));
            float dl = d * l;
            float idl = dl ? 1 / dl : 0.f;
            uint8_t* Lb = L + ib * block_size;
            const float* xb = x + ib * block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8(16, values, idl * xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >> 4;
            if (ib % 2 == 0) scales_l[ib / 2] = l_l;
            else scales_l[ib / 2] |= (l_l << 4);
            scales_h[ib / 8] |= (l_h << 2 * (ib % 8));
        }
    }
    else {
        dh[0] = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(scales[0]));
        if (ntry > 0) {
            float id = scales[0] ? 1 / scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_int8(16, values, id * x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size / 32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16 * i + j] = L[32 * i + j] | (L[32 * i + 16 + j] << 4);
        }
    }
}

static const int8_t kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

void quantize_row_iq4_nl_ref(const float* x, block_iq4_nl* y, int64_t k) {
    assert(k % block_iq4_nl::block_size == 0);
    int64_t nblock = k / block_iq4_nl::block_size;
    uint8_t L[block_iq4_nl::block_size];
    float weight[block_iq4_nl::block_size];
    uint16_t unused_h;
    uint8_t* unused_l = nullptr;
    float scale;
    block_iq4_nl* iq4 = y;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        quantize_row_iq4_nl_impl(block_iq4_nl::block_size, 32, x + block_iq4_nl::block_size * ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
            &scale, weight, L, kvalues_iq4nl, nullptr, -1);
    }
}

void quantize_row_iq4_nl(const float* x, void* y, int64_t k) {
    //assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, (block_iq4_nl*)y, k);
}

void quantize_row(const float* x, block_iq4_nl* y, int64_t k) {
    //assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_q8_1_ref(const float* x, block_q8_1* y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i * QK8_1 + j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        auto ds = std::bit_cast<std::array<ggml_fp16_t, 2>>(y[i].ds);
        ds[0] = fromFloat32<ggml_fp16_t>(d);

        int sum = 0;

        for (int j = 0; j < QK8_1 / 2; ++j) {
            const float v0 = x[i * QK8_1 + j] * id;
            const float v1 = x[i * QK8_1 + QK8_1 / 2 + j] * id;

            y[i].qs[j] = roundf(v0);
            y[i].qs[QK8_1 / 2 + j] = roundf(v1);

            sum += y[i].qs[j];
            sum += y[i].qs[QK8_1 / 2 + j];
        }

        ds[1] = fromFloat32<ggml_fp16_t>(sum * d);
        y[i].ds = std::bit_cast<uint32_t>(ds);
    }
}

void quantize_row_q8_1(const float* x, void* vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1* y = (block_q8_1 *)vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);

            accv = vaddq_s32(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv[8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = wasm_v128_load(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = wasm_f32x4_max(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = wasm_f32x4_max(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = wasm_f32x4_max(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
            wasm_f32x4_extract_lane(amaxv[0], 1)),
            MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        v128_t accv = wasm_i32x4_splat(0);

        for (int j = 0; j < 8; j++) {
            const v128_t v = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4 * j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4 * j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4 * j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4 * j + 3] = wasm_i32x4_extract_lane(vi, 3);

            accv = wasm_i32x4_add(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(
            d * (wasm_i32x4_extract_lane(accv, 0) +
                wasm_i32x4_extract_lane(accv, 1) +
                wasm_i32x4_extract_lane(accv, 2) +
                wasm_i32x4_extract_lane(accv, 3)));
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float max_scalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Apply the multiplier
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
        // Compute the sum of the quants and set y[i].s
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32(i0, i1);	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32(i2, i3);	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
        // Convert int16 to int8
        i0 = _mm256_packs_epi16(i0, i2);	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256((__m256i*)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128(i0);
        __m128i ni1 = _mm256_extractf128_si256(i0, 1);
        __m128i ni2 = _mm256_castsi256_si128(i1);
        __m128i ni3 = _mm256_extractf128_si256(i1, 1);
        __m128i ni4 = _mm256_castsi256_si128(i2);
        __m128i ni5 = _mm256_extractf128_si256(i2, 1);
        __m128i ni6 = _mm256_castsi256_si128(i3);
        __m128i ni7 = _mm256_extractf128_si256(i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
        const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(_mm_add_epi32(s0, s1)));

        // Convert int32 to int16
        ni0 = _mm_packs_epi32(ni0, ni1);
        ni2 = _mm_packs_epi32(ni2, ni3);
        ni4 = _mm_packs_epi32(ni4, ni5);
        ni6 = _mm_packs_epi32(ni6, ni7);
        // Convert int16 to int8
        ni0 = _mm_packs_epi16(ni0, ni2);
        ni4 = _mm_packs_epi16(ni4, ni6);

        _mm_storeu_si128((__m128i*)(y[i].qs + 0), ni0);
        _mm_storeu_si128((__m128i*)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_1);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x = __riscv_vle32_v_f32m4(x + i * QK8_1, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs, vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m1_i16m1(vs, tmp2, vl);

        // set y[i].s
        int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = GGML_FP32_TO_FP16(sum * d);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv[8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vec_max(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vec_max(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = vec_max(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
            vec_extract(amaxv[0], 1)),
            MAX(vec_extract(amaxv[0], 2),
                vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        vector int accv = vec_splats(0);

        for (int j = 0; j < 8; j++) {
            const vector float v = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_add(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])), 0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_add(accv, vec_sld(accv, accv, 4));
        accv = vec_add(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_FP32_TO_FP16(d * vec_extract(accv, 0));
    }

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        __m256 v0 = (__m256)__lasx_xvld(x, 0);
        __m256 v1 = (__m256)__lasx_xvld(x, 32);
        __m256 v2 = (__m256)__lasx_xvld(x, 64);
        __m256 v3 = (__m256)__lasx_xvld(x, 96);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s(-0.0f);
        __m256 max_abs = (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v0);
        max_abs = __lasx_xvfmax_s(max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v1));
        max_abs = __lasx_xvfmax_s(max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v2));
        max_abs = __lasx_xvfmax_s(max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v3));

        __m128 max4 = __lsx_vfmax_s(lasx_extractf128(max_abs, 1), lasx_extractf128(max_abs, 0));
        max4 = __lsx_vfmax_s(max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4));
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s(max4, (__m128)__lsx_vextrins_w((__m128i)tmp, (__m128i)max4, 0x10));
        const float max_scalar = ((v4f32)max4)[0];

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = __lasx_xvreplfr2vr_s(id);

        // Apply the multiplier
        v0 = __lasx_xvfmul_s(v0, mul);
        v1 = __lasx_xvfmul_s(v1, mul);
        v2 = __lasx_xvfmul_s(v2, mul);
        v3 = __lasx_xvfmul_s(v3, mul);

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s(v0);
        __m256i i1 = __lasx_xvftintrne_w_s(v1);
        __m256i i2 = __lasx_xvftintrne_w_s(v2);
        __m256i i3 = __lasx_xvftintrne_w_s(v3);

        __m128i ni0 = lasx_extracti128(i0, 0);
        __m128i ni1 = lasx_extracti128(i0, 1);
        __m128i ni2 = lasx_extracti128(i1, 0);
        __m128i ni3 = lasx_extracti128(i1, 1);
        __m128i ni4 = lasx_extracti128(i2, 0);
        __m128i ni5 = lasx_extracti128(i2, 1);
        __m128i ni6 = lasx_extracti128(i3, 0);
        __m128i ni7 = lasx_extracti128(i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = __lsx_vadd_w(__lsx_vadd_w(ni0, ni1), __lsx_vadd_w(ni2, ni3));
        const __m128i s1 = __lsx_vadd_w(__lsx_vadd_w(ni4, ni5), __lsx_vadd_w(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(__lsx_vadd_w(s0, s1)));

        // Convert int32 to int16
        ni0 = lsx_packs_w(ni0, ni1);
        ni2 = lsx_packs_w(ni2, ni3);
        ni4 = lsx_packs_w(ni4, ni5);
        ni6 = lsx_packs_w(ni6, ni7);
        // Convert int16 to int8
        ni0 = lsx_packs_h(ni0, ni2);
        ni4 = lsx_packs_h(ni4, ni6);

        __lsx_vst(ni0, (__m128i*)(y[i].qs + 0), 0);
        __lsx_vst(ni4, (__m128i*)(y[i].qs + 16), 0);
    }
#else
    (void)(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = std::min(127, v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1 / iscale;
        x += QK_K;
    }
}

void quantize_row_q8_K(const float* x, void* y, int64_t k) {
    quantize_row_q8_K_ref(x, (block_q8_K *)y, k);
}

size_t quantize_q4_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q4_1(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q5_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q5_1(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q8_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q2_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q3_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q4_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q5_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q6_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_tq1_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*);
size_t quantize_tq2_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*);
size_t quantize_iq2_xxs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq2_xs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq2_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq3_xxs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq3_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq1_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq1_m(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq4_nl(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq4_xs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);

void dequantize_row_q2_K(const block_q2_K* x, float* y, int64_t k);
void dequantize_row_q3_K(const block_q3_K* x, float* y, int64_t k);
void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k);
void dequantize_row_q5_K(const block_q5_K* x, float* y, int64_t k);
void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k);
void dequantize_row_iq4_nl(const block_iq4_nl* x, float* y, int64_t k);
void dequantize_row_iq4_xs(const block_iq4_xs* x, float* y, int64_t k);