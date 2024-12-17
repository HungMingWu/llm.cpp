module;
#include <assert.h>
#include <array>
#include <bit>
#include "block.h"

module ggml:quants;
import :types;

void dequantize_row_q4_0(const block_q4_0* x, float* y, int64_t k) {
    static const int qk = QK4_0;

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

void dequantize_row_q4_1(const block_q4_1* x, float* y, int64_t k) {
    static const int qk = QK4_1;

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

void dequantize_row_q5_0(const block_q5_0* x, float* y, int64_t k) {
    static const int qk = QK5_0;

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

void dequantize_row_q5_1(const block_q5_1* x, float* y, int64_t k) {
    static const int qk = QK5_1;

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

void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int j = 0; j < qk; ++j) {
            y[i * qk + j] = x[i].qs[j] * d;
        }
    }
}