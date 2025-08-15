module;
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <utility>
#include "block.h"

module ggml:cpu.vec_dot;
import :ds;
import :types;

using ggml_float = double;

template <typename T>
float ggml_vec_dot(int n, const T* x, const T* y, int nrc)
{
    assert(nrc == 1);
    // Waiting for C++26 SIMD
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += ggml_float(toFloat32(x[i])) * ggml_float(toFloat32(y[i]));
    }
    return sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q4_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q8_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q4_1* x, size_t bx, const block_q8_1* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_mxfp4* x, size_t bx, const block_q8_0* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q5_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q5_1* x, size_t bx, const block_q8_1* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q2_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q3_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q4_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q5_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_q6_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq2_xxs* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq2_xs* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq2_s* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq3_xxs* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq3_s* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq1_s* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq1_m* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq4_nl* x, size_t bx, const block_q8_0* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_iq4_xs* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_tq1_0* x, size_t bx, const block_q8_K* y, size_t by, int nrc);
void ggml_vec_dot(int n, float* s, size_t bs, const block_tq2_0* x, size_t bx, const block_q8_K* y, size_t by, int nrc);

template <typename src_t, typename dst_t>
void ggml_vec_dot(int n, float* s, size_t bs, const src_t* x, size_t bx, const dst_t* y, size_t by, int nrc)
{
    if constexpr (std::is_same_v<src_t, ggml_fp16_t> ||
        std::is_same_v<src_t, ggml_fp32_t> ||
        std::is_same_v<src_t, ggml_bf16_t>) {
        static_assert(std::is_same_v<src_t, dst_t>);
        *s = ggml_vec_dot<src_t>(n, x, y, nrc);
    }
    else {
		ggml_vec_dot(n, s, bs, x, bx, y, by, nrc);
    }
}