module;
#include <assert.h>
#include <bit>
#include "block.h"

#define UNUSED(x) (void)(x)

module ggml;
import :types;

void ggml_vec_dot(int n, float* s, size_t bs, const block_q4_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc)
{
    const int qk = block_q8_0::block_size;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0* restrict vx0 = vx;
        const block_q4_0* restrict vx1 = (const block_q4_0*)((const uint8_t*)vx + bx);
        const block_q8_0* restrict vy0 = vy;
        const block_q8_0* restrict vy1 = (const block_q8_0*)((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0* restrict b_x0 = &vx0[i];
            const block_q4_0* restrict b_x1 = &vx1[i];
            const block_q8_0* restrict b_y0 = &vy0[i];
            const block_q8_0* restrict b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d) * GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d) * GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d) * GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d) * GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0, (vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt() * 8;

    // VLA Implementation using switch case
    switch (vector_length) {
    case 128:
    {
        // predicate for activating higher lanes for 4 float32 elements
        const svbool_t ph4 = svptrue_pat_b32(SV_VL4);

        for (; ib + 1 < nb; ib += 2) {
            const block_q4_0* restrict x0 = &x[ib + 0];
            const block_q4_0* restrict x1 = &x[ib + 1];
            const block_q8_0* restrict y0 = &y[ib + 0];
            const block_q8_0* restrict y1 = &y[ib + 1];

            // load x
            const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
            const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

            // 4-bit -> 8-bit
            const svint8_t qx0l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx0r, 0x0F));
            const svint8_t qx0h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx0r, 0x04));
            const svint8_t qx1l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx1r, 0x0F));
            const svint8_t qx1h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx1r, 0x04));

            // sub 8
            const svint8_t qx0ls = svsub_n_s8_x(svptrue_b8(), qx0h, 8);
            const svint8_t qx0hs = svsub_n_s8_x(svptrue_b8(), qx0l, 8);
            const svint8_t qx1ls = svsub_n_s8_x(svptrue_b8(), qx1h, 8);
            const svint8_t qx1hs = svsub_n_s8_x(svptrue_b8(), qx1l, 8);

            // load y
            const svint8_t qy0h = svld1_s8(svptrue_b8(), y0->qs);
            const svint8_t qy0l = svld1_s8(svptrue_b8(), y0->qs + 16);
            const svint8_t qy1h = svld1_s8(svptrue_b8(), y1->qs);
            const svint8_t qy1l = svld1_s8(svptrue_b8(), y1->qs + 16);

            // dot product
            sumv0 = svmla_n_f32_x(ph4, sumv0, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                svdot_s32(svdup_n_s32(0), qx0ls, qy0l),
                svdot_s32(svdup_n_s32(0), qx0hs, qy0h))), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(ph4, sumv1, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                svdot_s32(svdup_n_s32(0), qx1ls, qy1l),
                svdot_s32(svdup_n_s32(0), qx1hs, qy1h))), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
    } break;
    case 256:
    {
        // predicate for activating higher lanes for 16 int8 elements
        const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
        // predicate for activating lower lanes for  16 int8 elements
        const svbool_t pl16 = svnot_b_z(svptrue_b8(), ph16);

        for (; ib + 1 < nb; ib += 2) {
            const block_q4_0* restrict x0 = &x[ib + 0];
            const block_q4_0* restrict x1 = &x[ib + 1];
            const block_q8_0* restrict y0 = &y[ib + 0];
            const block_q8_0* restrict y1 = &y[ib + 1];

            // load x
            const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
            const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

            // 4-bit -> 8-bit
            const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
            const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

            // sub 8
            const svint8_t qx0s = svsub_n_s8_x(svptrue_b8(), qx0, 8);
            const svint8_t qx1s = svsub_n_s8_x(svptrue_b8(), qx1, 8);

            // load y
            const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
            const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

            // dot product
            sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
    } break;
    case 512:
    {
        // predicate for activating higher lanes for 32 int8 elements
        const svbool_t ph32 = svptrue_pat_b8(SV_VL32);

        // predicate for activating higher lanes for 16 int8 elements
        const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
        // predicate for activating lower lanes for 16 int8 elements from first 32 int8 activated lanes
        const svbool_t pl16 = svnot_b_z(ph32, ph16);

        for (; ib + 1 < nb; ib += 2) {
            const block_q4_0* restrict x0 = &x[ib + 0];
            const block_q4_0* restrict x1 = &x[ib + 1];
            const block_q8_0* restrict y0 = &y[ib + 0];
            const block_q8_0* restrict y1 = &y[ib + 1];

            // load x
            const svuint8_t qx0r = svld1rq_u8(ph32, x0->qs);
            const svuint8_t qx1r = svld1rq_u8(ph32, x1->qs);

            // 4-bit -> 8-bit
            const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
            const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

            // sub 8
            const svint8_t qx0s = svsub_n_s8_x(ph32, qx0, 8);
            const svint8_t qx1s = svsub_n_s8_x(ph32, qx1, 8);

            // load y
            const svint8_t qy0 = svld1_s8(ph32, y0->qs);
            const svint8_t qy1 = svld1_s8(ph32, y1->qs);

            // dot product
            sumv0 = svmla_n_f32_x(ph32, sumv0, svcvt_f32_s32_x(ph32,
                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(ph32, sumv1, svcvt_f32_s32_x(ph32,
                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(ph32, svadd_f32_x(ph32, sumv0, sumv1));
    } break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0* restrict x0 = &x[ib + 0];
        const block_q4_0* restrict x1 = &x[ib + 1];
        const block_q8_0* restrict y0 = &y[ib + 0];
        const block_q8_0* restrict y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8(8);
        qx = _mm256_sub_epi8(qx, off);

        __m256i qy = _mm256_loadu_si256((const __m256i*)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i*)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i*)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i*)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i*)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_1), _mm_set1_epi8(8));
        const __m128i q4b_1_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
        const __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_2), _mm_set1_epi8(8));
        const __m128i q4b_2_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));

        const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
        const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
        const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
        const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
        const __m128i p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
        const __m128i p_2 = _mm_add_epi16(p16_2_0, p16_2_1);
        const __m256 p = sum_i16_pairs_float(p_2, p_1);

        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#elif defined(__SSSE3__)
    // set constants
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        _mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i*)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[ib].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps(GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d));

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i*)y[ib + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[ib + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps(d_0_1, p0);
        __m128 p1_d = _mm_mul_ps(d_0_1, p1);
        __m128 p2_d = _mm_mul_ps(d_2_3, p2);
        __m128 p3_d = _mm_mul_ps(d_2_3, p3);

        // Acummulate
        acc_0 = _mm_add_ps(p0_d, acc_0);
        acc_1 = _mm_add_ps(p1_d, acc_1);
        acc_2 = _mm_add_ps(p2_d, acc_2);
        acc_3 = _mm_add_ps(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk / 2);

    for (; ib < nb; ++ib) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[ib].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[ib].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[ib].qs + 16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        // subtract offset
        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 8, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 8, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
    }

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector signed char v8 = vec_splats((signed char)0x8);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char qxs = (vector signed char)vec_xl(0, x[ib].qs);
        vector signed char q8y0 = vec_xl(0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed char q4x0 = vec_and(qxs, lowMask);
        vector signed char q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed int vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = __lasx_xvreplgr2vr_b(8);
        qx = __lasx_xvsub_b(qx, off);

        __m256i qy = __lasx_xvld((const __m256i*)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s(d, q, acc);
    }

    sumf = hsum_float_8(acc);

#elif defined(__loongarch_sx)
    // set constants
    const __m128i low_mask = __lsx_vreplgr2vr_b(0xF);
    const __m128i off = __lsx_vreplgr2vr_b(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = (__m128)__lsx_vldi(0);
    __m128 acc_1 = (__m128)__lsx_vldi(0);
    __m128 acc_2 = (__m128)__lsx_vldi(0);
    __m128 acc_3 = (__m128)__lsx_vldi(0);

    for (; ib + 1 < nb; ib += 2) {

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = (__m128)__lsx_vreplgr2vr_w(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        const __m128i tmp_0_1 = __lsx_vld((const __m128i*)x[ib].qs, 0);

        __m128i bx_0 = __lsx_vand_v(low_mask, tmp_0_1);
        __m128i by_0 = __lsx_vld((const __m128i*)y[ib].qs, 0);
        bx_0 = __lsx_vsub_b(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_0_1, 4));
        __m128i by_1 = __lsx_vld((const __m128i*)(y[ib].qs + 16), 0);
        bx_1 = __lsx_vsub_b(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        //_mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        //_mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = (__m128)__lsx_vreplgr2vr_w(GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d));

        const __m128i tmp_2_3 = __lsx_vld((const __m128i*)x[ib + 1].qs, 0);

        __m128i bx_2 = __lsx_vand_v(low_mask, tmp_2_3);
        __m128i by_2 = __lsx_vld((const __m128i*)y[ib + 1].qs, 0);
        bx_2 = __lsx_vsub_b(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_2_3, 4));
        __m128i by_3 = __lsx_vld((const __m128i*)(y[ib + 1].qs + 16), 0);
        bx_3 = __lsx_vsub_b(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = __lsx_vffint_s_w(i32_0);
        __m128 p1 = __lsx_vffint_s_w(i32_1);
        __m128 p2 = __lsx_vffint_s_w(i32_2);
        __m128 p3 = __lsx_vffint_s_w(i32_3);

        // Apply the scale
        __m128 p0_d = __lsx_vfmul_s(d_0_1, p0);
        __m128 p1_d = __lsx_vfmul_s(d_0_1, p1);
        __m128 p2_d = __lsx_vfmul_s(d_2_3, p2);
        __m128 p3_d = __lsx_vfmul_s(d_2_3, p3);

        // Acummulate
        acc_0 = __lsx_vfadd_s(p0_d, acc_0);
        acc_1 = __lsx_vfadd_s(p1_d, acc_1);
        acc_2 = __lsx_vfadd_s(p2_d, acc_2);
        acc_3 = __lsx_vfadd_s(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk / 2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >> 4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk / 2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi * toFloat32(std::bit_cast<ggml_fp16_t>(x[ib].d)) *
            toFloat32(std::bit_cast<ggml_fp16_t>(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q8_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc)
{
    const int qk = block_q8_0::block_size;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q8_0* restrict vx0 = vx;
        const block_q8_0* restrict vx1 = (const block_q8_0*)((const uint8_t*)vx + bx);
        const block_q8_0* restrict vy0 = vy;
        const block_q8_0* restrict vy1 = (const block_q8_0*)((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0* restrict b_x0 = &vx0[i];
            const block_q8_0* restrict b_y0 = &vy0[i];

            const block_q8_0* restrict b_x1 = &vx1[i];
            const block_q8_0* restrict b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d) * GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d) * GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d) * GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d) * GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0, (vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s, vget_low_f32(sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt() * 8;

    //VLA Implemenation for SVE
    switch (vector_length) {
    case 128:
    {
        // predicate for activating lanes for 16 Int8 elements
        const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
        const svbool_t pl16 = svptrue_pat_b32(SV_VL4);

        for (; ib + 1 < nb; ib += 2) {
            const block_q8_0* restrict x0 = &x[ib + 0];
            const block_q8_0* restrict x1 = &x[ib + 1];
            const block_q8_0* restrict y0 = &y[ib + 0];
            const block_q8_0* restrict y1 = &y[ib + 1];

            // load x
            const svint8_t qx0_0 = svld1_s8(ph16, x0->qs);
            const svint8_t qx0_1 = svld1_s8(ph16, x0->qs + 16);
            const svint8_t qx1_0 = svld1_s8(ph16, x1->qs);
            const svint8_t qx1_1 = svld1_s8(ph16, x1->qs + 16);

            // load y
            const svint8_t qy0_0 = svld1_s8(ph16, y0->qs);
            const svint8_t qy0_1 = svld1_s8(ph16, y0->qs + 16);
            const svint8_t qy1_0 = svld1_s8(ph16, y1->qs);
            const svint8_t qy1_1 = svld1_s8(ph16, y1->qs + 16);

            sumv0 = svmla_n_f32_x(pl16, sumv0, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                svdot_s32(svdup_n_s32(0), qx0_0, qy0_0),
                svdot_s32(svdup_n_s32(0), qx0_1, qy0_1))), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(pl16, sumv1, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                svdot_s32(svdup_n_s32(0), qx1_0, qy1_0),
                svdot_s32(svdup_n_s32(0), qx1_1, qy1_1))), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(pl16, svadd_f32_x(pl16, sumv0, sumv1));
    } break;
    case 256:
    {
        //printf("sve256");
        for (; ib + 1 < nb; ib += 2) {
            const block_q8_0* restrict x0 = &x[ib + 0];
            const block_q8_0* restrict x1 = &x[ib + 1];
            const block_q8_0* restrict y0 = &y[ib + 0];
            const block_q8_0* restrict y1 = &y[ib + 1];

            // load x
            const svint8_t qx0 = svld1_s8(svptrue_b8(), x0->qs);
            const svint8_t qx1 = svld1_s8(svptrue_b8(), x1->qs);

            // load y
            const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
            const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

            sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                svdot_s32(svdup_n_s32(0), qx0, qy0)), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));
            sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                svdot_s32(svdup_n_s32(0), qx1, qy1)), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
        }

        sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
    } break;
    case 512:
    {
        // predicate for activating high 256 bit
        const svbool_t ph32 = svptrue_pat_b8(SV_VL32);
        // predicate for activating low 256 bit
        const svbool_t pl32 = svnot_b_z(svptrue_b8(), ph32);

        // predicate for activating high lanes for 8 float32 elements
        const svbool_t ph8 = svptrue_pat_b32(SV_VL8);
        // predicate for activating low lanes for 8 float32 elements
        const svbool_t pl8 = svnot_b_z(svptrue_b32(), ph8);

        svfloat32_t sumv00 = svdup_n_f32(0.0f);

        for (; ib + 1 < nb; ib += 2) {
            const block_q8_0* restrict x0 = &x[ib + 0];
            const block_q8_0* restrict x1 = &x[ib + 1];
            const block_q8_0* restrict y0 = &y[ib + 0];
            const block_q8_0* restrict y1 = &y[ib + 1];

            //load 32 int8_t in first half of vector and put another 32 int8_t in second vector lower bits
            // and add them to make one 64 element vector
            // load x
            const svint8_t qx_32 = svld1_s8(ph32, x0->qs);
            svint8_t qx_64 = svld1_s8(pl32, x0->qs + 2);

            qx_64 = svadd_s8_x(svptrue_b8(), qx_32, qx_64);

            // load y
            const svint8_t qy_32 = svld1_s8(ph32, y0->qs);
            svint8_t qy_64 = svld1_s8(pl32, y0->qs + 2);

            qy_64 = svadd_s8_x(svptrue_b8(), qy_32, qy_64);

            // scale creation
            const float32_t deq1 = GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d);
            const float32_t deq2 = GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d);

            // duplicate deq1 in first half of vector and deq2 in second half of vector
            const svfloat32_t temp = svdup_f32_m(svdup_f32_z(ph8, deq1), pl8, deq2);

            const svfloat32_t sumvt = svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx_64, qy_64));

            sumv00 = svmla_f32_m(svptrue_b32(), sumv00, sumvt, temp);
        }

        sumf = svaddv_f32(svptrue_b32(), sumv00);
        break;
    }
    default:
        assert(false && "Unsupported vector length");
        break;
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0* restrict x0 = &x[ib + 0];
        const block_q8_0* restrict x1 = &x[ib + 1];
        const block_q8_0* restrict y0 = &y[ib + 0];
        const block_q8_0* restrict y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
            ggml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
            ggml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
            ggml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), GGML_FP16_TO_FP32(x1->d) * GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i*)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i*)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128((const __m128i*)x[ib].qs);
        const __m128i qx_1_1 = _mm_loadu_si128((const __m128i*)x[ib].qs + 1);
        const __m128i qx_2_0 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m128i qx_2_1 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs + 1);
        const __m128i qy_1_0 = _mm_loadu_si128((const __m128i*)y[ib].qs);
        const __m128i qy_1_1 = _mm_loadu_si128((const __m128i*)y[ib].qs + 1);
        const __m128i qy_2_0 = _mm_loadu_si128((const __m128i*)y[ib + 1].qs);
        const __m128i qy_2_1 = _mm_loadu_si128((const __m128i*)y[ib + 1].qs + 1);

        const __m256 p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk);

    for (; ib < nb; ++ib) {
        // load elements
        vint8m1_t bx_0 = __riscv_vle8_v_i8m1(x[ib].qs, vl);
        vint8m1_t by_0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);

        vint16m2_t vw_mul = __riscv_vwmul_vv_i16m2(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m2_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi * (GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
    }
#elif defined(__POWER9_VECTOR__)
    const vector signed int v0 = vec_splats((int32_t)0);
    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char q8x0 = vec_xl(0, x[ib].qs);
        vector signed char q8x1 = vec_xl(16, x[ib].qs);
        vector signed char q8y0 = vec_xl(0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_mule(q8x0, q8y0);
        vector signed short qv1 = vec_mulo(q8x0, q8y0);
        vector signed short qv2 = vec_mule(q8x1, q8y1);
        vector signed short qv3 = vec_mulo(q8x1, q8y1);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_add(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = __lasx_xvld((const __m256i*)x[ib].qs, 0);
        __m256i qy = __lasx_xvld((const __m256i*)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = __lasx_xvfmadd_s(d, q, acc);
    }

    sumf = hsum_float_8(acc);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }

        sumf += sumi * (toFloat32(std::bit_cast<ggml_fp16_t>(x[ib].d)) * toFloat32(std::bit_cast<ggml_fp16_t>(y[ib].d)));
    }

    *s = sumf;
}