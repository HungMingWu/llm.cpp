module;
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <array>
#include <bit>
#include "block.h"
#include "table.h"

#define UNUSED(x) (void)(x)

module ggml;
import :cpu.vec_dot;
import :types;

void ggml_vec_dot_q1_0_q8_0_generic(int n, float* s, size_t bs, const block_q1_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc) {
    const int qk = block_q1_0::block_size;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float d0 = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const float d1 = toFloat32(std::bit_cast<ggml_fp16_t>(y[i * 4 + k].d));

            int sumi_block = 0;

            for (int j = 0; j < block_q8_0::block_size; j++) {
                const int bit_index = k * block_q8_0::block_size + j;
                const int byte_index = bit_index / 8;
                const int bit_offset = bit_index % 8;

                const int xi = ((x[i].qs[byte_index] >> bit_offset) & 1) ? 1 : -1;
                sumi_block += xi * y[i * 4 + k].qs[j];
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

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

    int ib = 0;
    float sumf = 0;

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

void ggml_vec_dot(int n, float* s, size_t bs, const block_q4_1* x, size_t bx, const block_q8_1* y, size_t by, int nrc)
{
    const int qk = QK8_1;
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

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk / 2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >> 4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk / 2]);
        }

        int sumi = sumi0 + sumi1;

        const auto x_dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[ib].dm);
        const auto y_ds = std::bit_cast<std::array<ggml_fp16_t, 2>>(y[ib].ds);

        sumf += toFloat32(x_dm[0]) * toFloat32(y_ds[0]) * sumi +
            toFloat32(x_dm[1]) * toFloat32(y_ds[1]);

    }

    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t /*bs*/, const block_mxfp4* x, size_t /*bx*/, const block_q8_0* y, size_t /*by*/, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    assert(n % block_mxfp4::block_size == 0);
    static_assert(block_mxfp4::block_size == block_q8_0::block_size, "QK_MXFP4 and QK8_0 must be the same");

    const int nb = n / block_mxfp4::block_size;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(y[ib].d)) * ggml_e8m0_to_fp32_half(x[ib].e);
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < block_q8_0::block_size / 2; ++j) {
            sumi1 += y[ib].qs[j + 0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + block_q8_0::block_size / 2] * kvalues_mxfp4[x[ib].qs[j] >> 4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t /*bs*/, const block_nvfp4* x, size_t /*bx*/, const block_q8_0* y, size_t /*by*/, int nrc) {
    assert(nrc == 1);
    assert(n % block_nvfp4::block_size == 0);

    // Each NVFP4 super-block (64 elements) spans 2 q8_0 blocks
    const int nb = n / block_nvfp4::block_size;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        for (int si = 0; si < 4; ++si) {
            const float d = toFloat32(std::bit_cast<ggml_ue4m3_t>(x[ib].d[si]));
            const int q8b = si / 2;
            const int q8o = (si % 2) * QK_NVFP4_SUB;
            const float dy = toFloat32(std::bit_cast<ggml_fp16_t>(y[2 * ib + q8b].d));

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                const uint8_t qv = x[ib].qs[si * (QK_NVFP4_SUB / 2) + j];
                sumi_lo += y[2 * ib + q8b].qs[q8o + j + 0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2 * ib + q8b].qs[q8o + j + QK_NVFP4_SUB / 2] * kvalues_mxfp4[qv >> 4];
            }
            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    * s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q5_0* x, size_t bx, const block_q8_0* y, size_t by, int nrc)
{
    const int qk = block_q8_0::block_size;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == block_q5_0::block_size);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >> 4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk / 2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += toFloat32(std::bit_cast<ggml_fp16_t>(x[ib].d)) *
            toFloat32(std::bit_cast<ggml_fp16_t>(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q5_1* x, size_t bx, const block_q8_1* y, size_t by, int nrc)
{
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == block_q5_1::block_size);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >> 4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk / 2]);
        }

        int sumi = sumi0 + sumi1;
        auto x_dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[ib].dm);
        auto y_ds = std::bit_cast<std::array<ggml_fp16_t, 2>>(y[ib].ds);
        sumf += toFloat32(x_dm[0]) * toFloat32(y_ds[0]) * sumi + toFloat32(x_dm[1]) * toFloat32(y_ds[1]);
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

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }

        sumf += sumi * (toFloat32(std::bit_cast<ggml_fp16_t>(x[ib].d)) * toFloat32(std::bit_cast<ggml_fp16_t>(y[ib].d)));
    }

    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q2_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t* q2 = x[i].qs;
        const  int8_t* q8 = y[i].qs;
        const uint8_t* sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        auto x_dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float dall = y[i].d * toFloat32(x_dm[0]);
        const float dmin = y[i].d * toFloat32(x_dm[1]);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K / 128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q3_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const int nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    uint32_t auxs[4];
    const int8_t* scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q3 = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        const  int8_t* q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t* a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K / 16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q4_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t* scales = (const uint8_t*)&utmp[0];
    const uint8_t* mins = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].qs;
        const  int8_t* q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t* a = aux8;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        auto x_dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(x_dm[0]) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = toFloat32(x_dm[1]) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q5_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t* scales = (const uint8_t*)&utmp[0];
    const uint8_t* mins = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].qs;
        const uint8_t* hm = x[i].qh;
        const  int8_t* q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t* a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        auto x_dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(x_dm[0]) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = toFloat32(x_dm[1]) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_q6_K* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const  int8_t* q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t* a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq2_xxs* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq2_xs* x, size_t bx, const block_q8_K* y, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        const uint16_t* q2 = x[i].qs;
        const uint8_t* sc = x[i].scales;
        const int8_t* q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint16_t ls1 = 2 * (sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2 * (sc[ib32] >> 4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq2_s* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        const int8_t* q8 = y[i].qs;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = qs + QK_K / 8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            int ls1 = 1 + 2 * (x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2 * (x[i].scales[ib32] >> 4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq3_xxs* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        const uint8_t* q3 = x[i].qs;
        const uint8_t* gas = x[i].qs + QK_K / 4;
        const int8_t* q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2 * (aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + q3[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + q3[2 * l + 1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs & kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq3_s* x, size_t bx, const block_q8_K* y, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        const int8_t* q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const uint32_t ls1 = 2 * (x[i].scales[ib32 / 2] & 0xf) + 1;
            const uint32_t ls2 = 2 * (x[i].scales[ib32 / 2] >> 4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[ib32 + 0] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[ib32 + 0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs[l] & kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs[l] & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[ib32 + 1] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[ib32 + 1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j + 0] * (signs[l] & kmask_iq2xs[j + 0] ? -1 : 1);
                    sumi += grid2[j] * q8[j + 4] * (signs[l] & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq1_s* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t* q8 = y[i].qs;
        const uint8_t* qs = x[i].qs;
        const uint16_t* qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = 2 * ((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2 * ib + 0] + y[i].bsums[2 * ib + 1]);
            qs += 4;
        }

        sumf += toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}


void ggml_vec_dot(int n, float* s, size_t bs, const block_iq1_m* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    iq1m_scale_t scale;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t* q8 = y[i].qs;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint16_t* sc = (const uint16_t*)x[i].scales;

        scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((uint16_t)qh[l / 2] << (8 - 4 * (l % 2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l / 2] += lsum1;
                sum2[l / 2] += lsum2 * delta[l];
            }

            const int ls1 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1;
            const int ls2 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += toFloat32(std::bit_cast<ggml_fp16_t>(scale)) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq4_nl* x, size_t bx, const block_q8_0* y, size_t by, int nrc)
{
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % block_iq4_nl::block_size == 0);
    static_assert(block_iq4_nl::block_size == block_q8_0::block_size, "QK4_NL and QK8_0 must be the same");

    const int nb = n / block_iq4_nl::block_size;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(y[ib].d)) * toFloat32(std::bit_cast<ggml_fp16_t>(x[ib].d));
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < block_iq4_nl::block_size / 2; ++j) {
            sumi1 += y[ib].qs[j + 0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + block_iq4_nl::block_size / 2] * kvalues_iq4nl[x[ib].qs[j] >> 4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_iq4_xs* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = toFloat32(std::bit_cast<ggml_fp16_t>(x[ibl].d)) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t* qs = x[ibl].qs;
        const int8_t* q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib / 2] >> 4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8 * (ls1 - 32);
            const float d2 = d4d8 * (ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_tq1_0* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t)q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j * 5 + l * 32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t)q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j * 5 + l * 16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t)q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs) * 5 + l * sizeof(x->qh) + j];
            }
        }

        sumf += (float)sum * (toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d)) * y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot(int n, float* s, size_t bs, const block_tq2_0* x, size_t bx, const block_q8_K* y, size_t by, int nrc)
{
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const int nb = n / QK_K;

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j * 4 + l * 32 + k] * (((x[i].qs[j + k] >> (l * 2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        sumf += (float)sumi * d;
    }

    *s = sumf;
}

