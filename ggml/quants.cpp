module;
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <array>
#include <bit>
#include <span>
#include "block.h"
#include "table.h"
#include "quant_table.h"

#define GGML_ABORT(...)

module ggml;
import :types;
import :quants;

static constexpr float GROUP_MAX_EPS = 1e-15f;

static uint32_t compress(ggml_fp32_t first, ggml_fp32_t second)
{
    std::array<ggml_fp16_t, 2> arr{ fromFloat32<ggml_fp16_t>(first), fromFloat32<ggml_fp16_t>(second) };
    return std::bit_cast<uint32_t>(arr);
}

static uint16_t castToUint16(ggml_fp32_t value)
{
    return std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(value));
}

static ggml_fp32_t toFloat32(uint16_t value)
{
    return toFloat32(std::bit_cast<ggml_fp16_t>(value));
}

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float* x, int8_t* L, int rmse_type,
    const float* qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + std::max(-nmax, std::min(nmax - 1, l));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax - 1, l));
            }
            scale = sumlx / suml2; best = scale * sumlx;
        }
    }
    return scale;
}

static float make_q3_quants(int n, int nmax, const float* x, int8_t* L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            L[i] = l;
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i] * x[i];
                float slx = sumlx - w * x[i] * L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w * L[i] * L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = std::max(-nmax, std::min(nmax - 1, new_l));
                    if (new_l != L[i]) {
                        slx += w * x[i] * new_l;
                        sl2 += w * new_l * new_l;
                        if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = l + nmax;
    }
    return 1 / iscale;
}

static float make_qkx3_quants(int n, int nmax, const float* x, const float* weights,
    uint8_t* L, float* the_min, uint8_t* Laux,
    float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * (x[i] - min));
        L[i] = std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i] * x[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights ? weights[i] : x[i] * x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qp_quants(int n, int nmax, const float* x, uint8_t* L, const float* quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = std::max(max, x[i]);
    }
    if (max < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale * L[i];
        float w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max;
        float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is * x[i]);
            l = std::min(nmax, l);
            float diff = x[i] - scale_is * l;
            float w = quant_weights[i];
            mse += w * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::min(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = std::min(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static float make_qkx2_quants(int n, int nmax, const float* x, const float* weights,
    uint8_t* L, float* the_min, uint8_t* Laux,
    float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * (x[i] - min));
        L[i] = std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void quantize_row_q4_0_impl(const float* x, block_q4_0* y, int64_t n_per_row, const float* quant_weights) {
    static constexpr size_t QK4_0 = block_q4_1::block_size;
    static_assert(QK4_0 == 32, "QK4_0 must be 32");

    if (!quant_weights) {
        quantize_row_q4_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK4_0];
    int8_t L[QK4_0];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j] * x[j];
    float sigma2 = sum_x2 / n_per_row;

    const int64_t nb = n_per_row / QK4_0;
    for (int ib = 0; ib < nb; ++ib) {
        const float* xb = x + QK4_0 * ib;
        const float* qw = quant_weights + QK4_0 * ib;
        for (int j = 0; j < QK4_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
        float d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
        y[ib].d = castToUint16(d);
        for (int j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j + 16] << 4);
        }
    }
}

static void quantize_row_q4_1_impl(const float* x, block_q4_1* y, int64_t n_per_row, const float* quant_weights) {
    static constexpr size_t QK4_1 = block_q4_1::block_size;
    static_assert(QK4_1 == 32, "QK4_1 must be 32");

    if (!quant_weights) {
        quantize_row_q4_1_ref(x, y, n_per_row);
        return;
    }

    float weight[QK4_1];
    uint8_t L[QK4_1], Laux[QK4_1];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j] * x[j];
    float sigma2 = sum_x2 / n_per_row;

    const int64_t nb = n_per_row / QK4_1;
    for (int ib = 0; ib < nb; ++ib) {
        const float* xb = x + QK4_1 * ib;
        const float* qw = quant_weights + QK4_1 * ib;
        for (int j = 0; j < QK4_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
        float min;
        float d = make_qkx3_quants(QK4_1, 15, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
        y[ib].dm = compress(d, -min);
        for (int j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j + 16] << 4);
        }
    }
}

static void quantize_row_q5_0_impl(const float* x, block_q5_0* y, int64_t n_per_row, const float* quant_weights) {
    static constexpr size_t QK5_0 = block_q4_1::block_size;
    static_assert(QK5_0 == 32, "QK5_0 must be 32");

    if (!quant_weights) {
        quantize_row_q5_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK5_0];
    int8_t L[QK5_0];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j] * x[j];
    float sigma2 = sum_x2 / n_per_row;

    const int64_t nb = n_per_row / QK5_0;
    for (int ib = 0; ib < nb; ++ib) {
        const float* xb = x + QK5_0 * ib;
        const float* qw = quant_weights + QK5_0 * ib;
        for (int j = 0; j < QK5_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
        float d = make_qx_quants(QK5_0, 16, xb, L, 1, weight);
        y[ib].d = castToUint16(d);

        uint32_t qh = 0;

        for (int j = 0; j < 16; ++j) {
            const uint8_t xi0 = L[j];
            const uint8_t xi1 = L[j + 16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
        }

        memcpy(&y[ib].qh, &qh, sizeof(qh));
    }
}

static void quantize_row_q5_1_impl(const float* x, block_q5_1* y, int64_t n_per_row, const float* quant_weights) {
    static constexpr size_t QK5_1 = block_q4_1::block_size;
    static_assert(QK5_1 == 32, "QK5_1 must be 32");

    if (!quant_weights) {
        quantize_row_q5_1_ref(x, y, n_per_row);
        return;
    }

    float weight[QK5_1];
    uint8_t L[QK5_1], Laux[QK5_1];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j] * x[j];
    float sigma2 = sum_x2 / n_per_row;

    const int64_t nb = n_per_row / QK5_1;
    for (int ib = 0; ib < nb; ++ib) {
        const float* xb = x + QK5_1 * ib;
        const float* qw = quant_weights + QK5_1 * ib;
        for (int j = 0; j < QK5_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
        float min;
        float d = make_qkx3_quants(QK5_1, 31, xb, weight, L, &min, Laux, -0.9f, 0.05f, 36, false);
        y[ib].dm = compress(d, -min);

        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xi0 = L[j];
            const uint8_t xi1 = L[j + 16];
            y[ib].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
        }
        memcpy(&y[ib].qh, &qh, sizeof(qh));
    }
}

static void quantize_row_q2_K_impl(const float* x, block_q2_K* y, int k, const float* quant_weights) {
    assert(quant_weights);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    const bool requantize = true;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    float sw[QK_K / 16];
    float weight[16];
    uint8_t Ls[QK_K / 16], Lm[QK_K / 16];

    for (int i = 0; i < nb; i++) {
        memset(sw, 0, QK_K / 16 * sizeof(float));
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += x[j] * x[j];
        float sigma2 = sumx2 / QK_K;
        for (int j = 0; j < QK_K / 16; ++j) {
            const float* qw = quant_weights + QK_K * i + 16 * j;
            for (int l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
            for (int l = 0; l < QK_K / 16; ++l) sw[j] += weight[l];
            scales[j] = make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float dm, mm;
        dm = make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
        mm = make_qp_quants(QK_K / 16, 15, mins, Lm, sw);

        y[i].dm = compress(dm, mm);

        for (int j = 0; j < QK_K / 16; ++j) {
            y[i].scales[j] = Ls[j] | (Lm[j] << 4);
        }

        if (requantize) {
            for (int j = 0; j < QK_K / 16; ++j) {
                const float d = dm * (y[i].scales[j] & 0xF);
                if (!d) continue;
                const float m = mm * (y[i].scales[j] >> 4);
                for (int ii = 0; ii < 16; ++ii) {
                    int l = nearest_int((x[16 * j + ii] + m) / d);
                    l = std::max(0, std::min(3, l));
                    L[16 * j + ii] = l;
                }
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

static void quantize_row_q3_K_impl(const float* x, block_q3_K* y, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int nb = n_per_row / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];
    float weight[16];
    float sw[QK_K / 16];
    int8_t Ls[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += x[j] * x[j];
        float sigma2 = 2 * sumx2 / QK_K;

        for (int j = 0; j < QK_K / 16; ++j) {
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * i + 16 * j;
                for (int l = 0; l < 16; ++l) weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
            }
            else {
                for (int l = 0; l < 16; ++l) weight[l] = x[16 * j + l] * x[16 * j + l];
            }
            float sumw = 0;
            for (int l = 0; l < 16; ++l) sumw += weight[l];
            sw[j] = sumw;

            scales[j] = make_qx_quants(16, 4, x + 16 * j, L + 16 * j, 1, weight);

        }

        memset(y[i].scales, 0, 12);

        float d_block = make_qx_quants(QK_K / 16, 32, scales, Ls, 1, sw);
        for (int j = 0; j < QK_K / 16; ++j) {
            int l = Ls[j];
            if (j < 8) {
                y[i].scales[j] = l & 0xF;
            }
            else {
                y[i].scales[j - 8] |= ((l & 0xF) << 4);
            }
            l >>= 4;
            y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
        }
        y[i].d = castToUint16(d_block);

        int8_t sc;
        for (int j = 0; j < QK_K / 16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
            sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            float d = toFloat32(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16 * j + ii] / d);
                l = std::max(-4, std::min(3, l));
                L[16 * j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K / 8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K / 8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

static void quantize_row_q4_K_impl(const float* x, block_q4_K* y, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float   weights[32];
    float   sw[QK_K / 32];
    float   mins[QK_K / 32];
    float   scales[QK_K / 32];

    for (int i = 0; i < nb; i++) {

        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2 * sum_x2 / QK_K;
        float av_x = sqrtf(sigma2);

        for (int j = 0; j < QK_K / 32; ++j) {
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * i + 32 * j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
            }
            else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float d_block = make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
        float m_block = make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            }
            else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        y[i].dm = compress(d_block, m_block);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = d_block * sc;
            if (!d) continue;
            const float dm = m_block * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = std::max(0, std::min(15, l));
                L[32 * j + ii] = l;
            }
        }
        uint8_t* q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;

    }
}

static void quantize_row_q5_K_impl(const float* x, block_q5_K* y, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float   mins[QK_K / 32];
    float   scales[QK_K / 32];
    float   sw[QK_K / 32];
    float   weights[32];

    for (int i = 0; i < nb; i++) {

        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2 * sum_x2 / QK_K;
        float av_x = sqrtf(sigma2);

        for (int j = 0; j < QK_K / 32; ++j) {
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * i + 32 * j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
            }
            else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;

            scales[j] = make_qkx3_quants(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float d_block = make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
        float m_block = make_qp_quants(QK_K / 32, 63, mins, Lm, sw);

        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            ls = std::min<uint8_t>(63, ls);
            lm = std::min<uint8_t>(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            }
            else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        y[i].dm = compress(d_block, m_block);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = d_block * sc;
            if (!d) continue;
            const float dm = m_block * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = std::max(0, std::min(31, l));
                L[32 * j + ii] = l;
            }
        }

        uint8_t* qh = y[i].qh;
        uint8_t* ql = y[i].qs;
        memset(qh, 0, QK_K / 8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;

    }
}

static void quantize_row_q6_K_impl(const float* x, block_q6_K* y, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K / 16];
    //float   weights[16];

    for (int i = 0; i < nb; i++) {

        //float sum_x2 = 0;
        //for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
        //float sigma2 = sum_x2/QK_K;

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K / 16; ++ib) {

            float scale;
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * i + 16 * ib;
                //for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
                //scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
                scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, qw);
            }
            else {
                scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
            }
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = castToUint16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f / max_scale;
        y[i].d = castToUint16(1 / iscale);
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            y[i].scales[ib] = std::min(127, nearest_int(iscale * scales[ib]));
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            float d = toFloat32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16 * j + ii] / d);
                l = std::max(-32, std::min(31, l));
                L[16 * j + ii] = l + 32;
            }
        }

        uint8_t* ql = y[i].ql;
        uint8_t* qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l + 0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l + 0] = q1 | (q3 << 4);
                ql[l + 32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;

    }
}

static int iq2_find_best_neighbour(const uint16_t* neighbours, std::span<const uint64_t> grid,
    const float* xval, const float* weight, float scale, int8_t* L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t* pg = (const int8_t*)(&grid[neighbours[j]]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t* pg = (const int8_t*)(&grid[grid_index]);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;
    return grid_index;
}

static void quantize_row_iq2_xxs_impl(const float* x, block_iq2_xxs* y, int64_t n, const float* quant_weights) {

    const auto &kgrid_q2xs = table_iq2_xxs_grid;
    const auto &kmap_q2xs = table_iq2_xxs_map;
    const auto &kneighbors_q2xs = table_iq2_xxs_neighbors;

    assert(quant_weights && "missing quantization weights");
    assert(n % QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n / QK_K;

    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2 * (QK_K / 32)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));
        memset(q2, 0, QK_K / 4);

        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float* xb = xbl + 32 * ib;
            const float* qw = quant_weights + QK_K * ibl + 32 * ib;
            for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                    else {
                        xval[8 * k + i] = -xb[8 * k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip % 2) {
                    int imin = 0; float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ + 1, xval, (uint8_t*)L, weight);
            float eff_max = scale * kMaxQ;
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
                float this_scale = 1 / id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8 * k + i] << 2 * i);
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t* neighbours = &kneighbors_q2xs[- kmap_q2xs[u] - 1];
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2; best = scale * sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                float id = 1 / scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (l << 2 * i);
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t* neighbours = &kneighbors_q2xs[-kmap_q2xs[u] - 1];
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                    }
                    const int8_t* pg = (const int8_t*)(&kgrid_q2xs[grid_index]);
                    for (int i = 0; i < 8; ++i) L[8 * k + i] = (pg[i] - 1) / 2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8 * k + i] << 2 * i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8 * k + i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2 * ib + 0] |= ((uint32_t)grid_index << 8 * k);
                q2[2 * ib + 1] |= (block_signs[k] << 7 * k);
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K / 4);
            continue;
        }

        float d = max_scale / 31;
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d));
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            q2[2 * ib + 1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K / 4);
    }
}

static void quantize_row_iq2_xs_impl(const float* x, block_iq2_xs* y, int64_t n, const float* quant_weights) {

    const auto &kgrid_q2xs = table_iq2_xs_grid;
    const auto& kmap_q2xs = table_iq2_xs_map;
    const auto &kneighbors_q2xs = table_iq2_xs_neighbors;

    assert(quant_weights && "missing quantization weights");
    assert(n % QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n / QK_K;

    float scales[QK_K / 16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2 * (QK_K / 16)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));
        memset(q2, 0, QK_K / 4);
        memset(y[ibl].scales, 0, QK_K / 32);

        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / 16; ++ib) {
            const float* xb = xbl + 16 * ib;
            const float* qw = quant_weights + QK_K * ibl + 16 * ib;
            for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                    else {
                        xval[8 * k + i] = -xb[8 * k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip % 2) {
                    int imin = 0; float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 16);
                continue;
            }
            float best = 0;
            float scale = max / (2 * kMaxQ - 1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                float this_scale = 1 / id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8 * k + i] << 2 * i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t* neighbours = &kneighbors_q2xs [-kmap_q2xs[u] - 1];
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2; best = scale * sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1 / scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (l << 2 * i);
                        L[8 * k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t* neighbours = &kneighbors_q2xs[-kmap_q2xs[u] - 1];
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8 * k + i] << 2 * i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8 * k + i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2 * ib + k] = grid_index | (block_signs[k] << 9);
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K / 4);
            continue;
        }

        float d = max_scale / 31;
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d));
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            if (ib % 2 == 0) y[ibl].scales[ib / 2] = l;
            else y[ibl].scales[ib / 2] |= (l << 4);
        }
        memcpy(y[ibl].qs, q2, QK_K / 4);

    }
}

static constexpr float GROUP_MAX_EPS_IQ2_S = 1e-8f;

static void quantize_row_iq2_s_impl(const float* x, block_iq2_s* y, int64_t n, const float* quant_weights) {

    const auto& kgrid_q2xs = table_iq2_s_grid;
    const auto& kmap_q2xs = table_iq2_s_map;
    const auto& kneighbors_q2xs = table_iq2_s_neighbors;

    assert(n % QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n / QK_K;

    float scales[QK_K / 16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));
        
        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = 2 * sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / 16; ++ib) {
            const float* xb = xbl + 16 * ib;
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * ibl + 16 * ib;
                for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            }
            else {
                for (int i = 0; i < 16; ++i) weight[i] = 0.25f * sigma2 + xb[i] * xb[i];
            }
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                    else {
                        xval[8 * k + i] = -xb[8 * k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ2_S) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max / (2 * kMaxQ - 1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                float this_scale = 1 / id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8 * k + i] << 2 * i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t* neighbours = &kneighbors_q2xs[-kmap_q2xs[u] - 1];
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2; best = scale * sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1 / scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (l << 2 * i);
                        L[8 * k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t* neighbours = &kneighbors_q2xs[-kmap_q2xs[u] - 1];
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8 * k + i] << 2 * i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8 * k + i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                const int i8 = 2 * ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8 / 4] |= ((grid_index >> 8) << 2 * (i8 % 4));
                y[ibl].qs[QK_K / 8 + i8] = block_signs[k];
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale / 31;
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d * 0.9875f));
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            if (ib % 2 == 0) y[ibl].scales[ib / 2] = l;
            else y[ibl].scales[ib / 2] |= (l << 4);
        }
    }
}

static int iq3_find_best_neighbour(const uint16_t* neighbours, std::span<const uint32_t> grid,
    const float* xval, const float* weight, float scale, int8_t* L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t* pg = (const int8_t*)(&grid[neighbours[j]]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t* pg = (const int8_t*)(&grid[grid_index]);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1) / 2;
    return grid_index;
}

static constexpr float GROUP_MAX_EPS_IQ3_XXS = 1e-8f;

static void quantize_row_iq3_xxs_impl(int grid_size, const float* x, void* vy, int64_t n,
    const float* quant_weights) {

    const auto& kgrid_q3xs = table_iq3_xxs_grid;
    const auto& kmap_q3xs = table_iq3_xxs_map;
    const auto& kneighbors_q3xs = table_iq3_xxs_neighbors;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    assert(n % QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n / QK_K;

    uint16_t* dh;
    uint8_t* qs;
    int block_size;
    if (grid_size == 256) {
        block_iq3_xxs* y = (block_iq3_xxs *)vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_xxs);
    }
    else {
        block_iq3_s* y = (block_iq3_s *)vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_s);
    }
    int quant_size = block_size - sizeof(uint16_t);

    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint8_t q3[3 * (QK_K / 8) + QK_K / 32];
    uint32_t* scales_and_signs = (uint32_t*)(q3 + QK_K / 4);
    uint8_t* qh = q3 + 3 * (QK_K / 8);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));
        memset(q3, 0, 3 * QK_K / 8 + QK_K / 32);

        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = 2 * sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float* xb = xbl + 32 * ib;
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * ibl + 32 * ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            }
            else {
                for (int i = 0; i < 32; ++i) weight[i] = xb[i] * xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                    else {
                        xval[8 * k + i] = -xb[8 * k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip % 2) {
                    int imin = 0; float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            float scale = max / (2 * kMaxQ - 1);
            for (int is = -15; is <= 15; ++is) {
                float id = (2 * kMaxQ - 1 + is * 0.2f) / max;
                float this_scale = 1 / id;
                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1));
                        Laux[4 * k + i] = std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4 * k + i] << 3 * i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t* neighbours = &kneighbors_q3xs[-kmap_q3xs[u] - 1];
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2; best = scale * sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1 / scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (l << 3 * i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t* neighbours = &kneighbors_q3xs[-kmap_q3xs[u] - 1];
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
                    }
                    const int8_t* pg = (const int8_t*)(&kgrid_q3xs[grid_index]);
                    for (int i = 0; i < 4; ++i) L[4 * k + i] = (pg[i] - 1) / 2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4 * k + i] << 3 * i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4 * k + i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                if (grid_size == 256) {
                    q3[8 * ib + k] = grid_index;
                }
                else {
                    q3[8 * ib + k] = grid_index & 255;
                    qh[ib] |= ((grid_index >> 8) << k);
                }

            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size / sizeof(ggml_fp16_t);
            qs += block_size;
            continue;
        }

        float d = max_scale / 31;
        dh[0] = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d * 1.0125f));  // small improvement via this fudge factor
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size / sizeof(ggml_fp16_t);
        qs += block_size;

    }
}

static void quantize_row_iq3_s_impl(int block_size, const float* x, block_iq3_s* y, int n,
    const float* quant_weights,
    float* scales,
    float* weight,
    float* xval,
    int8_t* L,
    int8_t* Laux,
    float* waux,
    bool* is_on_grid,
    bool* is_on_grid_aux,
    uint8_t* block_signs) {

    const auto& kgrid_q3xs = table_iq3_s_grid;
    const auto& kmap_q3xs = table_iq3_s_map;
    const auto& kneighbors_q3xs = table_iq3_s_neighbors;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    assert(n % QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n / QK_K;

    const int bs4 = block_size / 4;
    const int bs8 = block_size / 8;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));

        uint8_t* qs = y[ibl].qs;
        uint8_t* qh = y[ibl].qh;
        uint8_t* signs = y[ibl].signs;

        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = 2 * sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / block_size; ++ib) {
            const float* xb = xbl + block_size * ib;
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * ibl + block_size * ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            }
            else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i] * xb[i];
            }
            for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                    else {
                        xval[8 * k + i] = -xb[8 * k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = std::max(max, xval[i]);
            if (!max) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max / (2 * kMaxQ - 1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2 * kMaxQ - 1 + is * 0.2f) / max;
                float this_scale = 1 / id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1));
                        Laux[4 * k + i] = std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4 * k + i] << 3 * i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t* neighbours = &kneighbors_q3xs[-kmap_q3xs[u] - 1];
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2; best = scale * sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1 / scale;
                for (int k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (l << 3 * i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t* neighbours = &kneighbors_q3xs[-kmap_q3xs[u] - 1];
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
                    }
                    const int8_t* pg = (const int8_t*)(&kgrid_q3xs[grid_index]);
                    for (int i = 0; i < 4; ++i) L[4 * k + i] = (pg[i] - 1) / 2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4 * k + i] << 3 * i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4 * k + i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                qs[k] = grid_index & 255;
                qh[(ib * bs4 + k) / 8] |= ((grid_index >> 8) << ((ib * bs4 + k) % 8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale / 31;
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d * 1.033f));
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / block_size; ib += 2) {
            int l1 = nearest_int(0.5f * (id * scales[ib + 0] - 1));
            l1 = std::max(0, std::min(15, l1));
            int l2 = nearest_int(0.5f * (id * scales[ib + 1] - 1));
            l2 = std::max(0, std::min(15, l2));
            y[ibl].scales[ib / 2] = l1 | (l2 << 4);
        }

    }
}

static constexpr size_t IQ1S_BLOCK_SIZE = 32;

static int iq1_find_best_neighbour2(const uint16_t* neighbours, std::span<const uint64_t> grid,
    const float* xval, const float* weight, float scale, const float* xg, int8_t* L, int ngrid) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t* pg = (const int8_t*)(&grid[neighbours[j]]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1) / 2];
            float w = weight[i];
            float diff = scale * q - xval[i];
            d2 += w * diff * diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
        }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t* grid_i = (const int8_t*)(&grid[i]);
            float d2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = weight[j];
                float q = xg[(grid_i[j] - 1) / 2];
                float diff = scale * q - xval[i];
                d2 += w * diff * diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    if (grid_index < 0) {
        printf("Oops, did not find grid point\n");
        printf("Have %d neighbours\n", num_neighbors);
        for (int j = 1; j <= num_neighbors; ++j) {
            const int8_t* pg = (const int8_t*)(&grid[neighbours[j]]);
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 8; ++i) {
                float q = xg[(pg[i] - 1) / 2];
                float w = weight[i];
                sumqx += w * q * xval[i];
                sumq2 += w * q * q;
            }
            printf("    neighbour %d: sumqx = %g sumq2 = %g\n", j, (double)sumqx, (double)sumq2);
        }
    }
    assert(grid_index >= 0);
    const int8_t* pg = (const int8_t*)(&grid[grid_index]);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;
    return grid_index;
}

static int iq1_sort_helper(const void* left, const void* right) {
    const float* l = (const float*)left;
    const float* r = (const float*)right;
    return *l < *r ? -1 : *l > *r ? 1 : 0;
}

static constexpr float GROUP_MAX_EPS_IQ1_S = 1e-12f;

static void quantize_row_iq1_s_impl(const float* x, block_iq1_s* y, int64_t n, const float* quant_weights,
    float* scales,
    float* weight,
    float* sumx,
    float* sumw,
    float* pairs,
    int8_t* L,
    uint16_t* index,
    int8_t* shifts) {

    const auto& kgrid_q2xs = table_iq1_grid;
    const auto& kmap_q2xs = table_iq1_map;
    const auto& kneighbors_q2xs = table_iq1_neighbors;

    assert(quant_weights && "missing quantization weights");
    assert(n % QK_K == 0);

    const int64_t nbl = n / QK_K;

    const int block_size = IQ1S_BLOCK_SIZE;

    const float x_p[3] = { -1 + IQ1S_DELTA,  IQ1S_DELTA, 1 + IQ1S_DELTA };
    const float x_m[3] = { -1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA };

    int* idx = (int*)(pairs + 1);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(0.f));
        memset(y[ibl].qs, 0, QK_K / 8);
        memset(y[ibl].qh, 0, QK_K / 16);

        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = 2 * sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / block_size; ++ib) {
            const float* xb = xbl + block_size * ib;
            const float* qw = quant_weights + QK_K * ibl + block_size * ib;
            for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = std::max(max, fabsf(xb[i]));
            if (max < GROUP_MAX_EPS_IQ1_S) {
                scales[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2 * j] = xb[j];
                idx[2 * j] = j;
            }
            qsort(pairs, block_size, 2 * sizeof(float), iq1_sort_helper);
            {
                sumx[0] = sumw[0] = 0;
                for (int j = 0; j < block_size; ++j) {
                    int i = idx[2 * j];
                    sumx[j + 1] = sumx[j] + weight[i] * xb[i];
                    sumw[j + 1] = sumw[j] + weight[i];
                }
            }
            float best_score = -FLT_MIN, scale = max;
            int besti1 = -1, besti2 = -1, best_shift = 0;
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    float sumqx = (sumx[i1] - sumx[0]) * x_p[0] + (sumx[i2] - sumx[i1]) * x_p[1] + (sumx[block_size] - sumx[i2]) * x_p[2];
                    float sumq2 = (sumw[i1] - sumw[0]) * x_p[0] * x_p[0] + (sumw[i2] - sumw[i1]) * x_p[1] * x_p[1] + (sumw[block_size] - sumw[i2]) * x_p[2] * x_p[2];
                    if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
                        scale = sumqx / sumq2; best_score = scale * sumqx;
                        besti1 = i1; besti2 = i2; best_shift = 1;
                    }
                    sumqx = (sumx[i1] - sumx[0]) * x_m[0] + (sumx[i2] - sumx[i1]) * x_m[1] + (sumx[block_size] - sumx[i2]) * x_m[2];
                    sumq2 = (sumw[i1] - sumw[0]) * x_m[0] * x_m[0] + (sumw[i2] - sumw[i1]) * x_m[1] * x_m[1] + (sumw[block_size] - sumw[i2]) * x_m[2] * x_m[2];
                    if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
                        scale = sumqx / sumq2; best_score = scale * sumqx;
                        besti1 = i1; besti2 = i2; best_shift = -1;
                    }
                }
            }
            assert(besti1 >= 0 && besti2 >= 0 && best_shift != 0);
            for (int j = 0; j < besti1; ++j) L[idx[2 * j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2 * j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2 * j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale; best_shift = -best_shift;
            }
            bool all_on_grid = true;
            const float* xx = best_shift == 1 ? x_p : x_m;
            for (int k = 0; k < block_size / 8; ++k) {
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8 * k + j] << 2 * j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t* neighbours = &kneighbors_q2xs[-kmap_q2xs[u] - 1];
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8 * k, weight + 8 * k, scale, xx, L + 8 * k, NGRID_IQ1S);
                    assert(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx = 0, sumq2 = 0;
                for (int k = 0; k < block_size / 8; ++k) {
                    const int8_t* pg = (const int8_t*)(&kgrid_q2xs[index[k]]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8 * k + j];
                        float q = xx[(pg[j] - 1) / 2];
                        sumqx += w * q * xb[8 * k + j];
                        sumq2 += w * q * q;
                    }
                }
                if (sumqx > 0 && sumq2 > 0) scale = sumqx / sumq2;
            }
            uint16_t h = 0;
            for (int k = 0; k < block_size / 8; ++k) {
                y[ibl].qs[(block_size / 8) * ib + k] = index[k] & 255;
                h |= (index[k] >> 8) << 3 * k;
            }
            y[ibl].qh[ib] = h;
            assert(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale / 15;
        y[ibl].d = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d * 1.125f)); // 1.125f is another fudge factor. Don't ask me why it is needed.
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / block_size; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(7, l));
            if (shifts[ib] == -1) l |= 8;
            y[ibl].qh[ib] |= (l << 12);
        }
    }
}

static constexpr size_t IQ1M_BLOCK_SIZE = 16;
static constexpr float GROUP_MAX_EPS_IQ1_M = 1e-7f;

static void quantize_row_iq1_m_impl(const float* x, block_iq1_m* y, int64_t n, const float* quant_weights,
    float* scales,
    float* weight,
    float* pairs,
    int8_t* L,
    uint16_t* index,
    int8_t* shifts) {

    const auto& kgrid_q2xs = table_iq1_grid;
    const auto& kmap_q2xs = table_iq1_map;
    const auto& kneighbors_q2xs = table_iq1_neighbors;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    assert(n % QK_K == 0);

    const int64_t nbl = n / QK_K;

    const int block_size = IQ1M_BLOCK_SIZE;

    const float x_p[3] = { -1 + IQ1M_DELTA,  IQ1M_DELTA, 1 + IQ1M_DELTA };
    const float x_m[3] = { -1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA };
    const uint8_t masks[4] = { 0x00, 0x80, 0x08, 0x88 };

    int* idx = (int*)(pairs + 1);

    float sumqx[4], sumq2[4];

    iq1m_scale_t s;
    const float* xx;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        memset(y[ibl].qs, 0, QK_K / 8);
        memset(y[ibl].qh, 0, QK_K / 16);
        memset(y[ibl].scales, 0, QK_K / 32);

        float max_scale = 0;

        const float* xbl = x + QK_K * ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = 2 * sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / block_size; ++ib) {
            const float* xb = xbl + block_size * ib;
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * ibl + block_size * ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            }
            else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i] * xb[i];
            }
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = std::max(max, fabsf(xb[i]));
            if (max < GROUP_MAX_EPS_IQ1_M) {
                scales[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2 * j] = xb[j];
                idx[2 * j] = j;
            }
            qsort(pairs, block_size, 2 * sizeof(float), iq1_sort_helper);
            float best_score = -FLT_MIN, scale = max;
            int besti1 = -1, besti2 = -1, best_k = -1;
            // 0: +, +
            // 1: +, -
            // 2: -, +
            // 3: -, -
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    memset(sumqx, 0, 4 * sizeof(float));
                    memset(sumq2, 0, 4 * sizeof(float));
                    for (int j = 0; j < i1; ++j) {
                        int i = idx[2 * j];
                        if (i < block_size / 2) {
                            sumqx[0] += weight[i] * x_p[0] * xb[i];
                            sumqx[1] += weight[i] * x_p[0] * xb[i];
                            sumqx[2] += weight[i] * x_m[0] * xb[i];
                            sumqx[3] += weight[i] * x_m[0] * xb[i];
                            sumq2[0] += weight[i] * x_p[0] * x_p[0];
                            sumq2[1] += weight[i] * x_p[0] * x_p[0];
                            sumq2[2] += weight[i] * x_m[0] * x_m[0];
                            sumq2[3] += weight[i] * x_m[0] * x_m[0];
                        }
                        else {
                            sumqx[0] += weight[i] * x_p[0] * xb[i];
                            sumqx[2] += weight[i] * x_p[0] * xb[i];
                            sumqx[1] += weight[i] * x_m[0] * xb[i];
                            sumqx[3] += weight[i] * x_m[0] * xb[i];
                            sumq2[0] += weight[i] * x_p[0] * x_p[0];
                            sumq2[2] += weight[i] * x_p[0] * x_p[0];
                            sumq2[1] += weight[i] * x_m[0] * x_m[0];
                            sumq2[3] += weight[i] * x_m[0] * x_m[0];
                        }
                    }
                    for (int j = i1; j < i2; ++j) {
                        int i = idx[2 * j];
                        if (i < block_size / 2) {
                            sumqx[0] += weight[i] * x_p[1] * xb[i];
                            sumqx[1] += weight[i] * x_p[1] * xb[i];
                            sumqx[2] += weight[i] * x_m[1] * xb[i];
                            sumqx[3] += weight[i] * x_m[1] * xb[i];
                            sumq2[0] += weight[i] * x_p[1] * x_p[1];
                            sumq2[1] += weight[i] * x_p[1] * x_p[1];
                            sumq2[2] += weight[i] * x_m[1] * x_m[1];
                            sumq2[3] += weight[i] * x_m[1] * x_m[1];
                        }
                        else {
                            sumqx[0] += weight[i] * x_p[1] * xb[i];
                            sumqx[2] += weight[i] * x_p[1] * xb[i];
                            sumqx[1] += weight[i] * x_m[1] * xb[i];
                            sumqx[3] += weight[i] * x_m[1] * xb[i];
                            sumq2[0] += weight[i] * x_p[1] * x_p[1];
                            sumq2[2] += weight[i] * x_p[1] * x_p[1];
                            sumq2[1] += weight[i] * x_m[1] * x_m[1];
                            sumq2[3] += weight[i] * x_m[1] * x_m[1];
                        }
                    }
                    for (int j = i2; j < block_size; ++j) {
                        int i = idx[2 * j];
                        if (i < block_size / 2) {
                            sumqx[0] += weight[i] * x_p[2] * xb[i];
                            sumqx[1] += weight[i] * x_p[2] * xb[i];
                            sumqx[2] += weight[i] * x_m[2] * xb[i];
                            sumqx[3] += weight[i] * x_m[2] * xb[i];
                            sumq2[0] += weight[i] * x_p[2] * x_p[2];
                            sumq2[1] += weight[i] * x_p[2] * x_p[2];
                            sumq2[2] += weight[i] * x_m[2] * x_m[2];
                            sumq2[3] += weight[i] * x_m[2] * x_m[2];
                        }
                        else {
                            sumqx[0] += weight[i] * x_p[2] * xb[i];
                            sumqx[2] += weight[i] * x_p[2] * xb[i];
                            sumqx[1] += weight[i] * x_m[2] * xb[i];
                            sumqx[3] += weight[i] * x_m[2] * xb[i];
                            sumq2[0] += weight[i] * x_p[2] * x_p[2];
                            sumq2[2] += weight[i] * x_p[2] * x_p[2];
                            sumq2[1] += weight[i] * x_m[2] * x_m[2];
                            sumq2[3] += weight[i] * x_m[2] * x_m[2];
                        }
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (sumq2[k] > 0 && sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
                            scale = sumqx[k] / sumq2[k]; best_score = scale * sumqx[k];
                            besti1 = i1; besti2 = i2; best_k = k;
                        }
                    }
                }
            }
            assert(besti1 >= 0 && besti2 >= 0 && best_k >= 0);
            for (int j = 0; j < besti1; ++j) L[idx[2 * j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2 * j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2 * j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale;
                best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
            }
            bool all_on_grid = true;
            for (int k = 0; k < block_size / 8; ++k) {
                if (k == 0) xx = best_k < 2 ? x_p : x_m;
                else xx = best_k % 2 == 0 ? x_p : x_m;
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8 * k + j] << 2 * j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t* neighbours = &kneighbors_q2xs[-kmap_q2xs[u] - 1];
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8 * k, weight + 8 * k, scale, xx, L + 8 * k, NGRID_IQ1S);
                    assert(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx_f = 0, sumq2_f = 0;
                for (int k = 0; k < block_size / 8; ++k) {
                    if (k == 0) xx = best_k < 2 ? x_p : x_m;
                    else xx = best_k % 2 == 0 ? x_p : x_m;
                    const int8_t* pg = (const int8_t*)(&kgrid_q2xs[index[k]]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8 * k + j];
                        float q = xx[(pg[j] - 1) / 2];
                        sumqx_f += w * q * xb[8 * k + j];
                        sumq2_f += w * q * q;
                    }
                }
                if (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f / sumq2_f;
            }
            y[ibl].qs[2 * ib + 0] = index[0] & 255;
            y[ibl].qs[2 * ib + 1] = index[1] & 255;
            y[ibl].qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
            assert(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_k;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        uint16_t* sc = (uint16_t*)y[ibl].scales;
        float d = max_scale / 15;
        float id = 1 / d;
        float sumqx_f = 0, sumq2_f = 0;
        for (int ib = 0; ib < QK_K / block_size; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib + 0] - 1));
            l = std::max(0, std::min(7, l));
            sc[ib / 4] |= (l << 3 * (ib % 4));
            y[ibl].qh[ib] |= masks[shifts[ib]];
            const float* xb = xbl + block_size * ib;
            if (quant_weights) {
                const float* qw = quant_weights + QK_K * ibl + block_size * ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            }
            else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i] * xb[i];
            }
            for (int k = 0; k < block_size / 8; ++k) {
                if (k == 0) xx = shifts[ib] < 2 ? x_p : x_m;
                else xx = shifts[ib] % 2 == 0 ? x_p : x_m;
                const size_t offset = y[ibl].qs[2 * ib + k] + ((y[ibl].qh[ib] << (8 - 4 * k)) & 0x700);
                const int8_t* pg = (const int8_t*)(&kgrid_q2xs[offset]);
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8 * k + j];
                    float q = xx[(pg[j] - 1) / 2] * (2 * l + 1);
                    sumqx_f += w * q * xb[8 * k + j];
                    sumq2_f += w * q * q;
                }
            }
        }
        if (sumq2_f > 0) d = sumqx_f / sumq2_f;
        s = std::bit_cast<uint16_t>(fromFloat32<ggml_fp16_t>(d * 1.1125f)); // 1.1125f is another fudge factor. Don't ask me why it is needed.
        sc[0] |= ((s & 0x000f) << 12);
        sc[1] |= ((s & 0x00f0) << 8);
        sc[2] |= ((s & 0x0f00) << 4);
        sc[3] |= ((s & 0xf000) << 0);
    }
}

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

size_t quantize_q4_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights)
{
    if (!quant_weights) {
        quantize_row_q4_0_ref(src, static_cast<block_q4_0 *>(dst), (int64_t)nrow * n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q4_0_impl(src, (block_q4_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_q4_1(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights)
{
    if (!quant_weights) {
        quantize_row_q4_1_ref(src, static_cast<block_q4_1*>(dst), (int64_t)nrow * n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q4_1, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_1, n_per_row);
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q4_1_impl(src, (block_q4_1*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_mxfp4(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*) {
    quantize_row_mxfp4_ref(src, static_cast<block_mxfp4*>(dst), (int64_t)nrow * n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
}

size_t quantize_q5_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights)
{
    if (!quant_weights) {
        quantize_row_q5_0_ref(src, static_cast<block_q5_0*>(dst), (int64_t)nrow * n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q5_0_impl(src, (block_q5_0*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_q5_1(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights)
{
    if (!quant_weights) {
        quantize_row_q5_1_ref(src, static_cast<block_q5_1*>(dst), (int64_t)nrow * n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q5_1, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_1, n_per_row);
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q5_1_impl(src, (block_q5_1*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_q8_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*)
{
    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    quantize_row_q8_0_ref(src, static_cast<block_q8_0*>(dst), (int64_t)nrow * n_per_row);
    return nrow * row_size;
}

size_t quantize_q2_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q2_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q2_K_ref(src, static_cast<block_q2_K*>(dst), (int64_t)nrow * n_per_row);
    }
    else {
        char* qrow = (char*)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q2_K_impl(src, (block_q2_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

size_t quantize_q3_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q3_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q3_K_ref(src, (block_q3_K*)dst, (int64_t)nrow * n_per_row);
    }
    else {
        char* qrow = (char*)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q3_K_impl(src, (block_q3_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

size_t quantize_q4_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q4_K_ref(src, (block_q4_K*)dst, (int64_t)nrow * n_per_row);
    }
    else {
        char* qrow = (char*)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q4_K_impl(src, (block_q4_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

size_t quantize_q5_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q5_K_ref(src, (block_q5_K*)dst, (int64_t)nrow * n_per_row);
    }
    else {
        char* qrow = (char*)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q5_K_impl(src, (block_q5_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

size_t quantize_q6_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
    if (!quant_weights) {
        quantize_row_q6_K_ref(src, (block_q6_K*)dst, (int64_t)nrow * n_per_row);
    }
    else {
        char* qrow = (char*)dst;
        for (int64_t row = 0; row < nrow; ++row) {
            quantize_row_q6_K_impl(src, (block_q6_K*)qrow, n_per_row, quant_weights);
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrow * row_size;
}

size_t quantize_tq1_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*) {
    const size_t row_size = ggml_row_size(GGML_TYPE_TQ1_0, n_per_row);
    quantize_row_tq1_0_ref(src, (block_tq1_0*)dst, (int64_t)nrow * n_per_row);
    return nrow * row_size;
}

size_t quantize_tq2_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*) {
    const size_t row_size = ggml_row_size(GGML_TYPE_TQ2_0, n_per_row);
    quantize_row_tq2_0_ref(src, (block_tq2_0*)dst, (int64_t)nrow * n_per_row);
    return nrow * row_size;
}

size_t quantize_iq2_xxs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xxs_impl(src, (block_iq2_xxs *)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq2_xxs);
    }
    return nrow * nblock * sizeof(block_iq2_xxs);
}

size_t quantize_iq2_xs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_xs_impl(src, (block_iq2_xs*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq2_xs);
    }
    return nrow * nblock * sizeof(block_iq2_xs);
}

size_t quantize_iq2_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_s_impl(src, (block_iq2_s*)qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq2_s);
    }
    return nrow * nblock * sizeof(block_iq2_s);
}

size_t quantize_iq3_xxs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_xxs_impl(256, src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq3_xxs);
    }
    return nrow * nblock * sizeof(block_iq3_xxs);
}

static constexpr size_t IQ3S_BLOCK_SIZE = 32;

size_t quantize_iq3_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    int64_t nblock = n_per_row / QK_K;
    float scales[QK_K / IQ3S_BLOCK_SIZE];
    float weight[IQ3S_BLOCK_SIZE];
    float xval[IQ3S_BLOCK_SIZE];
    int8_t L[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    float  waux[IQ3S_BLOCK_SIZE];
    bool   is_on_grid[IQ3S_BLOCK_SIZE / 4];
    bool   is_on_grid_aux[IQ3S_BLOCK_SIZE / 4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE / 8];
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_s_impl(IQ3S_BLOCK_SIZE, src, (block_iq3_s *)qrow, n_per_row, quant_weights,
            scales, weight, xval, L, Laux, waux, is_on_grid, is_on_grid_aux, block_signs);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq3_s);
    }
    return nrow * nblock * sizeof(block_iq3_s);
}

size_t quantize_iq1_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    float  scales[QK_K / IQ1S_BLOCK_SIZE];
    float  weight[IQ1S_BLOCK_SIZE];
    int8_t L[IQ1S_BLOCK_SIZE];
    float  sumx[IQ1S_BLOCK_SIZE + 1];
    float  sumw[IQ1S_BLOCK_SIZE + 1];
    float  pairs[2 * IQ1S_BLOCK_SIZE];
    uint16_t index[IQ1S_BLOCK_SIZE / 8];
    int8_t shifts[QK_K / IQ1S_BLOCK_SIZE];
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_s_impl(src, (block_iq1_s*)qrow, n_per_row, quant_weights, scales, weight, sumx, sumw, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq1_s);
    }
    return nrow * nblock * sizeof(block_iq1_s);
}

size_t quantize_iq1_m(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    float  scales[QK_K / IQ1M_BLOCK_SIZE];
    float  weight[IQ1M_BLOCK_SIZE];
    int8_t L[IQ1M_BLOCK_SIZE];
    float  pairs[2 * IQ1M_BLOCK_SIZE];
    uint16_t index[IQ1M_BLOCK_SIZE / 8];
    int8_t shifts[QK_K / IQ1M_BLOCK_SIZE];
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_m_impl(src, (block_iq1_m*)qrow, n_per_row, quant_weights, scales, weight, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock * sizeof(block_iq1_m);
    }
    return nrow * nblock * sizeof(block_iq1_m);
}

size_t quantize_iq4_nl(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    static constexpr size_t QK4_NL = block_iq4_nl::block_size;
    assert(n_per_row % QK4_NL == 0);
    int64_t nblock = n_per_row / QK4_NL;
    char* qrow = (char*)dst;
    uint8_t L[QK4_NL];
    float weight[QK4_NL];
    uint16_t unused_h;
    uint8_t* unused_l = NULL;
    float scale;
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_nl* iq4 = (block_iq4_nl*)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float* qw = quant_weights ? quant_weights + QK4_NL * ibl : NULL;
            quantize_row_iq4_nl_impl(QK4_NL, 32, src + QK4_NL * ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                &scale, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;
        qrow += nblock * sizeof(block_iq4_nl);
    }
    return nrow * nblock * sizeof(block_iq4_nl);
}

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

size_t quantize_iq4_xs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights) {
    assert(n_per_row % QK_K == 0);
    int64_t nblock = n_per_row / QK_K;
    char* qrow = (char*)dst;
    uint8_t L[QK_K];
    float weight[32];
    float scales[QK_K / 32];
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_xs* iq4 = (block_iq4_xs*)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float* qw = quant_weights ? quant_weights + QK_K * ibl : NULL;
            quantize_row_iq4_nl_impl(QK_K, 32, src + QK_K * ibl, &iq4[ibl].d, iq4[ibl].qs, &iq4[ibl].scales_h, iq4[ibl].scales_l,
                scales, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;
        qrow += nblock * sizeof(block_iq4_xs);
    }
    return nrow * nblock * sizeof(block_iq4_xs);
}

void quantize_row_q4_0_ref(const float* x, block_q4_0* y, int64_t k) {
    static const int qk = block_q4_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = castToUint16(d);

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + 0 + j] * id;
            const float x1 = x[i * qk + qk / 2 + j] * id;

            const uint8_t xi0 = std::min<uint8_t>(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = std::min<uint8_t>(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_1_ref(const float* x, block_q4_1* y, int64_t k)
{
    const int qk = block_q4_1::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].dm = compress(d, min);

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = (x[i * qk + 0 + j] - min) * id;
            const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

            const uint8_t xi0 = std::min<uint8_t>(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = std::min<uint8_t>(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

static inline int best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0] * e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i] * e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}

void quantize_row_mxfp4_ref(const float* x, block_mxfp4* y, int64_t k) {
    static const int qk = block_mxfp4::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];

            if (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2 + 127) : 0;

        const float d = ggml_e8m0_to_fp32_half(e);

        y[i].e = e;

        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t x0 = best_index_mxfp4(x[i * qk + 0 + j], d);
            const uint8_t x1 = best_index_mxfp4(x[i * qk + qk / 2 + j], d);

            y[i].qs[j] = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

void quantize_row_q5_0_ref(const float* x, block_q5_0* y, int64_t k) {
    static const int qk = block_q5_0::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -16;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = castToUint16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + 0 + j] * id;
            const float x1 = x[i * qk + qk / 2 + j] * id;

            const uint8_t xi0 = std::min<uint8_t>(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = std::min<uint8_t>(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk / 2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

void quantize_row_q5_1_ref(const float* x, block_q5_1* y, int64_t k) {
    const int qk = block_q5_1::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].dm = compress(d, min);

        uint32_t qh = 0;

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = (x[i * qk + 0 + j] - min) * id;
            const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk / 2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int64_t k) {
    const int QK8_0 = block_q8_0::block_size;
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i * QK8_0 + j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = castToUint16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i * QK8_0 + j] * id;

            y[i].qs[j] = roundf(x0);
        }
    }
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

void quantize_row_q2_K_ref(const float* x, block_q2_K* y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16 * j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        std::array<ggml_fp16_t, 2> y_dm;
        if (max_scale > 0) {
            float iscale = q4scale / max_scale;
            for (int j = 0; j < QK_K / 16; ++j) {
                int l = nearest_int(iscale * scales[j]);
                y[i].scales[j] = l;
            }
            y_dm[0] = fromFloat32<ggml_fp16_t>(max_scale / q4scale);
        }
        else {
            for (int j = 0; j < QK_K / 16; ++j) y[i].scales[j] = 0;
            y_dm[0] = fromFloat32<ggml_fp16_t>(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale / max_min;
            for (int j = 0; j < QK_K / 16; ++j) {
                int l = nearest_int(iscale * mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y_dm[1] = fromFloat32<ggml_fp16_t>(max_min / q4scale);
        }
        else {
            y_dm[1] = fromFloat32<ggml_fp16_t>(0.f);
        }
        y[i].dm = std::bit_cast<uint32_t>(y_dm);
        for (int j = 0; j < QK_K / 16; ++j) {
            const float d = toFloat32(y_dm[0]) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = toFloat32(y_dm[1]) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16 * j + ii] + dm) / d);
                l = std::max(0, std::min(3, l));
                L[16 * j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void quantize_row_q3_K_ref(const float* x, block_q3_K* y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        ggml_fp32_t y_d;
        if (max_scale) {
            float iscale = -32.f / max_scale;
            for (int j = 0; j < QK_K / 16; ++j) {
                int8_t l = nearest_int(iscale * scales[j]);
                l = std::max<int8_t>(-32, std::min<int8_t>(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                }
                else {
                    y[i].scales[j - 8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
            }
            y_d = 1 / iscale;
        }
        else {
            y_d = 0.f;
        }
        y[i].d = castToUint16(y_d);

        int8_t sc;
        for (int j = 0; j < QK_K / 16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
            sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            float d = y_d * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16 * j + ii] / d);
                l = std::max(-4, std::min(3, l));
                L[16 * j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K / 8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K / 8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void quantize_row_q4_K_ref(const float* x, block_q4_K* y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = nearest_int(inv_scale * scales[j]);
            uint8_t lm = nearest_int(inv_min * mins[j]);
            ls = std::min<uint8_t>(63, ls);
            lm = std::min<uint8_t>(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            }
            else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        max_scale /= 63.f;
        max_min /= 63.f;
        y[i].dm = compress(max_scale, max_min);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = max_scale * sc;
            if (!d) continue;
            const float dm = max_min * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = std::max(0, std::min(15, l));
                L[32 * j + ii] = l;
            }
        }

        uint8_t* q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

void quantize_row_q5_K_ref(const float* x, block_q5_K* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    float weights[32];
    uint8_t Laux[32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = nearest_int(inv_scale * scales[j]);
            uint8_t lm = nearest_int(inv_min * mins[j]);
            ls = std::min<uint8_t>(63, ls);
            lm = std::min<uint8_t>(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            }
            else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        max_scale /= 63.f;
        max_min /= 63.f;
        y[i].dm = compress(max_scale, max_min);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = max_scale * sc;
            if (!d) continue;
            const float dm = max_min * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = std::max(0, std::min(31, l));
                L[32 * j + ii] = l;
            }
        }

        uint8_t* qh = y[i].qh;
        uint8_t* ql = y[i].qs;
        memset(qh, 0, QK_K / 8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

void quantize_row_q6_K_ref(const float* x, block_q6_K* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K / 16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = castToUint16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f / max_scale;
        y[i].d = castToUint16(1 / iscale);
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            y[i].scales[ib] = std::min(127, nearest_int(iscale * scales[ib]));
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            float d = toFloat32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16 * j + ii] / d);
                l = std::max(-32, std::min(31, l));
                L[16 * j + ii] = l + 32;
            }
        }

        uint8_t* ql = y[i].ql;
        uint8_t* qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l + 0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l + 0] = q1 | (q3 << 4);
                ql[l + 32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
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

void quantize_row_tq1_0_ref(const float* x, block_tq1_0* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = castToUint16(d);

        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n * 32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5 * 32;
        }
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n * 16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5 * 16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[j + m * sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4 * sizeof(y->qh);
    }
}

void quantize_row_tq2_0_ref(const float* x, block_tq2_0* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = castToUint16(d);

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lroundf(x[m + n * 32] * id) + 1;
                    q += (xi & 3) << (2 * n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4 * 32;
        }
    }
}

void quantize_row(const float* x, block_q4_0* y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row(const float* x, block_q4_1* y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row(const float* x, block_mxfp4* y, int64_t k) {
    quantize_row_mxfp4_ref(x, y, k);
}

void quantize_row(const float* x, block_q5_0* y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row(const float* x, block_q5_1* y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row(const float* x, block_q8_0* y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

void quantize_row(const float* x, block_q8_1* y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

void quantize_row(const float* x, block_q2_K* y, int64_t k) {
    quantize_row_q2_K_ref(x, y, k);
}

void quantize_row(const float* x, block_q3_K* y, int64_t k) {
	quantize_row_q3_K_ref(x, y, k);
}

void quantize_row(const float* x, block_q4_K* y, int64_t k) {
	quantize_row_q4_K_ref(x, y, k);
}

void quantize_row(const float* x, block_q5_K* y, int64_t k) {
	quantize_row_q5_K_ref(x, y, k);
}

void quantize_row(const float* x, block_q6_K* y, int64_t k) {
    quantize_row_q6_K_ref(x, y, k);
}

void quantize_row(const float* x, block_q8_K* y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

void quantize_row(const float* x, block_tq1_0* y, int64_t k) {
	quantize_row_tq1_0_ref(x, y, k);
}

void quantize_row(const float* x, block_tq2_0* y, int64_t k) {
	quantize_row_tq2_0_ref(x, y, k);
}

void quantize_row(const float* x, block_iq4_nl* y, int64_t k) {
    //assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row(const float* x, block_iq4_xs* y, int64_t k) {
    //assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}

// AAAAAAAAAAAAAAAAAAA

void dequantize_row(const block_q4_0* x, float* y, int64_t k)
{
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

void dequantize_row(const block_mxfp4* x, float* y, int64_t k) {
    static const int qk = block_mxfp4::block_size;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_e8m0_to_fp32_half(x[i].e);

        for (int j = 0; j < qk / 2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >> 4];

            y[i * qk + j + 0] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
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

void dequantize_row(const block_q8_0* x, float* y, int64_t k)
{
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

void dequantize_row(const block_q2_K* x, float* y, int64_t k) 
{
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float min = toFloat32(dm[1]);

        const uint8_t* q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row(const block_q3_K* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

void dequantize_row(const block_q4_K* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float min = toFloat32(dm[1]);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_row(const block_q5_K* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const auto dm = std::bit_cast<std::array<ggml_fp16_t, 2>>(x[i].dm);
        const float d = toFloat32(dm[0]);
        const float min = toFloat32(dm[1]);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row(const block_q6_K* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row(const block_tq1_0* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };

    for (int64_t i = 0; i < nb; ++i) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t)q * 3) >> 8;
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }

        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t)q * 3) >> 8;
                *y++ = (float)(xi - 1) * d;
            }
        }
    }
}

void dequantize_row(const block_tq2_0* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l * 2)) & 3;
                    *y++ = (float)(q - 1) * d;
                }
            }
        }
    }
}

void dequantize_row(const block_iq2_xxs* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row(const block_iq2_xs* x, float* y, int64_t k) 
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (x[i].qs[4 * ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row(const block_iq2_s* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = qs + QK_K / 8;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

void dequantize_row(const block_iq3_xxs* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32;

    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + QK_K / 4;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

void dequantize_row(const block_iq3_s* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

void dequantize_row(const block_iq1_s* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));
        const uint8_t* qs = x[i].qs;
        const uint16_t* qh = x[i].qh;

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

void dequantize_row(const block_iq1_m* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float delta[4];
    uint16_t idx[4];

    for (int i = 0; i < nb; i++) {

        const uint16_t* sc = (const uint16_t*)x[i].scales;
        iq1m_scale_t scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(scale));

        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);

            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

void dequantize_row(const block_iq4_nl* x, float* y, int64_t k)
{
    static constexpr size_t QK4_NL = block_iq4_nl::block_size;
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t* qs = x[i].qs;

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j + 0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j + QK4_NL / 2] = d * kvalues_iq4nl[qs[j] >> 4];
        }
        y += QK4_NL;
        qs += QK4_NL / 2;
    }
}

void dequantize_row(const block_iq4_xs* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t* qs = x[i].qs;

        const float d = toFloat32(std::bit_cast<ggml_fp16_t>(x[i].d));

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}