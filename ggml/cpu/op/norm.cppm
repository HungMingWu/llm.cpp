module;
#include <assert.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <bit>
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml:cpu.op.norm;
import :ds;
import :tensor;
import :cpu.ds;

inline static void ggml_vec_scale_f32(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

template <bool isRms>
static void ggml_compute_forward_norm_f32(
    const ggml_compute_params* params,
    ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    float eps = std::bit_cast<float>(dst->op_params[0]);
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
        for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
            for (int64_t i01 = ith; i01 < src0->ne[1]; i01 += nth) {
                const float* x = (float*)((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                double sum = [=] {
                    double sum = 0.0;
                    for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
                        if constexpr (isRms) {
                            sum += (double)(x[i00] * x[i00]);
                        }
                        else {
                            sum += (double)x[i00];
                        }
                    }
                    return sum;
                }();

                float mean = sum / src0->ne[0];

                float* y = (float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]);

                const float scale = [&] {
                    if constexpr (isRms) {
                        memcpy(y, x, src0->ne[0] * sizeof(float));
                        return 1.0f / sqrtf(mean + eps);
                    }
                    else {
                        double sum = 0.0;
                        for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
                            float v = x[i00] - mean;
                            y[i00] = v;
                            sum += (double)(v * v);
                        }
                        float variance = sum / src0->ne[0];
                        return 1.0f / sqrtf(variance + eps);
                    }
                }();

                if constexpr (isRms) {
                    // if you hit this, likely you got an inf somewhere earlier
                    assert(scale > 0.0f);
                }

                ggml_vec_scale_f32(src0->ne[0], y, scale);
            }
        }
    }
}

void ggml_compute_forward_norm(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32<false>(params, dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}

void ggml_compute_forward_rms_norm(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32<true>(params, dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}

static void ggml_vec_cpy_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }
static void ggml_vec_acc_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] += x[i]; }

static void ggml_compute_forward_rms_norm_back_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0]; // gradients from forward pass output
    const struct ggml_tensor* src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    float eps = std::bit_cast<float>(dst->op_params[0]);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
        for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
            for (int64_t i01 = ith; i01 < src0->ne[1]; i01 += nth) {
                // src1 is same shape as src0 => same indices
                const int64_t i11 = i01;
                const int64_t i12 = i02;
                const int64_t i13 = i03;

                const float* dz = (float*)((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                const float* x = (float*)((char*)src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);

                double sum_xx = 0.0;
                double sum_xdz = 0.0;

                for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
                    sum_xx += (double)(x[i00] * x[i00]);
                    sum_xdz += (double)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/src0->ne[0];
                const float mean_eps = (float)(sum_xx) / src0->ne[0] + eps;
                const float sum_eps = (float)(sum_xx)+eps * src0->ne[0];
                //const float mean_xdz = (float)(sum_xdz)/src0->ne[0];
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                const float rrms = 1.0f / sqrtf(mean_eps);
                //const float scale    = -rrms/(src0->ne[0] * mean_eps); // -1/(n*rms**3)

                {
                    // z = rms_norm(x)
                    //
                    // rms_norm(src1) =
                    //     scale(
                    //         src1,
                    //         div(
                    //             1,
                    //             sqrt(
                    //                 add(
                    //                     scale(
                    //                         sum(
                    //                             sqr(
                    //                                 src1)),
                    //                         (1.0/N)),
                    //                     eps))));

                    // postorder:
                    // ## op    args         grad
                    // 00 param src1         grad[#00]
                    // 01 const 1
                    // 02 sqr   (#00)        grad[#02]
                    // 03 sum   (#02)        grad[#03]
                    // 04 const 1/N
                    // 05 scale (#03, #04)   grad[#05]
                    // 06 const eps
                    // 07 add   (#05, #06)   grad[#07]
                    // 08 sqrt  (#07)        grad[#08]
                    // 09 div   (#01,#08)    grad[#09]
                    // 10 scale (#00,#09)    grad[#10]
                    //
                    // backward pass, given grad[#10]
                    // #10: scale
                    // grad[#00] += scale(grad[#10],#09)
                    // grad[#09] += sum(mul(grad[#10],#00))
                    // #09: div
                    // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
                    // #08: sqrt
                    // grad[#07] += mul(grad[#08], div(0.5, #08))
                    // #07: add
                    // grad[#05] += grad[#07]
                    // #05: scale
                    // grad[#03] += scale(grad[#05],#04)
                    // #03: sum
                    // grad[#02] += repeat(grad[#03], #02)
                    // #02:
                    // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
                    //
                    // substitute and simplify:
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#02] = repeat(grad[#03], #02)
                    // grad[#02] = repeat(scale(grad[#05],#04), #02)
                    // grad[#02] = repeat(scale(grad[#07],#04), #02)
                    // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N))), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum(mul(dz,x)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum_xdz * div(-1,rms*N*mean_eps))
                    // a = b*c + d*e
                    // a = b*c*f/f + d*e*f/f
                    // a = (b*c*f + d*e*f)*(1/f)
                    // a = (b*c*(1/c) + d*e*(1/c))*(1/(1/c))
                    // a = (b + d*e/c)*c
                    // b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps)
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
                    // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
                    // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
                    // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
                    // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
                    // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
                    // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                    // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                }
                // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                // post-order:
                // dx := x
                // dx := scale(dx,-mean_xdz/mean_eps)
                // dx := add(dx, dz)
                // dx := scale(dx, rrms)
                float* dx = (float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]);

                // dx[i00] = (x*(-sum_xdz/sum_eps) + dz) / sqrtf(mean_eps)
                ggml_vec_cpy_f32(src0->ne[0], dx, x);
                // ggml_vec_scale_f32(src0->ne[0], dx, -mean_xdz/mean_eps);
                ggml_vec_scale_f32(src0->ne[0], dx, (float)(-sum_xdz) / sum_eps);
                ggml_vec_acc_f32(src0->ne[0], dx, dz);
                ggml_vec_scale_f32(src0->ne[0], dx, rrms);
            }
        }
    }
}

void ggml_compute_forward_rms_norm_back(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_rms_norm_back_f32(params, dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}

static void ggml_compute_forward_group_norm_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    // TODO: optimize
    const float eps = std::bit_cast<float>(dst->op_params[1]);

    int n_channels = src0->ne[2];
    const int n_groups = std::bit_cast<int>(dst->op_params[0]);
    int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    for (int i = ith; i < n_groups; i += nth) {
        int start = i * n_channels_per_group;
        int end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        int step = end - start;

        for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
            double sum = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
                    const float* x = (float*)((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);

                    double sumr = 0.0;
                    for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
                        sumr += (double)x[i00];
                    }
                    sum += sumr;
                }
            }
            const float mean = sum / (src0->ne[0] * src0->ne[1] * step);

            double sum2 = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
                    const float* x = (float*)((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);

                    float* y = (float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]);

                    double sumr = 0.0;
                    for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (double)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            const float variance = sum2 / (src0->ne[0] * src0->ne[1] * step);
            const float scale = 1.0f / sqrtf(variance + eps);

            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
                    float* y = (float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]);
                    ggml_vec_scale_f32(src0->ne[0], y, scale);
                }
            }
        }
    }
}

void ggml_compute_forward_group_norm(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_group_norm_f32(params, dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}
