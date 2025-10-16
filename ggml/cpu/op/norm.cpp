module;
#include <assert.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include "../helper.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

template <bool isRms>
static void ggml_compute_forward_norm_f32(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    float eps = std::bit_cast<float>(dst->op_params[0]);
    GGML_ASSERT(eps >= 0.0f);
    auto y = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
    auto x = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

    for (int64_t i03 = 0; i03 < x.extent(0); i03++) {
        for (int64_t i02 = 0; i02 < x.extent(1); i02++) {
            for (int64_t i01 = 0; i01 < x.extent(2); i01++) {
                double sum = [=] {
                    double sum = 0.0;
                    for (int64_t i00 = 0; i00 < x.extent(3); i00++) {
                        if constexpr (isRms) {
                            sum += (double)(x[i03, i02, i01, i00] * x[i03, i02, i01, i00]);
                        }
                        else {
                            sum += (double)x[i03, i02, i01, i00];
                        }
                    }
                    return sum;
                }();

                float mean = sum / x.extent(3);
                const float scale = [&] {
                    if constexpr (isRms) {
                        for (int64_t i00 = 0; i00 < x.extent(3); i00++)
                            y[i03, i02, i01, i00] = x[i03, i02, i01, i00];
                        return 1.0f / sqrtf(mean + eps);
                    }
                    else {
                        double sum = 0.0;
                        for (int64_t i00 = 0; i00 < x.extent(3); i00++) {
                            float v = x[i03, i02, i01, i00] - mean;
                            y[i03, i02, i01, i00] = v;
                            sum += (double)(v * v);
                        }
                        float variance = sum / x.extent(3);
                        return 1.0f / sqrtf(variance + eps);
                    }
                }();

                if constexpr (isRms) {
                    // if you hit this, likely you got an inf somewhere earlier
                    assert(scale > 0.0f);
                }

                for (int64_t i00 = 0; i00 < x.extent(3); i00++)
                    y[i03, i02, i01, i00] *= scale;
            }
        }
    }
}

void ggml_compute_forward_norm(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32<false>(dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}

void ggml_compute_forward_rms_norm(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32<true>(dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}

static void ggml_compute_forward_group_norm_f32(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    // TODO: optimize
    const float eps = std::bit_cast<float>(dst->op_params[1]);

    int n_channels = src0->ne[2];
    const int n_groups = std::bit_cast<int>(dst->op_params[0]);
    int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;

    auto y = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
    auto x = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

    for (int i = 0; i < n_groups; i++) {
        int start = i * n_channels_per_group;
        int end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        int step = end - start;

        for (int64_t i03 = 0; i03 < x.extent(0); i03++) {
            double sum = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < x.extent(2); i01++) {
                    double sumr = 0.0;
                    for (int64_t i00 = 0; i00 < x.extent(3); i00++) {
                        sumr += (double)x[i03, i02, i01, i00];
                    }
                    sum += sumr;
                }
            }
            const float mean = sum / (x.extent(3) * x.extent(2) * step);

            double sum2 = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < x.extent(2); i01++) {
                    double sumr = 0.0;
                    for (int64_t i00 = 0; i00 < x.extent(3); i00++) {
                        float v = x[i03, i02, i01, i00] - mean;
                        y[i03, i02, i01, i00] = v;
                        sumr += (double)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            const float variance = sum2 / (x.extent(3) * x.extent(2) * step);
            const float scale = 1.0f / sqrtf(variance + eps);

            for (int64_t i02 = start; i02 < end; i02++)
                for (int64_t i01 = 0; i01 < x.extent(2); i01++)
                    for (int i00 = 0; i00 < x.extent(3); i00++)
                        y[i03, i02, i01, i00] *= scale;
        }
    }
}

void ggml_compute_forward_group_norm(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_group_norm_f32(dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}
