module;
#include <assert.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include "mdspan_helper.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
enum ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP, bool isRms>
static void ggml_compute_forward_norm_f32(
    ggml_tensor* dst_rms_norm,
    ggml_tensor* dst_fused = nullptr) {
    const ggml_tensor* src0 = dst_rms_norm->src[0];
    const ggml_tensor* src1 = nullptr;
    ggml_tensor* dst = dst_rms_norm;

    if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    float eps = std::bit_cast<float>(dst_rms_norm->op_params[0]);
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

                for (int64_t i00 = 0; i00 < x.extent(3); i00++) {
                    y[i03, i02, i01, i00] *= scale;
                    if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL)
                        y[i03, i02, i01, i00] *= x[i03, i02, i01, i00];
                }
            }
        }
    }
}

void ggml_compute_forward_norm(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE, false>(dst);
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
        ggml_compute_forward_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE, true>(dst);
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

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
void ggml_compute_forward_rms_norm_mul_fused(
    ggml_tensor* dst_rms_norm,
    ggml_tensor* dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    const ggml_tensor* src0 = dst_rms_norm->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL, true>(dst_rms_norm, dst_mul);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}