module;
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "mdspan.hpp"
#include <algorithm>
#include <bit>
#include "helper.h"

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :ds;
import :tensor;
import :cpu.ds;
import :cpu.op;

static void ggml_compute_forward_upscale_f32(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    float sf0 = (float)dst->ne[0] / src0->ne[0];
    float sf1 = (float)dst->ne[1] / src0->ne[1];
    float sf2 = (float)dst->ne[2] / src0->ne[2];
    float sf3 = (float)dst->ne[3] / src0->ne[3];
    float pixel_offset = 0.5f;

    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    auto dst_mdspan = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
    auto src0_mdspan = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = (dst->ne[0] > 1 && src0->ne[0] > 1) ? (float)(dst->ne[0] - 1) / (src0->ne[0] - 1) : sf0;
        sf1 = (dst->ne[1] > 1 && src0->ne[1] > 1) ? (float)(dst->ne[1] - 1) / (src0->ne[1] - 1) : sf1;
    }

    if (mode == GGML_SCALE_MODE_NEAREST) {
        for (int64_t i3 = 0; i3 < dst_mdspan.extent(0); i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = 0; i2 < dst_mdspan.extent(1); i2++) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < dst_mdspan.extent(2); i1++) {
                    const int64_t i01 = i1 / sf1;
                    for (int64_t i0 = 0; i0 < dst_mdspan.extent(3); i0++) {
                        const int64_t i00 = i0 / sf0;
                        dst_mdspan[i3, i2, i1, i0] = src0_mdspan[i03, i02, i01, i00];
                    }
                }
            }
        }
    } else if (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](float x) -> float {
            return std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        const float support1 = std::max(1.0f, 1.0f / sf1);
        const float invscale1 = 1.0f / support1;
        const float support0 = std::max(1.0f, 1.0f / sf0);
        const float invscale0 = 1.0f / support0;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                    const float y = ((float)i1 + pixel_offset) / sf1;
                    for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
                        const float x = ((float)i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        const int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        const int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, src0->ne[0]);
                        const int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        const int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, src0->ne[1]);

                        // bilinear filter with antialiasing
                        float val = 0.0f;
                        float total_weight = 0.0f;

                        for (int64_t sy = y_min; sy < y_max; sy++) {
                            const float weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            for (int64_t sx = x_min; sx < x_max; sx++) {
                                const float weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                const float weight = weight_x * weight_y;

                                if (weight <= 0.0f) {
                                    continue;
                                }

                                const float pixel = *(const float*)((const char*)src0->data + sx * src0->nb[0] + sy * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        if (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        float* dst_ptr = (float*)((char*)dst->data + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
                        *dst_ptr = val;
                    }
                }
            }
        }
    }
    else if (mode == GGML_SCALE_MODE_BILINEAR) {
        for (int64_t i3 = 0; i3 < dst_mdspan.extent(0); i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = 0; i2 < dst_mdspan.extent(1); i2++) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < dst_mdspan.extent(2); i1++) {
                    const float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, src0->ne[1] - 1));
                    y1 = std::max(int64_t(0), std::min(y1, src0->ne[1] - 1));

                    float dy = y - (float)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    for (int64_t i0 = 0; i0 < dst_mdspan.extent(3); i0++) {
                        const float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, src0->ne[0] - 1));
                        x1 = std::max(int64_t(0), std::min(x1, src0->ne[0] - 1));

                        float dx = x - (float)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        const float a = src0_mdspan[i03, i02, y0, x0];
                        const float b = src0_mdspan[i03, i02, y0, x1];
                        const float c = src0_mdspan[i03, i02, y1, x0];
                        const float d = src0_mdspan[i03, i02, y1, x1];;

                        const float val = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy;

                        dst_mdspan[i3, i2, i1, i0] = val;
                    }
                }
            }
        }
    }
    else if (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        const float a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](float x) { return ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](float x) { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](float p0, float p1, float p2, float p3, float x) {
            const float w0 = weight2(x + 1);
            const float w1 = weight1(x + 0);
            const float w2 = weight1(1 - x);
            const float w3 = weight2(2 - x);
            return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
        };

        for (int64_t i3 = 0; i3 < dst_mdspan.extent(0); i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = 0; i2 < dst_mdspan.extent(1); i2++) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < dst_mdspan.extent(2); i1++) {
                    const float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
                    const int64_t y0 = (int64_t)floorf(y);
                    const float dy = y - (float)y0;

                    for (int64_t i0 = 0; i0 < dst_mdspan.extent(3); i0++) {
                        const float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
                        const int64_t x0 = (int64_t)floorf(x);
                        const float dx = x - (float)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> float {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, src0->ne[0] - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, src0->ne[1] - 1));
                            return src0_mdspan[i03, i02, i01, i00];
                        };

                        const float val = bicubic(
                            bicubic(p(-1, -1), p(0, -1), p(1, -1), p(2, -1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        dst_mdspan[i3, i2, i1, i0] = val;
                    }
                }
            }
        }
    }
    else {
        GGML_ABORT("unsupported upscale mode");
    }
}

void ggml_compute_forward_upscale(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_upscale_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
