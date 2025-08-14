module;
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "mdspan.hpp"
#include <algorithm>
#include <bit>
#include "../helper.h"

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

    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    auto dst_mdspan = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
    auto src0_mdspan = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

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
    }
    else if (mode == GGML_SCALE_MODE_BILINEAR) {
        float pixel_offset = 0.5f;
        if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
            pixel_offset = 0.0f;
            sf0 = (float)(dst->ne[0] - 1) / (src0->ne[0] - 1);
            sf1 = (float)(dst->ne[1] - 1) / (src0->ne[1] - 1);
        }

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
