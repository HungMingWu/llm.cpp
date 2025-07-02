module;
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <algorithm>
#include <bit>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml:cpu.op.upscale;
import :ds;
import :tensor;
import :cpu.ds;

static void ggml_compute_forward_upscale_f32(
    const ggml_compute_params* params,
    ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    float sf0 = (float)dst->ne[0] / src0->ne[0];
    float sf1 = (float)dst->ne[1] / src0->ne[1];
    float sf2 = (float)dst->ne[2] / src0->ne[2];
    float sf3 = (float)dst->ne[3] / src0->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    if (mode == GGML_SCALE_MODE_NEAREST) {
        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = ith; i2 < dst->ne[2]; i2 += nth) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                    const int64_t i01 = i1 / sf1;
                    for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
                        const int64_t i00 = i0 / sf0;

                        const float* x = (float*)((char*)src0->data + i00 * src0->nb[0] + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                        float* y = (float*)((char*)dst->data + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);

                        *y = *x;
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

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = ith; i2 < dst->ne[2]; i2 += nth) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                    const float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, src0->ne[1] - 1));
                    y1 = std::max(int64_t(0), std::min(y1, src0->ne[1] - 1));

                    float dy = y - (float)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
                        const float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, src0->ne[0] - 1));
                        x1 = std::max(int64_t(0), std::min(x1, src0->ne[0] - 1));

                        float dx = x - (float)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        const float a = *(const float*)((const char*)src0->data + x0 * src0->nb[0] + y0 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                        const float b = *(const float*)((const char*)src0->data + x1 * src0->nb[0] + y0 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                        const float c = *(const float*)((const char*)src0->data + x0 * src0->nb[0] + y1 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                        const float d = *(const float*)((const char*)src0->data + x1 * src0->nb[0] + y1 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);

                        const float val = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy;

                        float* y_dst = (float*)((char*)dst->data + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
                        *y_dst = val;
                    }
                }
            }
        }
    }
    else {
        GGML_ABORT("unsupported upscale mode");
    }
}

void ggml_compute_forward_upscale(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_upscale_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
