module;
#include <assert.h>
#include <algorithm>
#include <bit>
#include "mdspan.hpp"

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

static void ggml_compute_forward_scale_f32(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float s = std::bit_cast<float>(dst->op_params[0]); // scale factor
    float b = std::bit_cast<float>(dst->op_params[1]);// bias

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    const size_t nb01 = src0->nb[1];

    const size_t nb1 = dst->nb[1];

    std::experimental::mdspan dst_data(static_cast<float*>(dst->data), nr, nc);
    std::experimental::mdspan src0_data(static_cast<const float*>(src0->data), nr, nc);

    if (b == 0.0f) {
        for (int64_t i1 = 0; i1 < nr; i1++) {
            if (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    dst_data[i1, i0] = src0_data[i1, i0];
				}
            }
            for (int64_t i0 = 0; i0 < nc; i0++) {
                dst_data[i1, i0] *= s;
            }
        }
    }
    else {
        for (int64_t i1 = 0; i1 < nr; i1++) {
            for (int64_t i0 = 0; i0 < nc; i0++) {
                dst_data[i1, i0] = src0_data[i1, i0] * s + b;
            }
        }
    }
}

void ggml_compute_forward_scale(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_scale_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
