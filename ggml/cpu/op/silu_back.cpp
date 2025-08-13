module;
#include <assert.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include "mdspan.hpp"
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

inline static float ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f / (1.0f + expf(-x));
    return dy * s * (1.0f + x * (1.0f - s));
}

static void ggml_compute_forward_silu_back_f32(ggml_tensor* dst) {
    const ggml_tensor* grad = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    assert(ggml_is_contiguous_1(grad));
    assert(ggml_is_contiguous_1(src1));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src1, dst));
    assert(ggml_are_same_shape(src1, grad));

    const int nc = src1->ne[0];
    const int nr = ggml_nrows(src1);

    std::experimental::mdspan dst_data(static_cast<float*>(dst->data), nr, nc);
    std::experimental::mdspan grad_data(static_cast<const float*>(grad->data), nr, nc);
    std::experimental::mdspan src1_data(static_cast<const float*>(src1->data), nr, nc);

    for (int i1 = 0; i1 < nr; i1++) {
        for (int64_t i0 = 0; i0 < nc; i0++) {
            dst_data[i1, i0] = ggml_silu_backward_f32(src1_data[i1, i0], grad_data[i1, i0]);
#ifndef NDEBUG
            assert(!isnan(dst_data[i1, i0]));
            assert(!isinf(dst_data[i1, i0]));
#endif
		}
    }
}

void ggml_compute_forward_silu_back(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_silu_back_f32(dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}