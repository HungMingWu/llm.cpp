module;
#include <stdint.h>
#include <algorithm>
#include <bit>
#include "mdspan.hpp"
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

static void ggml_compute_forward_argsort_f32(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    ggml_sort_order order = std::bit_cast<ggml_sort_order>(dst->op_params[0]);
    std::experimental::mdspan dst_data(static_cast<int32_t*>(dst->data), dst->ne[1], dst->ne[0]);
    std::experimental::mdspan src0_data(static_cast<const float*>(src0->data), src0->ne[1], src0->ne[0]);

    for (int64_t i = 0; i < dst->ne[1]; i++) {
        for (int64_t j = 0; j < dst->ne[0]; j++) dst_data[i, j] = j;
        if (order == GGML_SORT_ORDER_ASC) {
            for (int64_t a = 0; a < dst->ne[0]; a++)
                for (int64_t b = a + 1; b < dst->ne[0]; b++)
                    if (src0_data[i, dst_data[i, a]] > src0_data[i, dst_data[i, b]]) {
                        std::swap(dst_data[i, a], dst_data[i, b]);
                    }
        }
        else {
            for (int64_t a = 0; a < dst->ne[0]; a++)
                for (int64_t b = a + 1; b < dst->ne[0]; b++)
                    if (src0_data[i, dst_data[i, a]] < src0_data[i, dst_data[i, b]]) {
                        std::swap(dst_data[i, a], dst_data[i, b]);
                    }
        }
    }
}

void ggml_compute_forward_argsort(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_argsort_f32(dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}
