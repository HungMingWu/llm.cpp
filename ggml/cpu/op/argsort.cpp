module;
#include <stdint.h>
#include <algorithm>
#include <bit>
#include <numeric>
#include <span>
#include "mdspan.hpp"
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

template <enum ggml_sort_order order>
struct argsort_cmp {
    std::span<const float> data;
    bool operator()(int32_t a, int32_t b) const {
        if constexpr (order == GGML_SORT_ORDER_ASC) {
            return data[a] < data[b];
        }
        else {
            return data[a] > data[b];
        }
    }
};

static void ggml_compute_forward_argsort_f32(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    ggml_sort_order order = std::bit_cast<ggml_sort_order>(dst->op_params[0]);
    std::mdspan dst_data(static_cast<int32_t*>(dst->data), dst->ne[3] * dst->ne[2] * dst->ne[1], dst->ne[0]);
    std::mdspan src0_data(static_cast<const float*>(src0->data), src0->ne[1], src0->ne[0]);

    for (int64_t i = 0; i < dst_data.extent(0); i++) {
        std::span<const float> src_span(&src0_data[i, 0], src0_data.extent(1));
        std::span<int32_t> dst_span(&dst_data[i, 0], dst_data.extent(1));
        std::ranges::iota(dst_span, 0);

        switch (order) {
        case GGML_SORT_ORDER_ASC:
            std::ranges::sort(dst_span, argsort_cmp<GGML_SORT_ORDER_ASC>{src_span});
            break;

        case GGML_SORT_ORDER_DESC:
            std::ranges::sort(dst_span, argsort_cmp<GGML_SORT_ORDER_DESC>{src_span});
            break;

        default:
            GGML_ABORT("invalid sort order");
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
