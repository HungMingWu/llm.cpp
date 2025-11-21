module;
#include <assert.h>
#include "mdspan.hpp"
#include <algorithm>
#include <bit>
#include <iostream>
#include "helper.h"

#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :ds;
import :tensor;
import :cpu.ds;
import :cpu.op;

template <typename T>
static void ggml_compute_forward_concat(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(T));

    const int32_t dim = std::bit_cast<int32_t>(dst->op_params[0]);

    GGML_ASSERT(dim >= 0 && dim < 4);

    auto dst_data = make_strided_mdspan(static_cast<T*>(dst->data), dst->ne, dst->nb);
    auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);
    auto src1_data = make_strided_mdspan(static_cast<const T*>(src1->data), src1->ne, src1->nb);
    // TODO: smarter multi-theading
    for (int64_t i3 = 0; i3 < dst_data.extent(0); i3++) {
        for (int64_t i2 = 0; i2 < dst_data.extent(1); i2++) {
            for (int64_t i1 = 0; i1 < dst_data.extent(2); i1++) {
                for (int64_t i0 = 0; i0 < dst_data.extent(3); i0++) {
                    const T value = [&]() {
                        if (i0 < src0_data.extent(3) && i1 < src0_data.extent(2) && i2 < src0_data.extent(1) && i3 < src0_data.extent(0))
                            return src0_data[i3, i2, i1, i0];
                        else if (dim == 0)
                            return src1_data[i3, i2, i1, i0 - src0_data.extent(3)];
                        else if (dim == 1)
                            return src1_data[i3, i2, i1 - src0_data.extent(2), i0];
                        else if (dim == 2)
                            return src1_data[i3, i2 - src0_data.extent(1), i1, i0];
                        else
                            return src1_data[i3 - src0_data.extent(0), i2, i1, i0];
                    }();
                    dst_data[i3, i2, i1, i0] = value;
                }
            }
        }
    }
}

void ggml_compute_forward_concat(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I16:
    {
        ggml_compute_forward_concat<ggml_fp16_t>(dst);
    } break;
    case GGML_TYPE_I8:
    {
        ggml_compute_forward_concat<int8_t>(dst);
    } break;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
    {
        ggml_compute_forward_concat<ggml_fp32_t>(dst);
    } break;
    default:
    {
        ggml_compute_forward_concat<char>(dst);
    }
    }
}