module;
#include <assert.h>
#include "mdspan.hpp"
#include <algorithm>
#include <bit>
#include <iostream>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :ds;
import :tensor;
import :cpu.ds;
import :cpu.op;

template<typename T>
auto make_strided_mdspan(T* data, const std::array<int64_t, 4>& extents, const std::array<size_t, 4>& strides) {
    using extents_type = std::experimental::dims<4>;
    // Reverse order, maybe chage it.
    extents_type ext(extents[3], extents[2], extents[1], extents[0]);
    assert(strides[0] == sizeof(T));
	std::array<std::size_t, 4> newStride = {
        strides[3] / strides[0],
        strides[2] / strides[0],
        strides[1] / strides[0], 
        strides[0] / strides[0]};
    auto mapping = std::experimental::layout_stride::mapping{ ext, newStride };
    return std::experimental::mdspan(data, mapping);
}

template <typename T>
static void ggml_compute_forward_concat(
    const ggml_compute_params* params,
    ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(T));

    const int ith = params->ith;
    const int nth = params->nth;

    const int32_t dim = std::bit_cast<int32_t>(dst->op_params[0]);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim] = src0->ne[dim];

    std::experimental::mdspan dst_mdspan(static_cast<T*>(dst->data), dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]);
    auto src0_mdspan = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);
    auto src1_mdspan = make_strided_mdspan(static_cast<const T*>(src1->data), src1->ne, src1->nb);
    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < dst_mdspan.extent(0); i3++) {
        for (int i2 = ith; i2 < dst_mdspan.extent(1); i2 += nth) {
            for (int i1 = 0; i1 < dst_mdspan.extent(2); i1++) {
                for (int i0 = 0; i0 < dst_mdspan.extent(3); i0++) {
                    const T x = [&]() {
                        if (i0 < src0_mdspan.extent(3) && i1 < src0_mdspan.extent(2) && i2 < src0_mdspan.extent(1) && i3 < src0_mdspan.extent(0))
                            return src0_mdspan[i3, i2, i1, i0];
                        else
                            return src1_mdspan[i3 - o[3], i2 - o[2], i1 - o[1], i0 - o[0]];;
                    }();
                    dst_mdspan[i3, i2, i1, i0] = x;
                }
            }
        }
    }
}

void ggml_compute_forward_concat(
    const ggml_compute_params* params,
    ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I16:
    {
        ggml_compute_forward_concat<ggml_fp16_t>(params, dst);
    } break;
    case GGML_TYPE_I8:
    {
        ggml_compute_forward_concat<int8_t>(params, dst);
    } break;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
    {
        ggml_compute_forward_concat<ggml_fp32_t>(params, dst);
    } break;
    default:
    {
        ggml_compute_forward_concat<char>(params, dst);
    }
    }
}