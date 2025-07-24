#pragma once
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "mdspan.hpp"
#include <array>

template<typename T>
auto make_strided_mdspan(T* data, const std::array<int64_t, 4>& extents, const std::array<size_t, 4>& strides) {
    using extents_type = std::experimental::dims<4>;
    // Reverse order, maybe chage it.
    extents_type ext(extents[3], extents[2], extents[1], extents[0]);
    size_t stride_size = strides[0];
    if (stride_size > strides[1]) stride_size = strides[1];
    assert(stride_size == sizeof(T));
    std::array<std::size_t, 4> newStride = {
        strides[3] / stride_size,
        strides[2] / stride_size,
        strides[1] / stride_size,
        strides[0] / stride_size };
    auto mapping = std::experimental::layout_stride::mapping{ ext, newStride };
    return std::experimental::mdspan(data, mapping);
}
