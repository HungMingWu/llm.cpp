#pragma once
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "mdspan.hpp"
#include <array>

namespace details {
    template <size_t N, std::size_t... I>
    constexpr auto construct_extent_type(const std::span<const int64_t, N>& extents, std::index_sequence<I...>)
    {
        // Reverse order, maybe change it.
        using extents_type = std::experimental::dims<N>;
        return extents_type(extents[N - I - 1]...);
    }

    template <size_t N, std::size_t... I>
    constexpr auto construct_stride(const std::span<const size_t, N>& strides, std::index_sequence<I...>)
    {
        auto new_strides = std::array{ strides[N - I - 1]... };
		auto stride_size = *std::min_element(strides.begin(), strides.end());
        for (auto &v : new_strides) {
            assert(v % stride_size == 0 && "Stride must be divisible by the minimum stride size");
            v /= stride_size;
		}
        return new_strides;
    }

    template <typename T, size_t N, typename Indx = std::make_index_sequence<N>>
    auto make_strided_mdspan(T* data, const std::span<const int64_t, N>& extents, const std::span<const size_t, N>& strides) {
        auto ext = details::construct_extent_type(extents, Indx{});
        auto new_strides = details::construct_stride(strides, Indx{});
        auto mapping = std::experimental::layout_stride::mapping{ ext, new_strides };
        return std::experimental::mdspan(data, mapping);
    }
}

template <size_t M = 4, typename T, size_t N>
auto make_strided_mdspan(T* data, const std::array<int64_t, N>& extents, const std::array<size_t, N>& strides) {
	return details::make_strided_mdspan(data,
        std::span{ extents }.template first<M>(),
        std::span{strides}.template first<M>());
}
