#pragma once
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "mdspan.hpp"
#include <array>

namespace details {
    template <size_t N, std::size_t... I>
    HOST DEVICE auto construct_extent_type(const std::span<const int64_t, N>& extents, std::index_sequence<I...>)
    {
        // Reverse order, maybe change it.
        using extents_type = std::experimental::dims<N>;
        return extents_type(extents[N - I - 1]...);
    }

    template <size_t N, std::size_t... I>
    HOST DEVICE auto construct_stride(const std::span<const size_t, N>& strides, std::index_sequence<I...>)
    {
        auto new_strides = std::array{ strides[N - I - 1]... };
		// std::min_element fails at CUDA device function, roll back to old style
        auto stride_size = strides[0];
        for (size_t i = 1; i < strides.size(); i++) {
            stride_size = std::min(stride_size, strides[i]);
        }
        for (auto &v : new_strides) {
            assert(v % stride_size == 0 && "Stride must be divisible by the minimum stride size");
            v /= stride_size;
        }
        return new_strides;
    }

    template <typename T, size_t N, typename Indx = std::make_index_sequence<N>>
    HOST DEVICE auto make_strided_mdspan(T* data, const std::span<const int64_t, N>& extents, const std::span<const size_t, N>& strides) {
        auto ext = details::construct_extent_type(extents, Indx{});
        auto new_strides = details::construct_stride(strides, Indx{});
        auto mapping = std::experimental::layout_stride::mapping<decltype(ext)> { ext, new_strides };
        return std::experimental::mdspan(data, mapping);
    }
}

template <size_t M = 4, typename T, size_t N>
HOST DEVICE auto make_strided_mdspan(T* data, const std::array<int64_t, N>& extents, const std::array<size_t, N>& strides) {
	return details::make_strided_mdspan(data,
        std::span{ extents }.template first<M>(),
        std::span{strides}.template first<M>());
}

template <size_t M = 4, typename T, size_t N>
HOST DEVICE auto make_strided_mdspan(T* data, const int64_t (&extents)[N], const size_t (&strides)[N]) {
    return details::make_strided_mdspan(data,
        std::span{ extents }.template first<M>(),
        std::span{ strides }.template first<M>());
}

template <typename T, std::size_t... Extents>
class mdarray {
public:
    static constexpr std::size_t rank = sizeof...(Extents);

    using extents_type = std::experimental::extents<std::size_t, Extents...>;
    using mdspan_type = std::experimental::mdspan<T, extents_type>;

private:
    static constexpr std::size_t total_size = (Extents * ...);

    std::array<T, total_size> storage_{};
    mdspan_type view_{ storage_.data() };

public:
    // ===== constructors =====
    constexpr mdarray() = default;

    constexpr explicit mdarray(const T& value) {
        storage_.fill(value);
    }

    // ===== element access =====
    template <typename... Indices>
    constexpr T& operator()(Indices... indices) noexcept {
        static_assert(sizeof...(Indices) == rank);
        return view_(indices...);
    }

    template <typename... Indices>
    constexpr const T& operator()(Indices... indices) const noexcept {
        static_assert(sizeof...(Indices) == rank);
        return view_(indices...);
    }

    // ===== raw access =====
    constexpr T* data() noexcept { return storage_.data(); }
    constexpr const T* data() const noexcept { return storage_.data(); }

    // ===== mdspan view =====
    constexpr mdspan_type mdspan() noexcept { return view_; }
    constexpr mdspan_type mdspan() const noexcept { return view_; }
};