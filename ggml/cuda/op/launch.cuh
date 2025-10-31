#pragma once
#include "common.cuh"
namespace detail {
    static __device__
    auto calcIndex(int64_t index, int64_t i)
    {
        assert(index < i);
        return std::make_tuple(index);
    }

    template <typename... Args>
    requires (std::convertible_to<Args, int64_t> && ...)
    static __device__
        auto calcIndex(int64_t index, int64_t first, Args... args)
    {
        return std::tuple_cat(
            std::make_tuple(index % first),
            calcIndex(index / first, args...)
        );
    }
}

template <typename Functor, typename... Args>
requires (std::convertible_to<Args, int64_t> && ...)
static __global__ void launch_kernel(int64_t total_elements, std::tuple<Args...> tuple, Functor functor)
{
    int64_t idx5 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx5 >= total_elements) return;
    auto index = std::apply([=](auto... args) {
        return detail::calcIndex(idx5, args...);
    }, tuple);
    std::apply(functor, index);
}

template <typename Functor, typename... Args>
requires (std::convertible_to<Args, int64_t> && ...)
static void launch_functor(cudaStream_t stream, std::tuple<Args...> tuple, Functor functor)
{
    static constexpr size_t BLOCK_SIZE = 256;
    const int64_t total_elements = std::apply([](auto... args) -> int64_t {
        return ((1 * args) * ...);
    }, tuple);
    assert(total_elements < (1ll << 40 ) - 256);
    if (total_elements == 0) return;
    const int64_t blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    launch_kernel << < blocks, BLOCK_SIZE, 0, stream >> > (total_elements, tuple, functor);
}