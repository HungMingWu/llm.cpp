#pragma once
#include <array>
#include "common.cuh"
namespace detail {
    static __device__
    auto calcIndex(int64_t index, uint3 i)
    {
        return std::make_tuple(index);
    }

    template <typename... Args>
    requires (std::is_same_v<Args, uint3> && ...)
    static __device__
    auto calcIndex(int64_t index, uint3 first, Args... args)
    {
        uint2 div_mod = fast_div_modulo(index, first);
        return std::tuple_cat(
            std::make_tuple(div_mod.y),
            calcIndex(div_mod.x, args...)
        );
    }
}

template <typename Functor, typename... Args>
requires (std::is_same_v<Args, uint3> && ...)
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
    auto transformedTuple = std::apply([=](auto... args) {
        return std::make_tuple(init_fastdiv_values(static_cast<uint32_t>(args))...);
    }, tuple);
    const int64_t blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    launch_kernel << < blocks, BLOCK_SIZE, 0, stream >> > (total_elements, transformedTuple, functor);
}