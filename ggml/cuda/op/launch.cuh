#pragma once
#include <array>
#include "common.cuh"
namespace detail {
    static constexpr size_t BLOCK_SIZE = 256;

    template <typename... Args>
    requires (std::convertible_to<Args, int64_t> && ...)
    int64_t calcElements(const std::tuple<Args...> &tuple)
    {
        return std::apply([](auto... args) -> int64_t {
            return ((1 * args) * ...);
        }, tuple);
    }

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

    template <typename... Args>
    requires (std::convertible_to<Args, int64_t> && ...)
    static auto transformTuple(const std::tuple<Args...>& tuple)
    {
        return std::apply([=](auto... args) {
            return std::make_tuple(init_fastdiv_values(static_cast<uint32_t>(args))...);
        }, tuple);
    }

    template <typename... Args>
    requires (std::is_same_v<Args, uint3> && ...)
    static __device__ auto rollbackTuple(int64_t idx, const std::tuple<Args...>& tuple)
    {
        return std::apply([=](auto... args) {
            return calcIndex(idx, args...);
        }, tuple);
    }

    template <typename Functor, typename... Args>
    requires (std::is_same_v<Args, uint3> && ...)
    static __global__ void launch_kernel(int64_t total_elements, std::tuple<Args...> tuple, Functor functor)
    {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;
        std::apply(functor, rollbackTuple(idx, tuple));
    }

    template <typename Functor, typename... Args>
    requires (std::is_same_v<Args, uint3> && ...)
    static __global__ void launch_kernel_with_threads(int64_t total_elements, std::tuple<Args...> tuple, Functor functor)
    {
        int64_t idx = blockIdx.y * gridDim.x + blockIdx.x;
        if (idx >= total_elements) return;
        std::apply(functor, std::tuple_cat(rollbackTuple(idx, tuple), std::make_tuple(threadIdx.x)));
    }
}

template <typename Functor, typename... Args>
requires (std::convertible_to<Args, int64_t> && ...)
static void launch_functor(cudaStream_t stream, std::tuple<Args...> tuple, Functor functor)
{
    const int64_t total_elements = detail::calcElements(tuple);
    assert(total_elements < (1ll << 40 ) - 256);
    if (total_elements == 0) return;
    const int64_t blocks = (total_elements + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE;

    detail::launch_kernel << < blocks, detail::BLOCK_SIZE, 0, stream >> >
        (total_elements, detail::transformTuple(tuple), functor);
}

template <typename Functor, typename... Args>
requires (std::convertible_to<Args, int64_t> && ...)
static void launch_functor_with_threads(cudaStream_t stream, std::tuple<Args...> tuple, int64_t threads, int64_t memory, Functor functor)
{
    const int64_t total_elements = detail::calcElements(tuple);
    assert(total_elements < (1ll << 40) - 256);
    if (total_elements == 0) return;

    const int64_t block_x = (total_elements + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE;
    const dim3 block_nums(block_x, detail::BLOCK_SIZE, 1);
    detail::launch_kernel_with_threads << < block_nums, threads, memory, stream >> >
        (total_elements, detail::transformTuple(tuple), functor);
}