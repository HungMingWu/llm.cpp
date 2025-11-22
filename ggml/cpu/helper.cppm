module;
#include <ranges>

module ggml:cpu.helper;
import :ds;

template <typename... Args>
requires (std::is_same_v<Args, size_t> && ...)
auto make_cartesian_product(Args... args)
{
    return std::views::cartesian_product((std::views::iota(0ull, args))...);
}
