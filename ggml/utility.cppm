module;
#include <type_traits>

export module ggml:utility;

export
{
	template <typename T, typename ...Args>
		requires (std::conjunction_v<std::is_same<T, Args>...>)
	constexpr bool is_one_of(T value, Args&&... args)
	{
		return ((value == args) || ...);
	}

	template <typename T, typename... Args>
		requires (std::conjunction_v<std::is_same<T, Args>...>)
	constexpr bool is_not_one_of(T value, Args&&... args) {
		return ((value != args) && ...);
	}
}
