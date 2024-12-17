module;
#include <stdint.h>
#include <bit>
#include <type_traits>

export module ggml:types;

namespace internal {
	// Wait for C++26 constexpr fabsf
	constexpr float fabsf(float x) {
		uint32_t i = std::bit_cast<uint32_t>(x);
		i &= 0x7fffffff;
		return std::bit_cast<float>(i);
	}
}

template <typename T, typename tag>
struct strong_type {
	T value;
	constexpr strong_type() = default;
	constexpr strong_type(T value) : value(value) {}
	constexpr operator const T& () const { return value; }
	constexpr operator T& () { return value; }
};

export
{
    // ieee 754-2008 half-precision float16
	using ggml_fp16_t = strong_type<uint16_t, struct fp16>;
	using ggml_fp32_t = float;
	using ggml_i16_t = int16_t;
	using ggml_i32_t = int32_t;

	template <typename T>
	constexpr bool isIntegerType_v = std::is_same_v<T, ggml_i16_t> || std::is_same_v<T, ggml_i32_t>;

	// google brain half-precision bfloat16
	using ggml_bf16_t = strong_type<uint16_t, struct bf16>;

	template <typename T>
	ggml_fp32_t toFloat32(T value);

	template <typename T>
	T fromFloat32(ggml_fp32_t value);

	template <>
	ggml_fp32_t toFloat32(ggml_fp32_t value) { return value; }

	template <>
	ggml_fp32_t fromFloat32(ggml_fp32_t value) { return value; }

	// FP16 <-> FP32
	// ref: https://github.com/Maratyszcza/FP16
	template <>
	constexpr ggml_fp32_t toFloat32(ggml_fp16_t h) {
		const uint32_t w = (uint32_t)h << 16;
		const uint32_t sign = w & 0x80000000u;
		const uint32_t two_w = w + w;

		const uint32_t exp_offset = 0xE0u << 23;
		const float exp_scale = std::bit_cast<float>(0x7800000u);
		const float normalized_value = std::bit_cast<float>((two_w >> 4) + exp_offset) * exp_scale;

		const uint32_t magic_mask = 126u << 23;
		const float magic_bias = 0.5f;
		const float denormalized_value = std::bit_cast<float>((two_w >> 17) | magic_mask) - magic_bias;

		const uint32_t denormalized_cutoff = 1u << 27;
		const uint32_t result = sign |
			std::bit_cast<uint32_t>(two_w < denormalized_cutoff ? denormalized_value : normalized_value);
		return std::bit_cast<float>(result);
	}

	template <>
	ggml_fp16_t fromFloat32(ggml_fp32_t f)
	{
		const float scale_to_inf = std::bit_cast<float>(0x77800000u);
		const float scale_to_zero = std::bit_cast<float>(0x08800000u);
		float base = (internal::fabsf(f) * scale_to_inf) * scale_to_zero;

		const uint32_t w = std::bit_cast<uint32_t>(f);
		const uint32_t shl1_w = w + w;
		const uint32_t sign = w & 0x80000000u;
		uint32_t bias = shl1_w & 0xFF000000u;
		if (bias < 0x71000000u) {
			bias = 0x71000000u;
		}

		base = std::bit_cast<float>((bias >> 1) + 0x07800000u) + base;
		const uint32_t bits = std::bit_cast<uint32_t>(base);
		const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
		const uint32_t mantissa_bits = bits & 0x00000FFFu;
		const uint32_t nonsign = exp_bits + mantissa_bits;
		return (sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00 : nonsign);
	}

	/**
	 * Converts float32 to brain16.
	 *
	 * This is binary identical with Google Brain float conversion.
	 * Floats shall round to nearest even, and NANs shall be quiet.
	 * Subnormals aren't flushed to zero, except perhaps when used.
	 * This code should vectorize nicely if using modern compilers.
	 */
	template <>
	constexpr ggml_bf16_t fromFloat32(ggml_fp32_t s) {
		uint32_t i = std::bit_cast<uint32_t>(s);
		if ((i & 0x7fffffff) > 0x7f800000) { /* nan */
			return (i >> 16) | 64; /* force to quiet */
		}
		return (i + (0x7fff + ((i >> 16) & 1))) >> 16;
	}

	/**
	 * Converts brain16 to float32.
	 *
	 * The bfloat16 floating point format has the following structure:
	 *
	 *       ¢zsign
	 *       ¢x
	 *       ¢x   ¢zexponent
	 *       ¢x   ¢x
	 *       ¢x   ¢x      ¢zmantissa
	 *       ¢x   ¢x      ¢x
	 *       ¢x¢z¢w¢w¢r¢w¢w¢w¢{¢z¢w¢r¢w¢w¢w¢{
	 *     0b0000000000000000 brain16
	 *
	 * Since bf16 has the same number of exponent bits as a 32bit float,
	 * encoding and decoding numbers becomes relatively straightforward.
	 *
	 *       ¢zsign
	 *       ¢x
	 *       ¢x   ¢zexponent
	 *       ¢x   ¢x
	 *       ¢x   ¢x      ¢zmantissa
	 *       ¢x   ¢x      ¢x
	 *       ¢x¢z¢w¢w¢r¢w¢w¢w¢{¢z¢w¢r¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢{
	 *     0b00000000000000000000000000000000 IEEE binary32
	 *
	 * For comparison, the standard fp16 format has fewer exponent bits.
	 *
	 *       ¢zsign
	 *       ¢x
	 *       ¢x  ¢zexponent
	 *       ¢x  ¢x
	 *       ¢x  ¢x    ¢zmantissa
	 *       ¢x  ¢x    ¢x
	 *       ¢x¢z¢w¢r¢w¢{¢z¢w¢r¢w¢w¢w¢w¢w¢w¢{
	 *     0b0000000000000000 IEEE binary16
	 *
	 * @see IEEE 754-2008
	 */
	template <>
	constexpr ggml_fp32_t toFloat32(ggml_bf16_t h) // consider just doing << 16
	{
		uint32_t i = (uint32_t)h << 16;
		return std::bit_cast<ggml_fp32_t>(i);
	}
}
