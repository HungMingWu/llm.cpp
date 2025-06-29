module;
#include <stdint.h>
#include <array>
#include <bit>

module ggml;
import :types;

namespace inner
{
#if 0
	// FP16 <-> FP32
	// ref: https://github.com/Maratyszcza/FP16
	constexpr uint32_t normalized_value(uint32_t two_w) {
		const uint32_t exp_offset = 0xE0u << 23;
		const float exp_scale = std::bit_cast<float>(0x7800000u);
		const float normalized_value = std::bit_cast<float>((two_w >> 4) + exp_offset) * exp_scale;
		return std::bit_cast<uint32_t>(normalized_value);
	}

	constexpr uint32_t denormalized_value(uint32_t two_w) {
		const uint32_t magic_mask = 126u << 23;
		const float magic_bias = 0.5f;
		const float denormalized_value = std::bit_cast<float>((two_w >> 17) | magic_mask) - magic_bias;
		return std::bit_cast<uint32_t>(denormalized_value);
	}

	constexpr ggml_fp32_t toFloat32(ggml_fp16_t h) {
		const uint32_t w = (uint32_t)h << 16;
		const uint32_t sign = w & 0x80000000u;
		const uint32_t two_w = w + w;
		const uint32_t denormalized_cutoff = 1u << 27;
		const uint32_t result = sign |
			(two_w < denormalized_cutoff ? denormalized_value(two_w) : normalized_value(two_w));
		return std::bit_cast<float>(result);
	}
#endif

	constexpr float toFloat32(ggml_fp16_t h) {
		uint32_t result;

		uint32_t sign = (std::bit_cast<uint16_t>(h) & 0x8000) << 16;
		uint32_t exp = (std::bit_cast<uint16_t>(h) & 0x7C00) >> 10;
		uint32_t mantissa = std::bit_cast<uint16_t>(h) & 0x03FF;

		if (exp == 0) {
			if (mantissa == 0) {
				result = sign;
			}
			else {
				exp = 127 - 15 + 1;
				while ((mantissa & 0x400) == 0) {
					mantissa <<= 1;
					exp--;
				}
				mantissa &= 0x3FF;
				result = sign | (exp << 23) | (mantissa << 13);
			}
		}
		else if (exp == 0x1F) {
			result = sign | 0x7F800000 | (mantissa << 13);
		}
		else {
			exp = exp - 15 + 127;
			result = sign | (exp << 23) | (mantissa << 13);
		}

		return std::bit_cast<float>(result);
	}

	consteval std::array<float, 65536> create_fp16_to_fp32_table() {
		std::array<float, 65536> table{};
		for (uint16_t i = 0; i < 256; ++i) {
			for (uint16_t j = 0; j < 256; j++)
				table[i * 256 + j] = inner::toFloat32(ggml_fp16_t(i * 256 + j));
		}
		return table;
	}

	static constexpr std::array<float, 65536> fp16_to_fp32_table = create_fp16_to_fp32_table();
}

ggml_fp32_t toFloat32(ggml_fp16_t h) {
	return inner::fp16_to_fp32_table[std::bit_cast<uint16_t>(h)];
}