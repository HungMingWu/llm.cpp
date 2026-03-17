module;
#include <math.h>
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

ggml_fp32_t toFloat32(ggml_e8m0_t x_) {
	uint8_t x = std::bit_cast<uint8_t>(x_);
	uint32_t bits;  // Stores the raw bit representation of the float

	// Handle special case for minimum exponent (denormalized float)
	if (x == 0) {
		// Bit pattern for 2^(-127):
		// - Sign bit: 0 (positive)
		// - Exponent: 0 (denormalized number)
		// - Mantissa: 0x400000 (0.5 in fractional form)
		// Value = 0.5 * 2^(-126) = 2^(-127)
		bits = 0x00400000;
	}
	// note: disabled as we don't need to handle NaNs
	//// Handle special case for NaN (all bits set)
	//else if (x == 0xFF) {
	//    // Standard quiet NaN pattern:
	//    // - Sign bit: 0
	//    // - Exponent: all 1s (0xFF)
	//    // - Mantissa: 0x400000 (quiet NaN flag)
	//    bits = 0x7FC00000;
	//}
	// Normalized values (most common case)
	else {
		// Construct normalized float by shifting exponent into position:
		// - Exponent field: 8 bits (positions 30-23)
		// - Mantissa: 0 (implicit leading 1)
		// Value = 2^(x - 127)
		bits = (uint32_t)x << 23;
	}

	// Safely reinterpret bit pattern as float without type-punning issues
	return std::bit_cast<ggml_fp32_t>(bits);
}

template <>
ggml_ue4m3_t fromFloat32(ggml_fp32_t x)
{
	if (!(x > 0.0f)) {
		return 0;
	}
	if (x > 448.0f) {
		x = 448.0f;
	}
	uint32_t bits = std::bit_cast<uint32_t>(x);
	int fp32_exp = ((bits >> 23) & 0xFF) - 127;
	int fp32_man = (bits >> 20) & 0x7;
	int ue4m3_exp = fp32_exp + 7;
	if (ue4m3_exp <= 0) {
		// subnormal: value = man * 2^-9, man = round(x * 2^9)
		int man = (int)(x * 512.0f + 0.5f);
		if (man > 7) {
			man = 7;
		}
		if (man < 1) {
			return 0;
		}
		return (uint8_t)man;
	}
	if (ue4m3_exp >= 15) {
		return 0x7E;
	}
	int round_bit = (bits >> 19) & 1;
	int ue4m3_man = fp32_man + round_bit;
	if (ue4m3_man > 7) {
		ue4m3_man = 0;
		ue4m3_exp++;
		if (ue4m3_exp >= 15) {
			return 0x7E;
		}
	}
	return std::bit_cast<ggml_ue4m3_t>((uint8_t)((ue4m3_exp << 3) | ue4m3_man));
}

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ggml_fp32_t toFloat32(ggml_ue4m3_t x_)
{
	uint8_t x = std::bit_cast<uint8_t>(x_);
	if (x == 0 || x == 0x7F) {
		return 0.0f;
	}
	int   exp = (x >> 3) & 0xF;
	int   man = x & 0x7;
	ggml_fp32_t raw;
	if (exp == 0) {
		raw = ldexpf((float)man, -9);
	}
	else {
		raw = ldexpf(1.0f + (ggml_fp32_t)man / 8.0f, exp - 7);
	}
	return raw * 0.5f;
}
