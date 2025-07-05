module;
#include <stddef.h>
#include <stdint.h>
#include <utility>
#include "block.h"

module ggml:quants;

void quantize_row(const float* x, block_q4_0* y, int64_t k);
void quantize_row(const float* x, block_q4_1* y, int64_t k);
void quantize_row(const float* x, block_q5_0* y, int64_t k);
void quantize_row(const float* x, block_q5_1* y, int64_t k);
void quantize_row(const float* x, block_q8_0* y, int64_t k);
void quantize_row(const float* x, block_q8_1* y, int64_t k);
void quantize_row(const float* x, block_q2_K* y, int64_t k);
void quantize_row(const float* x, block_q3_K* y, int64_t k);
void quantize_row(const float* x, block_q4_K* y, int64_t k);
void quantize_row(const float* x, block_q5_K* y, int64_t k);
void quantize_row(const float* x, block_q6_K* y, int64_t k);
void quantize_row(const float* x, block_q8_K* y, int64_t k);
void quantize_row(const float* x, block_tq1_0* y, int64_t k);
void quantize_row(const float* x, block_tq2_0* y, int64_t k);
void quantize_row(const float* x, block_iq4_nl* y, int64_t k);
void quantize_row(const float* x, block_iq4_xs* y, int64_t k);

void dequantize_row(const block_q4_0* x, float* y, int64_t k);
void dequantize_row(const block_q4_1* x, float* y, int64_t k);
void dequantize_row(const block_q5_0* x, float* y, int64_t k);
void dequantize_row(const block_q5_1* x, float* y, int64_t k);
void dequantize_row(const block_q8_0* x, float* y, int64_t k);
void dequantize_row(const block_q2_K* x, float* y, int64_t k);
void dequantize_row(const block_q3_K* x, float* y, int64_t k);
void dequantize_row(const block_q4_K* x, float* y, int64_t k);
void dequantize_row(const block_q5_K* x, float* y, int64_t k);
void dequantize_row(const block_q6_K* x, float* y, int64_t k);
void dequantize_row(const block_tq1_0* x, float* y, int64_t k);
void dequantize_row(const block_tq2_0* x, float* y, int64_t k);
void dequantize_row(const block_iq2_xxs* x, float* y, int64_t k);
void dequantize_row(const block_iq2_xs* x, float* y, int64_t k);
void dequantize_row(const block_iq2_s* x, float* y, int64_t k);
void dequantize_row(const block_iq3_xxs* x, float* y, int64_t k);
void dequantize_row(const block_iq3_s* x, float* y, int64_t k);
void dequantize_row(const block_iq1_s* x, float* y, int64_t k);
void dequantize_row(const block_iq1_m* x, float* y, int64_t k);
void dequantize_row(const block_iq4_nl* x, float* y, int64_t k);
void dequantize_row(const block_iq4_xs* x, float* y, int64_t k);

void quantize_row_q4_0_ref(const float* x, block_q4_0* y, int64_t k);
void quantize_row_q4_1_ref(const float* x, block_q4_1* y, int64_t k);
void quantize_row_q5_0_ref(const float* x, block_q5_0* y, int64_t k);
void quantize_row_q5_1_ref(const float* x, block_q5_1* y, int64_t k);
void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int64_t k);
void quantize_row_q2_K_ref(const float* x, block_q2_K* y, int64_t k);
void quantize_row_q3_K_ref(const float* x, block_q3_K* y, int64_t k);
void quantize_row_q4_K_ref(const float* x, block_q4_K* y, int64_t k);
void quantize_row_q5_K_ref(const float* x, block_q5_K* y, int64_t k);
void quantize_row_q6_K_ref(const float* x, block_q6_K* y, int64_t k);
void quantize_row_tq1_0_ref(const float* x, block_tq1_0* y, int64_t k);
void quantize_row_tq2_0_ref(const float* x, block_tq2_0* y, int64_t k);

template <typename T, typename... Types>
constexpr bool is_one_of_type = (false || ... || std::is_same_v<T, Types>);

template <typename T>
constexpr bool is_quant_type_v = is_one_of_type<T,
	block_q4_0,
	block_q4_1,
	block_q5_0,
	block_q5_1,
	block_q8_0,
	block_q8_1,
	block_q2_K,
	block_q3_K,
	block_q4_K,
	block_q5_K,
	block_q6_K,
	block_q8_K,
	block_tq1_0,
	block_tq2_0,
	block_iq2_xxs,
	block_iq2_xs,
	block_iq2_s,
	block_iq3_xxs,
	block_iq3_s,
	block_iq1_s,
	block_iq1_m,
	block_iq4_nl,
	block_iq4_xs>;

size_t quantize_q4_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q4_1(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q5_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q5_1(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q8_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q2_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q3_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q4_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q5_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_q6_K(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_tq1_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*);
size_t quantize_tq2_0(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float*);
size_t quantize_iq2_xxs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq2_xs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq2_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq3_xxs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq3_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq1_s(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq1_m(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq4_nl(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
size_t quantize_iq4_xs(const float* src, void* dst, int64_t nrow, int64_t n_per_row, const float* quant_weights);
