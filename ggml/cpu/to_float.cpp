module;
#include <assert.h>
#include <string.h>
#include "block.h"
#define GGML_ABORT(...)

module ggml;
import :ds;
import :quants;
import :cpu.to_float;

void to_float(ggml_type type, const void* x, float* y, int64_t n)
{
	switch (type) {
		case GGML_TYPE_F32:
			memcpy(y, x, n * sizeof(float));
			break;
		case GGML_TYPE_I16: {
			const int16_t* x_int = static_cast<const int16_t*>(x);
			for (int i = 0; i < n; i++)
				y[i] = static_cast<float>(x_int[i]);
			break;
		}
		case GGML_TYPE_I32: {
			const int32_t* x_int = static_cast<const int32_t*>(x);
			for (int i = 0; i < n; i++)
				y[i] = static_cast<float>(x_int[i]);
			break;
		}
		case GGML_TYPE_I64: {
			const int64_t* x_int = static_cast<const int64_t*>(x);
			for (int i = 0; i < n; i++)
				y[i] = static_cast<float>(x_int[i]);
			break;
		}
		case GGML_TYPE_F16:
			to_float(static_cast<const ggml_fp16_t*>(x), y, n);
			break;
		case GGML_TYPE_BF16:
			to_float(static_cast<const ggml_bf16_t*>(x), y, n);
			break;
		case GGML_TYPE_Q2_K:
			dequantize_row(static_cast<const block_q2_K*>(x), y, n);
			break;
		case GGML_TYPE_Q3_K:
			dequantize_row(static_cast<const block_q3_K*>(x), y, n);
			break;
		case GGML_TYPE_Q4_0:
			dequantize_row(static_cast<const block_q4_0*>(x), y, n);
			break;
		case GGML_TYPE_Q4_1:
			dequantize_row(static_cast<const block_q4_1*>(x), y, n);
			break;
		case GGML_TYPE_Q4_K:
			dequantize_row(static_cast<const block_q4_K*>(x), y, n);
			break;
		case GGML_TYPE_Q5_0:
			dequantize_row(static_cast<const block_q5_0*>(x), y, n);
			break;
		case GGML_TYPE_Q5_1:
			dequantize_row(static_cast<const block_q5_1*>(x), y, n);
			break;
		case GGML_TYPE_Q5_K:
			dequantize_row(static_cast<const block_q5_K*>(x), y, n);
			break;
		case GGML_TYPE_Q6_K:
			dequantize_row(static_cast<const block_q6_K*>(x), y, n);
			break;
		case GGML_TYPE_Q8_0:
			dequantize_row(static_cast<const block_q8_0*>(x), y, n);
			break;
		case GGML_TYPE_IQ1_M:
			dequantize_row(static_cast<const block_iq1_m*>(x), y, n);
			break;
		case GGML_TYPE_IQ1_S:
			dequantize_row(static_cast<const block_iq1_s*>(x), y, n);
			break;
		case GGML_TYPE_IQ2_XS:
			dequantize_row(static_cast<const block_iq2_xs*>(x), y, n);
			break;
		case GGML_TYPE_IQ2_S:
			dequantize_row(static_cast<const block_iq2_s*>(x), y, n);
			break;
		case GGML_TYPE_IQ2_XXS:
			dequantize_row(static_cast<const block_iq2_xxs*>(x), y, n);
			break;
		case GGML_TYPE_IQ3_S:
			dequantize_row(static_cast<const block_iq3_s*>(x), y, n);
			break;
		case GGML_TYPE_IQ3_XXS:
			dequantize_row(static_cast<const block_iq3_xxs*>(x), y, n);
			break;
		case GGML_TYPE_IQ4_NL:
			dequantize_row(static_cast<const block_iq4_nl*>(x), y, n);
			break;
		case GGML_TYPE_IQ4_XS:
			dequantize_row(static_cast<const block_iq4_xs*>(x), y, n);
			break;
		case GGML_TYPE_MXFP4:
			dequantize_row(static_cast<const block_mxfp4*>(x), y, n);
			break;
		default:
			assert(false);
			GGML_ABORT("unsupported type for to_float");
	}
}
