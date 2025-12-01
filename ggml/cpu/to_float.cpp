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
		case GGML_TYPE_I32: {
			const int32_t* x_int = static_cast<const int32_t*>(x);
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
		case GGML_TYPE_Q4_0:
			dequantize_row(static_cast<const block_q4_0*>(x), y, n);
			break;
		case GGML_TYPE_Q4_1:
			dequantize_row(static_cast<const block_q4_1*>(x), y, n);
			break;
		case GGML_TYPE_Q5_0:
			dequantize_row(static_cast<const block_q5_0*>(x), y, n);
			break;
		case GGML_TYPE_Q5_1:
			dequantize_row(static_cast<const block_q5_1*>(x), y, n);
			break;
		case GGML_TYPE_Q8_0:
			dequantize_row(static_cast<const block_q8_0*>(x), y, n);
			break;
		case GGML_TYPE_MXFP4:
			dequantize_row(static_cast<const block_mxfp4*>(x), y, n);
			break;
		default:
			assert(false);
			GGML_ABORT("unsupported type for to_float");
	}
}
