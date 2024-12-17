module;
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :traits;
import :quants;
import :cpu.from_float;

bool ggml_quantize_requires_imatrix(enum ggml_type type) {
    return
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ1_S;//   ||
    //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
	enum ggml_type   type,
	const float* src,
	void* dst,
	int64_t   start,
	int64_t   nrows,
	int64_t   n_per_row,
	const float* imatrix) {
    const int64_t n = (int64_t)nrows * n_per_row;

    if (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != nullptr);
    }

    //GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    //ggml_quantize_init(type); // this is noop if already initialized

    const size_t start_row = start / n_per_row;
    const size_t row_size = ggml_row_size(type, n_per_row);

    size_t result = 0;
    switch (type) {
    case GGML_TYPE_Q4_0:    result = quantize_q4_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q4_1:    result = quantize_q4_1(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q5_0:    result = quantize_q5_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q5_1:    result = quantize_q5_1(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q8_0:    result = quantize_q8_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q2_K:    result = quantize_q2_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q3_K:    result = quantize_q3_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q4_K:    result = quantize_q4_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q5_K:    result = quantize_q5_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q6_K:    result = quantize_q6_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_TQ1_0:   result = quantize_tq1_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_TQ2_0:   result = quantize_tq2_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ2_S:   result = quantize_iq2_s(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ3_S:   result = quantize_iq3_s(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ1_S:   result = quantize_iq1_s(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ1_M:   result = quantize_iq1_m(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
#if 0
    case GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
#endif
    case GGML_TYPE_F16:
    {
        size_t elemsize = sizeof(ggml_fp16_t);
        from_float(src + start, (ggml_fp16_t*)dst + start, n);
        result = n * elemsize;
    } break;
    case GGML_TYPE_BF16:
    {
        size_t elemsize = sizeof(ggml_bf16_t);
        from_float(src + start, (ggml_bf16_t*)dst + start, n);
        result = n * elemsize;
    } break;
    case GGML_TYPE_F32:
    {
        size_t elemsize = sizeof(float);
        result = n * elemsize;
        memcpy((uint8_t*)dst + start * elemsize, src + start, result);
    } break;
    default:
        assert(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    return result;
}
