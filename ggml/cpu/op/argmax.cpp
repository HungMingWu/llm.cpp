module;
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <limits>
#include "mdspan.hpp"

#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;

static void ggml_compute_forward_argmax_f32(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	assert(src0->nb[0] == sizeof(float));
	assert(dst->nb[0] == sizeof(int32_t));

	std::experimental::mdspan dst_data(static_cast<int32_t*>(dst->data), dst->ne[0]);
	std::experimental::mdspan src0_data(static_cast<float*>(src0->data), src0->ne[1], src0->ne[0]);

	for (int64_t i = 0; i < src0_data.extent(0); i++) {
		float maxValue = -std::numeric_limits<float>::infinity();
		int32_t idx = 0;
		for (int32_t j = 0; j < src0_data.extent(1); ++j) {
			if (src0_data[i, j] > maxValue) {
				maxValue = src0_data[i, j];
				idx = j;
			}
		}
		dst_data[i] = idx;
	}
}

void ggml_compute_forward_argmax(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_argmax_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
