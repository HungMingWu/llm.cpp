module;
#include <assert.h>
#include <stdint.h>
#include "mdspan.hpp"
#include "mdspan_helper.h"
#include <algorithm>
#include <ranges>
#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;

template <typename T>
static void ggml_compute_forward_sum(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_scalar(dst));
	assert(src0->nb[0] == sizeof(T));

	float sum = 0;

	std::mdspan dst_data(static_cast<T*>(dst->data), 1);
	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);

	for (int64_t i03 = 0; i03 < src0_data.extent(0); i03++) {
		for (int64_t i02 = 0; i02 < src0_data.extent(1); i02++) {
			for (int64_t i01 = 0; i01 < src0_data.extent(2); i01++) {
				for (int64_t i00 = 0; i00 < src0_data.extent(3); i00++) {
					sum += toFloat32(src0_data[i03, i02, i01, i00]);
				}
			}
		}
	}
	dst_data[0] = fromFloat32<T>(sum);
}

void ggml_compute_forward_sum(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sum<ggml_fp32_t>(dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_sum<ggml_fp16_t>(dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_sum<ggml_bf16_t>(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
