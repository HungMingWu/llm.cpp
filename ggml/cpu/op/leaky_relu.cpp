module;
#include <assert.h>
#include <bit>
#include "mdspan.hpp"
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

template <typename T>
static void ggml_compute_forward_leaky_relu(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_contiguous_1(src0));
	assert(ggml_is_contiguous_1(dst));
	assert(ggml_are_same_shape(src0, dst));

	const int64_t nr = ggml_nrows(src0);
	const int64_t nc = src0->ne[0];

	const float negative_slope = std::bit_cast<float>(dst->op_params[0]);

	assert(dst->nb[0] == sizeof(T));
	assert(src0->nb[0] == sizeof(T));

	std::experimental::mdspan dst_data(static_cast<T*>(dst->data), nr, nc);
	std::experimental::mdspan src0_data(static_cast<const T*>(src0->data), nr, nc);

	for (int64_t i = 0; i < nr; i++) {
		for (int64_t j = 0; j < nc; j++) {
			ggml_fp32_t v = toFloat32(src0_data[i, j]);
			dst_data[i, j] = fromFloat32<T>(((v > 0.f) ? v : 0.f) + negative_slope * ((v < 0.0f) ? v : 0.f));
		}
	}
}

void ggml_compute_forward_leaky_relu(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_leaky_relu<ggml_fp32_t>(dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_leaky_relu<ggml_fp16_t>(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
