module;
#include <assert.h>
#include <stdint.h>
#include "mdspan.hpp"
#include <algorithm>
#include <ranges>
#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;
import :cpu.helper;

static void ggml_compute_forward_mean_f32(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(src0->nb[0] == sizeof(float));
	assert(dst->ne[0] == 1);
	assert(dst->ne[1] == src0->ne[1]);
	assert(dst->ne[2] == src0->ne[2]);
	assert(dst->ne[3] == src0->ne[3]);

	std::mdspan dst_data(static_cast<float*>(dst->data), dst->ne[3], dst->ne[2], dst->ne[1]);
	std::mdspan src0_data(static_cast<float*>(src0->data), src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]);

	for (auto [i, j, k] : make_cartesian_product(src0_data.extent(0), src0_data.extent(1), src0_data.extent(2))) {
		double sum = 0.0f;
		for (size_t l = 0; l < src0_data.extent(3); l++) {
			sum += src0_data[i, j, k, l];
		}
		dst_data[i, j, k] = sum / static_cast<float>(src0_data.extent(3));
	};
}

void ggml_compute_forward_mean(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_mean_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}