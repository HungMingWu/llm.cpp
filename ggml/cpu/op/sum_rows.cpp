module;
#include <assert.h>
#include <stdint.h>
#include "mdspan.hpp"
#include <algorithm>>
#include <ranges>
#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;

static void ggml_compute_forward_sum_rows_f32(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(src0->nb[0] == sizeof(float));
	assert(dst->nb[0] == sizeof(float));

	assert(dst->ne[0] == 1);
	assert(dst->ne[1] == src0->ne[1]);
	assert(dst->ne[2] == src0->ne[2]);
	assert(dst->ne[3] == src0->ne[3]);

	std::experimental::mdspan dst_mdspan(static_cast<float*>(dst->data), dst->ne[3], dst->ne[2], dst->ne[1]);
	std::experimental::mdspan src0_mdspan(static_cast<float*>(src0->data), src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]);

	auto cp = std::views::cartesian_product(
		std::views::iota(0ul, src0_mdspan.extent(0)),
		std::views::iota(0ul, src0_mdspan.extent(1)),
		std::views::iota(0ul, src0_mdspan.extent(2))
	);

	std::ranges::for_each(cp,
		[=](const auto& id) {
			auto [i, j, k] = id;
			float sum = 0.0;
			for (int64_t i0 = 0; i0 < src0_mdspan.extent(3); i0++) {
				sum += src0_mdspan[i, j, k, i0];
			}
			dst_mdspan[i, j, k] = sum;
		});
}

void ggml_compute_forward_sum_rows(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sum_rows_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}