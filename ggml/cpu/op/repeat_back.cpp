module;
#include <assert.h>
#include "mdspan.hpp"
#include "mdspan_helper.h"
#define GGML_ABORT(...)

module ggml;
import :cpu.op;

static void ggml_compute_forward_repeat_back_f32(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_can_repeat(dst, src0));

	// guaranteed to be an integer due to the check in ggml_can_repeat
	const int nr0 = (int)(src0->ne[0] / dst->ne[0]);
	const int nr1 = (int)(src0->ne[1] / dst->ne[1]);
	const int nr2 = (int)(src0->ne[2] / dst->ne[2]);
	const int nr3 = (int)(src0->ne[3] / dst->ne[3]);

	// TODO: support for transposed / permuted tensors
	assert(dst->nb[0] == sizeof(float));
	assert(src0->nb[0] == sizeof(float));

	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

	for (int64_t k3 = 0; k3 < dst_data.extent(0); k3++)
		for (int64_t k2 = 0; k2 < dst_data.extent(1); k2++)
			for (int64_t k1 = 0; k1 < dst_data.extent(2); k1++)
				for (int64_t k0 = 0; k0 < dst_data.extent(3); k0++)
					dst_data[k3, k2, k1, k0] = 0.0f;

	// TODO: maybe this is not optimal?
	for (int64_t i3 = 0; i3 < nr3; i3++) {
		for (int64_t k3 = 0; k3 < dst_data.extent(0); k3++) {
			for (int64_t i2 = 0; i2 < nr2; i2++) {
				for (int64_t k2 = 0; k2 < dst_data.extent(1); k2++) {
					for (int64_t i1 = 0; i1 < nr1; i1++) {
						for (int64_t k1 = 0; k1 < dst_data.extent(2); k1++) {
							for (int64_t i0 = 0; i0 < nr0; i0++) {
								for (int64_t j = 0; j < dst_data.extent(3); j++) {
									dst_data[k3, k2, k1, j] +=
										src0_data[i3 * dst_data.extent(0) + k3, i2 * dst_data.extent(1) + k2,
										i1 * dst_data.extent(2) + k1, i0 * dst_data.extent(3) + j];
								}
							}
						}
					}
				}
			}
		}
	}
}

void ggml_compute_forward_repeat_back(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_repeat_back_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}