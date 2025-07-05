module;
#include <assert.h>
#include <numeric>
#include <span>
#define GGML_ABORT(...)

module ggml:cpu.op.sum_rows;
import :ds;

static void ggml_compute_forward_sum_rows_f32(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(src0->nb[0] == sizeof(float));
	assert(dst->nb[0] == sizeof(float));

	assert(dst->ne[0] == 1);
	assert(dst->ne[1] == src0->ne[1]);
	assert(dst->ne[2] == src0->ne[2]);
	assert(dst->ne[3] == src0->ne[3]);

	for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
		for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
			for (int64_t i1 = 0; i1 < src0->ne[1]; i1++) {
				float* src_row = (float*)((char*)src0->data + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);
				float* dst_row = (float*)((char*)dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
				std::span<float> src_span(src_row, src0->ne[0]);
				dst_row[0] = std::reduce(src_span.begin(), src_span.end(), 0.0f);
			}
		}
	}
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