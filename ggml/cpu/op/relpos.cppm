module;
#include <assert.h>
#include <stdint.h>
#define GGML_ABORT(...)

module ggml:cpu.op.relpos;
import :types;
import :cpu.ds;

static void ggml_compute_forward_get_rel_pos_f16(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

	const int64_t w = dst->ne[1];

	ggml_fp16_t* src0_data = (ggml_fp16_t*)src0->data;
	ggml_fp16_t* dst_data = (ggml_fp16_t*)dst->data;

	for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
		for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
			const int64_t pos = (w - i1 - 1) + i2;
			for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
				dst_data[i2 * dst->ne[1] * dst->ne[0] + i1 * dst->ne[0] + i0] = src0_data[pos * src0->ne[0] + i0];
			}
		}
	}
}

void ggml_compute_forward_get_rel_pos(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_get_rel_pos_f16(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
