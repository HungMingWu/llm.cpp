module;
#include <assert.h>
#define GGML_ABORT(...)

module ggml:cpu.op.repeat_back;
import ggml;

inline static void ggml_vec_acc_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] += x[i]; }

static void ggml_compute_forward_repeat_back_f32(
	ggml_tensor* dst) {

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

	if (ggml_is_contiguous(dst)) {
		ggml_vec_set_f32(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3], (float*)dst->data, 0);
	}
	else {
		for (int k3 = 0; k3 < dst->ne[3]; k3++) {
			for (int k2 = 0; k2 < dst->ne[2]; k2++) {
				for (int k1 = 0; k1 < dst->ne[1]; k1++) {
					ggml_vec_set_f32(dst->ne[0],
						(float*)((char*)dst->data + k1 * dst->nb[1] + k2 * dst->nb[2] + k3 * dst->nb[3]),
						0);
				}
			}
		}
	}

	// TODO: maybe this is not optimal?
	for (int i3 = 0; i3 < nr3; i3++) {
		for (int k3 = 0; k3 < dst->ne[3]; k3++) {
			for (int i2 = 0; i2 < nr2; i2++) {
				for (int k2 = 0; k2 < dst->ne[2]; k2++) {
					for (int i1 = 0; i1 < nr1; i1++) {
						for (int k1 = 0; k1 < dst->ne[1]; k1++) {
							for (int i0 = 0; i0 < nr0; i0++) {
								ggml_vec_acc_f32(dst->ne[0],
									(float*)((char*)dst->data + (k3)*dst->nb[3] + (k2)*dst->nb[2] + (k1)*dst->nb[1]),
									(float*)((char*)src0->data + (i3 * dst->ne[3] + k3) * src0->nb[3] + (i2 * dst->ne[2] + k2) * src0->nb[2] + (i1 * dst->ne[1] + k1) * src0->nb[1] + (i0 * dst->ne[0]) * src0->nb[0]));
							}
						}
					}
				}
			}
		}
	}
}

void ggml_compute_forward_repeat_back(
	ggml_tensor* dst) {

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