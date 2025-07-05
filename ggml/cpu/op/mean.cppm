module;
#include <assert.h>
#include <stdint.h>
#define GGML_ABORT(...)

module ggml:cpu.op.mean;
import :ds;

static void ggml_vec_sum_f32(const int n, float* s, const float* x) {
#ifndef GGML_USE_ACCELERATE
	double sum = 0.0;
	for (int i = 0; i < n; ++i) {
		sum += (double)x[i];
	}
	*s = sum;
#else
	vDSP_sve(x, 1, s, n);
#endif
}

static void ggml_compute_forward_mean_f32(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(src0->nb[0] == sizeof(float));
	assert(dst->ne[0] == 1);
	assert(dst->ne[1] == src0->ne[1]);
	assert(dst->ne[2] == src0->ne[2]);
	assert(dst->ne[3] == src0->ne[3]);

	for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
		for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
			for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
				ggml_vec_sum_f32(src0->ne[0],
					(float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]),
					(float*)((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]));

				*(float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]) /= (float)src0->ne[0];
			}
		}
	}
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