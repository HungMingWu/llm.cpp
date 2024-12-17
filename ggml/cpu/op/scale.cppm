module;
#include <string.h>
#include <algorithm>
#include <bit>

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

inline static void ggml_vec_scale_f321(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
	vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
	const int np = (n & ~(GGML_F32_STEP - 1));

	GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	GGML_F32_VEC ay[GGML_F32_ARR];

	for (int i = 0; i < np; i += GGML_F32_STEP) {
		for (int j = 0; j < GGML_F32_ARR; j++) {
			ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
			ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

			GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] *= v;
	}
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] *= v;
	}
#endif
}

module ggml:cpu.op.scale;
import :ds;
import :tensor;
import :cpu.ds;

static void ggml_compute_forward_scale_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(ggml_is_contiguous(src0));
	GGML_ASSERT(ggml_is_contiguous(dst));
	GGML_ASSERT(ggml_are_same_shape(src0, dst));

	// scale factor
	float v = std::bit_cast<float>(dst->op_params[0]);

	const int ith = params->ith;
	const int nth = params->nth;

	const int nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	const size_t nb01 = src0->nb[1];

	const size_t nb1 = dst->nb[1];

	for (int i1 = ir0; i1 < ir1; i1++) {
		if (dst->data != src0->data) {
			// src0 is same shape as dst => same indices
			memcpy((char*)dst->data + i1 * nb1, (char*)src0->data + i1 * nb01, nc * sizeof(float));
		}
		ggml_vec_scale_f321(nc, (float*)((char*)dst->data + i1 * nb1), v);
	}
}

void ggml_compute_forward_scale(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_scale_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
