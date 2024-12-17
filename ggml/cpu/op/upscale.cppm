module;
#include <assert.h>
#include <stdint.h>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml:cpu.op.upscale;
import :ds;
import :tensor;
import :cpu.ds;

static void ggml_compute_forward_upscale_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(src0->type == GGML_TYPE_F32);

	const int ith = params->ith;
	const int nth = params->nth;

	const float sf0 = (float)dst->ne[0] / src0->ne[0];
	const float sf1 = (float)dst->ne[1] / src0->ne[1];
	const float sf2 = (float)dst->ne[2] / src0->ne[2];
	const float sf3 = (float)dst->ne[3] / src0->ne[3];

	// TODO: optimize

	for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
		const int64_t i03 = i3 / sf3;
		for (int64_t i2 = ith; i2 < dst->ne[2]; i2 += nth) {
			const int64_t i02 = i2 / sf2;
			for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
				const int64_t i01 = i1 / sf1;
				for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
					const int64_t i00 = i0 / sf0;

					const float* x = (float*)((char*)src0->data + i00 * src0->nb[0] + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
					float* y = (float*)((char*)dst->data + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);

					*y = *x;
				}
			}
		}
	}
}

void ggml_compute_forward_upscale(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_upscale_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
