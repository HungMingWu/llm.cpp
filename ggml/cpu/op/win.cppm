module;
#include <assert.h>
#include <stdint.h>
#define GGML_ABORT(...)

module ggml:cpu.op.win;
import :cpu.ds;

static void ggml_compute_forward_win_part_f32(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	const int32_t nep0 = ((const int32_t*)(dst->op_params))[0];
	const int32_t nep1 = ((const int32_t*)(dst->op_params))[1];
	const int32_t w = ((const int32_t*)(dst->op_params))[2];

	assert(src0->ne[0] == dst->ne[0]);
	assert(dst->ne[3] == nep0 * nep1);

	// TODO: optimize / multi-thread
	for (int py = 0; py < nep1; ++py) {
		for (int px = 0; px < nep0; ++px) {
			const int64_t i3 = py * nep0 + px;
			for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
				for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
					for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
						const int64_t i02 = py * w + i2;
						const int64_t i01 = px * w + i1;
						const int64_t i00 = i0;

						const int64_t i = i3 * dst->ne[2] * dst->ne[1] * dst->ne[0] + i2 * dst->ne[1] * dst->ne[0] + i1 * dst->ne[0] + i0;
						const int64_t j = i02 * src0->ne[1] * src0->ne[0] + i01 * src0->ne[0] + i00;

						if (py * w + i2 >= src0->ne[2] || px * w + i1 >= src0->ne[1]) {
							((float*)dst->data)[i] = 0.0f;
						}
						else {
							((float*)dst->data)[i] = ((float*)src0->data)[j];
						}
					}
				}
			}
		}
	}
}

void ggml_compute_forward_win_part(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_win_part_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_win_unpart_f32(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	const int32_t w = ((const int32_t*)(dst->op_params))[0];

	// padding
	const int px = (w - dst->ne[1] % w) % w;
	//const int py = (w - ne2%w)%w;

	const int npx = (px + dst->ne[1]) / w;
	//const int npy = (py + ne2)/w;

	assert(dst->ne[0] == src0->ne[0]);

	// TODO: optimize / multi-thread
	for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
		for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
			for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
				const int ip2 = i2 / w;
				const int ip1 = i1 / w;

				const int64_t i02 = i2 % w;
				const int64_t i01 = i1 % w;
				const int64_t i00 = i0;

				const int64_t i = (ip2 * npx + ip1) * src0->ne[2] * src0->ne[1] * src0->ne[0] + i02 * src0->ne[1] * src0->ne[0] + i01 * src0->ne[0] + i00;
				const int64_t j = i2 * dst->ne[1] * dst->ne[0] + i1 * dst->ne[0] + i0;

				((float*)dst->data)[j] = ((float*)src0->data)[i];
			}
		}
	}
}

void ggml_compute_forward_win_unpart(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_win_unpart_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}