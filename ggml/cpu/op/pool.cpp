module;
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <bit>
#include "mdspan.hpp"

#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;

template <typename src0_t>
static void ggml_compute_forward_pool_1d_sk_p0(
	const enum ggml_op_pool op,
	const int k,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), dst->ne[1], dst->ne[0]);
	std::experimental::mdspan src0_data(static_cast<const src0_t*>(src0->data), src0->ne[1], src0->ne[0]);

	const int64_t rs = dst->ne[0];

	for (int64_t i1 = 0; i1 < src0_data.extent(0); i1++) {
		int64_t j = 0;
		for (int64_t i0 = 0; i0 < rs; ++i0) {
			switch (op) {
			case GGML_OP_POOL_AVG:   dst_data[i1, i0] = 0;        break;
			case GGML_OP_POOL_MAX:   dst_data[i1, i0] = -FLT_MAX; break;
			case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
			}
			for (int ki = 0; ki < k; ++ki) {
				const float srow_j = toFloat32(src0_data[i1, j]);
				switch (op) {
				case GGML_OP_POOL_AVG:                         dst_data[i1, i0] += srow_j; break;
				case GGML_OP_POOL_MAX:   if (srow_j > dst_data[i1, i0]) dst_data[i1, i0] = srow_j; break;
				case GGML_OP_POOL_COUNT:                       GGML_ABORT("fatal error");
				}
				++j;
			}
			switch (op) {
			case GGML_OP_POOL_AVG:         dst_data[i1, i0] /= k; break;
			case GGML_OP_POOL_MAX:                       break;
			case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
			}
		}
	}
}

void ggml_compute_forward_pool_1d(ggml_tensor* dst) {
	const int32_t* opts = (const int32_t*)dst->op_params;
	enum ggml_op_pool op = std::bit_cast<ggml_op_pool>(opts[0]);
	const int k0 = opts[1];
	const int s0 = opts[2];
	const int p0 = opts[3];
	assert(p0 == 0); // padding not supported
	assert(k0 == s0); // only s = k supported

	switch (dst->src[0]->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_pool_1d_sk_p0<ggml_fp32_t>(op, k0, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_pool_1d_sk_p0<ggml_fp16_t>(op, k0, dst);
	} break;
	default:
		GGML_ABORT("unsupported type for ggml_compute_forward_pool_1d");
	}
}

template <typename src0_t>
static void ggml_compute_forward_pool_2d(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	const int32_t* opts = (const int32_t*)dst->op_params;
	auto op = std::bit_cast<ggml_op_pool>(opts[0]);
	const int k0 = opts[1];
	const int k1 = opts[2];
	const int s0 = opts[3];
	const int s1 = opts[4];
	const int p0 = opts[5];
	const int p1 = opts[6];
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), dst->ne[2], dst->ne[1], dst->ne[0]);
	std::experimental::mdspan src0_data(static_cast<const src0_t*>(src0->data), src0->ne[2], src0->ne[1], src0->ne[0]);

	const int ka = k0 * k1;
	const int offset0 = -p0;
	const int offset1 = -p1;

	for (int64_t i2 = 0; i2 < dst_data.extent(0); i2++) {
		for (int64_t i1 = 0; i1 < dst_data.extent(1); i1++) {
			for (int64_t i0 = 0; i0 < dst_data.extent(2); i0++) {
				float& out = dst_data[i2, i1, i0];
				switch (op) {
				case GGML_OP_POOL_AVG:     out = 0;        break;
				case GGML_OP_POOL_MAX:     out = -FLT_MAX; break;
				case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
				}

				const int ix = offset0 + i0 * s0;
				const int iy = offset1 + i1 * s1;

				for (int ky = 0; ky < k1; ++ky) {
					if (iy + ky < 0 || iy + ky >= src0->ne[1]) continue;
					for (int kx = 0; kx < k0; ++kx) {
						int j = ix + kx;
						if (j < 0 || j >= src0->ne[0]) continue;
						const float srow_j = toFloat32(src0_data[i2, iy + ky, j]);
						switch (op) {
						case GGML_OP_POOL_AVG:                     out += srow_j; break;
						case GGML_OP_POOL_MAX: if (srow_j > out)  out = srow_j; break;
						case GGML_OP_POOL_COUNT:               GGML_ABORT("fatal error");
						}
					}
				}
				switch (op) {
				case GGML_OP_POOL_AVG:           out /= ka; break;
				case GGML_OP_POOL_MAX:                       break;
				case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
				}
			}
		}
	}
}

void ggml_compute_forward_pool_2d(
	ggml_tensor* dst) {
	switch (dst->src[0]->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_pool_2d<ggml_fp32_t>(dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_pool_2d<ggml_fp16_t>(dst);
	} break;
	default:
		GGML_ABORT("unsupported type for ggml_compute_forward_pool_2d");
	}
}
