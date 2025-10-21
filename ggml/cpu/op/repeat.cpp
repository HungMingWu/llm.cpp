module;
#include <assert.h>
#include <stdint.h>
#include "mdspan.hpp"
#include "helper.h"
#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;

template <typename T>
static void ggml_compute_forward_repeat(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_can_repeat(src0, dst));

	// guaranteed to be an integer due to the check in ggml_can_repeat
	const int nr0 = (int)(dst->ne[0] / src0->ne[0]);
	const int nr1 = (int)(dst->ne[1] / src0->ne[1]);
	const int nr2 = (int)(dst->ne[2] / src0->ne[2]);
	const int nr3 = (int)(dst->ne[3] / src0->ne[3]);

	auto dst_data = make_strided_mdspan(static_cast<T*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);

	// TODO: support for transposed / permuted tensors
	assert(dst->nb[0] == sizeof(T));
	assert(src0->nb[0] == sizeof(T));

	// TODO: maybe this is not optimal?
	for (int64_t ratio3 = 0; ratio3 < nr3; ratio3++) {
		for (int64_t k3 = 0; k3 < src0_data.extent(0); k3++) {
			const int64_t offset3 = ratio3 * src0_data.extent(0);
			for (int64_t ratio2 = 0; ratio2 < nr2; ratio2++) {
				for (int64_t k2 = 0; k2 < src0_data.extent(1); k2++) {
					const int64_t offset2 = ratio2 * src0_data.extent(1);
					for (int64_t ratio1 = 0; ratio1 < nr1; ratio1++) {
						for (int64_t k1 = 0; k1 < src0_data.extent(2); k1++) {
							const int64_t offset1 = ratio1 * src0_data.extent(2);
							for (int64_t ratio0 = 0; ratio0 < nr0; ratio0++) {
								const int64_t offset0 = ratio0 * src0_data.extent(3);
								for (int64_t k0 = 0; k0 < src0_data.extent(3); k0++) {
									dst_data[offset3 + k3, offset2 + k2, offset1 + k1, offset0 + k0] = 
										src0_data[k3, k2, k1, k0];
								}
							}
						}
					}
				}
			}
		}
	}
}

void ggml_compute_forward_repeat(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	case GGML_TYPE_I16:
	{
		ggml_compute_forward_repeat<ggml_fp16_t>(dst);
	} break;
	case GGML_TYPE_F32:
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_repeat<ggml_fp32_t>(dst);
	} break;
	// TODO: templateify the implemenation and support for I64
	//       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
	//case GGML_TYPE_I64:
	//    {
	//        ggml_compute_forward_repeat_i64(params, dst);
	//    } break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}