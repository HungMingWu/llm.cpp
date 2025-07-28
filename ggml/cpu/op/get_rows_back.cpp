module;
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <span>
#include "mdspan.hpp"
#include "../helper.h"
#define GGML_ABORT(...)

module ggml;
import :ds;
import :cpu.op;

template <typename T>
static void ggml_compute_forward_get_rows_back(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	assert(ggml_is_contiguous(dst));

	// ggml_compute_forward_dup_same_cont(params, opt0, dst);

	memset(dst->data, 0, dst->nbytes());

	const int nc = src0->ne[0];
	const int nr = src1->nelements();

	assert(dst->ne[0] == nc);
	assert(src0->nb[0] == sizeof(T));

	auto dst_data = make_strided_mdspan<2>(static_cast<ggml_fp32_t*>(dst->data), dst->ne, dst->nb);;
	std::experimental::mdspan src0_data(static_cast<const ggml_fp32_t*>(src0->data), src0->ne[1], src0->ne[0]);
	std::experimental::mdspan src1_data(static_cast<const int32_t*>(src1->data), src1->ne[0]);

	for (int i = 0; i < nr; ++i) {
		const int r = src1_data[i];
		for (int j = 0; j < nc; ++j)
			dst_data[r, j] += toFloat32(src0_data[i, j]);
	}
}

void ggml_compute_forward_get_rows_back(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_get_rows_back<ggml_fp16_t>(dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_get_rows_back<ggml_fp32_t>(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}

	//static bool first = true;
	//printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
	//if (first) {
	//    first = false;
	//} else {
	//    for (int k = 0; k < dst->ne[1]; ++k) {
	//        for (int j = 0; j < dst->ne[0]/16; ++j) {
	//            for (int i = 0; i < 16; ++i) {
	//                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
	//            }
	//            printf("\n");
	//        }
	//        printf("\n");
	//    }
	//    printf("\n");
	//    exit(0);
	//}
}