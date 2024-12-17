module;
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <bit>
#include <numeric>
#include <span>
#define GGML_ABORT(...)
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml:cpu.op.argsort;
import :ds;
import :tensor;
import :cpu.ds;

static void ggml_compute_forward_argsort_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = std::bit_cast<ggml_sort_order>(dst->op_params[0]);

    for (int64_t i = ith; i < nr; i += nth) {
        int32_t* dst_data = (int32_t*)((char*)dst->data + i * dst->nb[1]);
        const float* src_data = (float*)((char*)src0->data + i * src0->nb[1]);

		std::span<int32_t> dst_span(dst_data, dst->ne[0]);
		// clang doesn't support range itoa right now
		std::iota(dst_span.begin(), dst_span.end(), 0);

		std::ranges::sort(dst_span, [&src_data, order](int32_t a, int32_t b) {
			return (order == GGML_SORT_ORDER_ASC && src_data[a] < src_data[b]) ||
				(order == GGML_SORT_ORDER_DESC && src_data[a] > src_data[b]);
		});
    }
}

void ggml_compute_forward_argsort(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_argsort_f32(params, dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}
