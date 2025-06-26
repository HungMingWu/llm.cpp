module;
#include <assert.h>
#include <bit>
#define GGML_ABORT(...)

module ggml:cpu.op.leaky_relu;
import ggml;

template <typename T>
inline static void ggml_vec_leaky_relu(const int n, T* y, const T* x, const float ns) {
	for (int i = 0; i < n; ++i) {
		ggml_fp32_t v = toFloat32(x[i]);
		y[i] = fromFloat32<T>(((v > 0.f) ? v : 0.f) + ns * ((v < 0.0f) ? v : 0.f));
	}
}

template <typename T>
static void ggml_compute_forward_leaky_relu(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_contiguous_1(src0));
	assert(ggml_is_contiguous_1(dst));
	assert(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	const float negative_slope = std::bit_cast<float>(dst->op_params[0]);

	assert(dst->nb[0] == sizeof(T));
	assert(src0->nb[0] == sizeof(T));

	for (int i = 0; i < n; i++) {
		ggml_vec_leaky_relu(nc,
			(T*)((char*)dst->data + i * (dst->nb[1])),
			(T*)((char*)src0->data + i * (src0->nb[1])), negative_slope);
	}
}

void ggml_compute_forward_leaky_relu(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_leaky_relu<ggml_fp32_t>(dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_leaky_relu<ggml_fp16_t>(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
