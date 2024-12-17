module;
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <span>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml:cpu.op.unary;

import :ds;
import :tensor;
import :cpu.ds;

static float sgn(float x) {
	return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static float step(float x) {
	return (x > 0.f) ? 1.f : 0.f;
}

static float neg(float x) {
	return -x;
}

static float gelu(float x) {
	static const float GELU_COEF_A = 0.044715f;
	static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
	return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

static float gelu_quick(float x) {
	static const float GELU_QUICK_COEF = -1.702f;
	return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

// Sigmoid Linear Unit (SiLU) function
static float silu(float x) {
	return x / (1.0f + expf(-x));
}

static float sigmoid(float x) {
	return 1.f / (1.f + expf(-x));;
}

static float relu(float x) {
	return (x > 0.f) ? x : 0.f;
}

static float hardswish(float x) {
	return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static float hardsigmoid(float x) {
	return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static float elu(float x) {
	return (x > 0.f) ? x : expm1f(x);
}

template <float (*Func)(float)>
static void transform(const int n, float* y, const float* x) {
	// Wait for C++26 SIMD
	for (int i = 0; i < n; ++i) y[i] = Func(x[i]);
}

template <float (*Func)(float)>
static void ggml_compute_forward_f32(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

	assert(ggml_is_contiguous_1(src0));
	assert(ggml_is_contiguous_1(dst));
	assert(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const size_t nc = src0->ne[0];

	for (int i = 0; i < n; i++) {
		std::span<float> dst_span{ (float*)((char*)dst->data + i * (dst->nb[1])) , nc };
		std::span<float> src0_span{ (float*)((char*)src0->data + i * (src0->nb[1])) , nc };
		std::ranges::transform(src0_span, dst_span.begin(), Func);
	}
}

template <float (*Func)(float)>
static void ggml_compute_forward(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_f32<Func>(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_gelu_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_contiguous_1(src0));
	assert(ggml_is_contiguous_1(dst));
	assert(ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int i1 = ir0; i1 < ir1; i1++) {
		transform<gelu>(nc,
			(float*)((char*)dst->data + i1 * (dst->nb[1])),
			(float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
		for (int k = 0; k < nc; k++) {
			const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
			//UNUSED(x);
			//assert(!isnan(x));
			//assert(!isinf(x));
		}
#endif
	}
}

static void ggml_compute_forward_gelu(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_gelu_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_gelu_quick_f32(
	const ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_contiguous_1(src0));
	assert(ggml_is_contiguous_1(dst));
	assert(ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int i1 = ir0; i1 < ir1; i1++) {
		transform<gelu_quick>(nc,
			(float*)((char*)dst->data + i1 * (dst->nb[1])),
			(float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
		for (int k = 0; k < nc; k++) {
			const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
			//UNUSED(x);
			//assert(!isnan(x));
			//assert(!isinf(x));
		}
#endif
	}
}

static void ggml_compute_forward_gelu_quick(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_gelu_quick_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_silu_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_contiguous_1(src0));
	assert(ggml_is_contiguous_1(dst));
	assert(ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int i1 = ir0; i1 < ir1; i1++) {
		transform<silu>(nc,
			(float*)((char*)dst->data + i1 * (dst->nb[1])),
			(float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
		for (int k = 0; k < nc; k++) {
			const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
			//UNUSED(x);
			//assert(!isnan(x));
			//assert(!isinf(x));
		}
#endif
	}
}

static void ggml_compute_forward_silu(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_silu_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

void ggml_compute_forward_unary(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_unary_op op = ggml_get_unary_op(dst);

	switch (op) {
	case GGML_UNARY_OP_ABS:
	{
		ggml_compute_forward<fabsf>(params, dst);
	} break;
	case GGML_UNARY_OP_SGN:
	{
		ggml_compute_forward<sgn>(params, dst);
	} break;
	case GGML_UNARY_OP_NEG:
	{
		ggml_compute_forward<neg>(params, dst);
	} break;
	case GGML_UNARY_OP_STEP:
	{
		ggml_compute_forward<step>(params, dst);
	} break;
	case GGML_UNARY_OP_TANH:
	{
		ggml_compute_forward<tanhf>(params, dst);
	} break;
	case GGML_UNARY_OP_ELU:
	{
		ggml_compute_forward<elu>(params, dst);
	} break;
	case GGML_UNARY_OP_RELU:
	{
		ggml_compute_forward<relu>(params, dst);
	} break;
	case GGML_UNARY_OP_SIGMOID:
	{
		ggml_compute_forward<sigmoid>(params, dst);
	} break;
	case GGML_UNARY_OP_GELU:
	{
		ggml_compute_forward_gelu(params, dst);
	} break;
	case GGML_UNARY_OP_GELU_QUICK:
	{
		ggml_compute_forward_gelu_quick(params, dst);
	} break;
	case GGML_UNARY_OP_SILU:
	{
		ggml_compute_forward_silu(params, dst);
	} break;
	case GGML_UNARY_OP_HARDSWISH:
	{
		ggml_compute_forward<hardswish>(params, dst);
	} break;
	case GGML_UNARY_OP_HARDSIGMOID:
	{
		ggml_compute_forward<hardsigmoid>(params, dst);
	} break;
	case GGML_UNARY_OP_EXP:
	{
		ggml_compute_forward<expf>(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_sqr_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = x[i] * x[i]; }

static void ggml_compute_forward_sqr_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

	assert(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	assert(dst->nb[0] == sizeof(float));
	assert(src0->nb[0] == sizeof(float));

	for (int i = 0; i < n; i++) {
		ggml_vec_sqr_f32(nc,
			(float*)((char*)dst->data + i * (dst->nb[1])),
			(float*)((char*)src0->data + i * (src0->nb[1])));
	}
}

void ggml_compute_forward_sqr(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sqr_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_sqrt_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
static void ggml_compute_forward_sqrt_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

	assert(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	assert(dst->nb[0] == sizeof(float));
	assert(src0->nb[0] == sizeof(float));

	for (int i = 0; i < n; i++) {
		ggml_vec_sqrt_f32(nc,
			(float*)((char*)dst->data + i * (dst->nb[1])),
			(float*)((char*)src0->data + i * (src0->nb[1])));
	}
}

void ggml_compute_forward_sqrt(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sqrt_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_log_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]); }

static void ggml_compute_forward_log_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

	GGML_ASSERT(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[0] == sizeof(float));

	for (int i = 0; i < n; i++) {
		ggml_vec_log_f32(nc,
			(float*)((char*)dst->data + i * (dst->nb[1])),
			(float*)((char*)src0->data + i * (src0->nb[1])));
	}
}

void ggml_compute_forward_log(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_log_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_sin_f32(const int n, float* y, const float* x) {
	for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);
}

static void ggml_compute_forward_sin_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

	GGML_ASSERT(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[0] == sizeof(float));

	for (int i = 0; i < n; i++) {
		ggml_vec_sin_f32(nc,
			(float*)((char*)dst->data + i * (dst->nb[1])),
			(float*)((char*)src0->data + i * (src0->nb[1])));
	}
}

void ggml_compute_forward_sin(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sin_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_cos_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]); }

static void ggml_compute_forward_cos_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

	GGML_ASSERT(ggml_are_same_shape(src0, dst));

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[0] == sizeof(float));

	for (int i = 0; i < n; i++) {
		ggml_vec_cos_f32(nc,
			(float*)((char*)dst->data + i * (dst->nb[1])),
			(float*)((char*)src0->data + i * (src0->nb[1])));
	}
}

void ggml_compute_forward_cos(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_cos_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
