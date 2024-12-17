module;
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <print>
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

static float gelu_erf(float x) {
	static const float SQRT_2_INV = 0.70710678118654752440084436210484f;
	return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
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

template <typename T, float (*Func)(float)>
static void ggml_compute_forward(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	const int ith = params->ith;
	const int nth = params->nth;

	const size_t nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int i1 = ir0; i1 < ir1; i1++) {
		std::span<T> dst_span{ (T*)((char*)dst->data + i1 * (dst->nb[1])) , nc };
		std::span<const T> src0_span{ (T*)((char*)src0->data + i1 * (src0->nb[1])) , nc };
		std::ranges::transform(src0_span, dst_span.begin(), [](auto value) {
			return fromFloat32<T>(Func(toFloat32(value)));
		});

		if constexpr (
			Func == gelu ||
			Func == gelu_erf ||
			Func == gelu_quick ||
			Func == silu
		) {
#ifndef NDEBUG
			for (auto x : dst_span) {
				assert(!isnan(toFloat32(x)));
				assert(!isinf(toFloat32(x)));
			}
#endif
		}
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
		ggml_compute_forward<ggml_fp32_t, Func>(params, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward<ggml_fp16_t, Func>(params, dst);
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
		ggml_compute_forward<gelu>(params, dst);
	} break;
	case GGML_UNARY_OP_GELU_ERF:
	{
		ggml_compute_forward<gelu_erf>(params, dst);
	} break;
	case GGML_UNARY_OP_GELU_QUICK:
	{
		ggml_compute_forward<gelu_quick>(params, dst);
	} break;
	case GGML_UNARY_OP_SILU:
	{
		ggml_compute_forward<silu>(params, dst);
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

template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t* y, const src0_t* x) {
	for (int i = 0; i < n; i++) {
		y[i] = fromFloat32<dst_t>(op(toFloat32(x[i])));
	}
}

static std::pair<int64_t, int64_t> get_thread_range1(const struct ggml_compute_params* params, const struct ggml_tensor* src0) {
	const int64_t ith = params->ith;
	const int64_t nth = params->nth;

	const int64_t nr = ggml_nrows(src0);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	return { ir0, ir1 };
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params* params, ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));
	GGML_ASSERT(dst->nb[0] == sizeof(dst_t));
	GGML_ASSERT(src0->nb[0] == sizeof(src0_t));

	const auto [ir0, ir1] = get_thread_range1(params, src0);

	for (int64_t ir = ir0; ir < ir1; ++ir) {
		const int64_t i03 = ir / (src0->ne[2] * src0->ne[1]);
		const int64_t i02 = (ir - i03 * src0->ne[2] * src0->ne[1]) / src0->ne[1];
		const int64_t i01 = (ir - i03 * src0->ne[2] * src0->ne[1] - i02 * src0->ne[1]);

		dst_t* dst_ptr = (dst_t*)((char*)dst->data + i03 * dst->nb[3] + i02 * dst->nb[2] + i01 * dst->nb[1]);
		const src0_t* src0_ptr = (const src0_t*)((const char*)src0->data + i03 * src0->nb[3] + i02 * src0->nb[2] + i01 * src0->nb[1]);

		vec_unary_op<op>(dst->ne[0], dst_ptr, src0_ptr);
	}
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params* params, ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	/*  */ if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) { // all f32
		apply_unary_op<op, float, float>(params, dst);
	}
	else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) { // all f16
		apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
	}
	else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
		apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
	}
	else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
		apply_unary_op<op, ggml_bf16_t, float>(params, dst);
	}
	else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
		apply_unary_op<op, ggml_fp16_t, float>(params, dst);
	}
	else {
		std::println(stderr, "{}: unsupported types: dst: {}, src0: {}", __func__,
			ggml_type_name(dst->type), ggml_type_name(src0->type));
		GGML_ABORT("fatal error");
	}
}

static inline float op_sqr(float x) {
	return x * x;
}

static inline float op_sqrt(float x) {
	return sqrtf(x);
}

static inline float op_sin(float x) {
	return sinf(x);
}

static inline float op_cos(float x) {
	return cosf(x);
}

static inline float op_log(float x) {
	return logf(x);
}

void ggml_compute_forward_sqr(
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	unary_op<op_log>(params, dst);
}