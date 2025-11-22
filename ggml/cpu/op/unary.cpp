module;
#include <assert.h>
#include <math.h>
#include <print>
#include "helper.h"

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;

import :cpu.op;
import :cpu.helper;

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

template <typename T, typename Func>
static void ggml_compute_forward_unary(ggml_tensor* dst, Func func) {
	const ggml_tensor* src0 = dst->src[0];

	auto dst_data = make_strided_mdspan(static_cast<T*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);

	for (auto [i03, i02, i01, i00] : 
		make_cartesian_product(dst_data.extent(0), dst_data.extent(1), dst_data.extent(2), dst_data.extent(3)))
	{
		dst_data[i03, i02, i01, i00] = fromFloat32<T>(func(toFloat32(src0_data[i03, i02, i01, i00])));
		if (
			func == &gelu ||
			func == &gelu_erf ||
			func == &gelu_quick ||
			func == &silu
		) {
#ifndef NDEBUG
			assert(!isnan(toFloat32(dst_data[i03, i02, i01, i00])));
			assert(!isinf(toFloat32(dst_data[i03, i02, i01, i00])));
#endif
		}
	}
}

template <typename Func>
static void ggml_compute_forward_unary(ggml_tensor* dst, Func func) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_unary<ggml_fp32_t, Func>(dst, func);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_unary<ggml_fp16_t, Func>(dst, func);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}


template <typename src0_t, typename dst_t, typename Op>
static void apply_unary_op(ggml_tensor* dst, Op op) {
	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));
	GGML_ASSERT(dst->nb[0] == sizeof(dst_t));
	GGML_ASSERT(src0->nb[0] == sizeof(src0_t));

	auto dst_data = make_strided_mdspan(static_cast<dst_t*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(src0->data), src0->ne, src0->nb);

	for (auto [i03, i02, i01, i00] : 
		make_cartesian_product(dst_data.extent(0), dst_data.extent(1), dst_data.extent(2), dst_data.extent(3)))
		dst_data[i03, i02, i01, i00] = fromFloat32<dst_t>(op(toFloat32(src0_data[i03, i02, i01, i00])));
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <typename Op>
static void unary_op_functor(ggml_tensor* dst, Op op) {
	const ggml_tensor* src0 = dst->src[0];

	/*  */ if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) { // all f32
		apply_unary_op<float, float>(dst, op);
	}
	else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) { // all f16
		apply_unary_op<ggml_fp16_t, ggml_fp16_t>(dst, op);
	}
	else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
		apply_unary_op<ggml_bf16_t, ggml_bf16_t>(dst, op);
	}
	else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
		apply_unary_op<ggml_bf16_t, float>(dst, op);
	}
	else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
		apply_unary_op<ggml_fp16_t, float>(dst, op);
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

static inline float op_expm1(float x) {
	return expf(x) - 1.0f;
}

static inline float op_softplus(float x) {
	return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

void ggml_compute_forward_sqr(ggml_tensor* dst) {
	unary_op_functor(dst, op_sqr);
}

void ggml_compute_forward_sqrt(ggml_tensor* dst) {
	unary_op_functor(dst, op_sqrt);
}

void ggml_compute_forward_sin(ggml_tensor* dst) {
	unary_op_functor(dst, op_sin);
}

void ggml_compute_forward_cos(ggml_tensor* dst) {
	unary_op_functor(dst, op_cos);
}

void ggml_compute_forward_log(ggml_tensor* dst) {
	unary_op_functor(dst, op_log);
}

static inline float op_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
	if (x > 0.0f) {
		return alpha_p * x * x + beta * x;
	}
	else {
		const float min_x_eps = fminf(x, eps);
		return (expm1f(min_x_eps) - x) * alpha_n + beta * x;
	}
}

void ggml_compute_forward_xielu(ggml_tensor* dst) {
	const float alpha_n = std::bit_cast<float>(dst->op_params[1]);
	float alpha_p = std::bit_cast<float>(dst->op_params[2]);
	float beta = std::bit_cast<float>(dst->op_params[3]);
	float eps = std::bit_cast<float>(dst->op_params[4]);

	const auto xielu_op = [alpha_n, alpha_p, beta, eps](float f) {
		return op_xielu(f, alpha_n, alpha_p, beta, eps);
	};
	unary_op_functor(dst, xielu_op);
}

void ggml_compute_forward_unary(
	ggml_tensor* dst) {

	const ggml_unary_op op = ggml_get_unary_op(dst);

	switch (op) {
	case GGML_UNARY_OP_ABS:
	{
		ggml_compute_forward_unary(dst, fabsf);
	} break;
	case GGML_UNARY_OP_SGN:
	{
		ggml_compute_forward_unary(dst, sgn);
	} break;
	case GGML_UNARY_OP_NEG:
	{
		ggml_compute_forward_unary(dst, neg);
	} break;
	case GGML_UNARY_OP_STEP:
	{
		ggml_compute_forward_unary(dst, step);
	} break;
	case GGML_UNARY_OP_TANH:
	{
		ggml_compute_forward_unary(dst, tanhf);
	} break;
	case GGML_UNARY_OP_ELU:
	{
		ggml_compute_forward_unary(dst, elu);
	} break;
	case GGML_UNARY_OP_RELU:
	{
		ggml_compute_forward_unary(dst, relu);
	} break;
	case GGML_UNARY_OP_SIGMOID:
	{
		ggml_compute_forward_unary(dst, sigmoid);
	} break;
	case GGML_UNARY_OP_GELU:
	{
		ggml_compute_forward_unary(dst, gelu);
	} break;
	case GGML_UNARY_OP_GELU_ERF:
	{
		ggml_compute_forward_unary(dst, gelu_erf);
	} break;
	case GGML_UNARY_OP_GELU_QUICK:
	{
		ggml_compute_forward_unary(dst, gelu_quick);
	} break;
	case GGML_UNARY_OP_SILU:
	{
		ggml_compute_forward_unary(dst, silu);
	} break;
	case GGML_UNARY_OP_HARDSWISH:
	{
		ggml_compute_forward_unary(dst, hardswish);
	} break;
	case GGML_UNARY_OP_HARDSIGMOID:
	{
		ggml_compute_forward_unary(dst, hardsigmoid);
	} break;
	case GGML_UNARY_OP_EXP:
	{
		ggml_compute_forward_unary(dst, expf);
	} break;
	case GGML_UNARY_OP_FLOOR:
	{
		ggml_compute_forward_unary(dst, floorf);
	} break;
	case GGML_UNARY_OP_CEIL:
	{
		ggml_compute_forward_unary(dst, ceilf);
	} break;
	case GGML_UNARY_OP_ROUND:
	{
		ggml_compute_forward_unary(dst, roundf);
	} break;
	case GGML_UNARY_OP_TRUNC:
	{
		ggml_compute_forward_unary(dst, truncf);
	} break;
	case GGML_UNARY_OP_XIELU:
	{
		ggml_compute_forward_xielu(dst);
	} break;
	case GGML_UNARY_OP_EXPM1:
	{
		ggml_compute_forward_unary(dst, op_expm1);
	} break;
	case GGML_UNARY_OP_SOFTPLUS:
	{
		ggml_compute_forward_unary(dst, op_softplus);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}
