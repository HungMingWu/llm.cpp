module;
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#define GGML_ASSERT(...) assert(__VA_ARGS_)

export module ggml:op;
import :ds;

export {
	ggml_tensor* ggml_dup(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_count_equal(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

    ggml_tensor* ggml_set(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset);

	ggml_tensor* ggml_cpy(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_add(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_mul(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_div(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_add1(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_scale(
		ggml_context* ctx,
		ggml_tensor* a,
		float s);

	// normalize along rows
	ggml_tensor* ggml_norm(
		ggml_context* ctx,
		ggml_tensor* a,
		float eps);

	ggml_tensor* ggml_rms_norm(
		ggml_context* ctx,
		ggml_tensor* a,
		float eps);

	ggml_tensor* ggml_ssm_conv(
		ggml_context* ctx,
		ggml_tensor* sx,
		ggml_tensor* c);

	ggml_tensor* ggml_ssm_scan(
		ggml_context* ctx,
		ggml_tensor* s,
		ggml_tensor* x,
		ggml_tensor* dt,
		ggml_tensor* A,
		ggml_tensor* B,
		ggml_tensor* C);

	ggml_tensor* ggml_rwkv_wkv6(
		ggml_context* ctx,
		ggml_tensor* k,
		ggml_tensor* v,
		ggml_tensor* r,
		ggml_tensor* tf,
		ggml_tensor* td,
		ggml_tensor* state);

	ggml_tensor* ggml_gated_linear_attn(
		ggml_context* ctx,
		ggml_tensor* k,
		ggml_tensor* v,
		ggml_tensor* q,
		ggml_tensor* g,
		ggml_tensor* state,
		float scale);
}
