module;
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include <format>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <type_traits>
#include "inplace_vector.hpp"

#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;

static inline bool ggml_is_padded_1d(const ggml_tensor* tensor) {
	return
		tensor->nb[0] == ggml_type_size(tensor->type) &&
		tensor->nb[2] == tensor->nb[1] * tensor->ne[1] &&
		tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

template <typename... Args>
ggml_tensor* build(bool inplace, ggml_context* ctx, ggml_tensor* a, ggml_op op, Args... args)
{
	static_assert((std::is_same<Args, ggml_tensor*>::value && ...),
		"All arguments must be of type ggml_tensor");
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result->op = op;
	(result->src.push_back(args), ...);
	return result;
}

static ggml_tensor* ggml_dup_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	bool                  inplace) {
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_DUP;
	result->src.push_back(a);

	return result;
}

static ggml_tensor* ggml_set_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	size_t                nb1,
	size_t                nb2,
	size_t                nb3,
	size_t                offset,
	bool                  inplace) {
	GGML_ASSERT(a->nelements() >= b->nelements());

	// make a view of the destination
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	GGML_ASSERT(offset < (size_t)(1 << 30));
	int32_t params[] = { (int32_t)nb1, (int32_t)nb2, (int32_t)nb3, (int32_t)offset, inplace ? 1 : 0 };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_SET;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

static ggml_tensor* ggml_cpy_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(a->nelements() == b->nelements());

	// make a view of the destination
	ggml_tensor* result = ggml_view_tensor(ctx, b);
	if (b->name.length() > 0) {
		result->set_name("{} (copy of {})", b->name, a->name);
	}
	else {
		result->set_name("{} (copy)", a->name);
	}

	result->op = GGML_OP_CPY;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

static ggml_tensor* ggml_add_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool                  inplace) {
	GGML_ASSERT(ggml_can_repeat(b, a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_ADD;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

static ggml_tensor* ggml_scale_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	float s,
	float b,
	bool inplace) {
	GGML_ASSERT(ggml_is_padded_1d(a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	float params[2] = { s, b };
	ggml_set_op_params(*result, &params, sizeof(params));

	result->op = GGML_OP_SCALE;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_dup(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace) {
	return ggml_dup_impl(ctx, a, inplace);
}

ggml_tensor* ggml_count_equal(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(ggml_are_same_shape(a, b));

	ggml_tensor* result = ctx->create(GGML_TYPE_I64, 1);

	result->op = GGML_OP_COUNT_EQUAL;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_set(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	size_t                nb1,
	size_t                nb2,
	size_t                nb3,
	size_t                offset,
	bool inplace) {
	return ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, inplace);
}

ggml_tensor* ggml_cpy(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	return ggml_cpy_impl(ctx, a, b);
}

ggml_tensor* ggml_add(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace)
{
	return ggml_add_impl(ctx, a, b, inplace);
}

ggml_tensor* ggml_mul(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace)
{
	GGML_ASSERT(ggml_can_repeat(b, a));
	return build(inplace, ctx, a, GGML_OP_MUL, a, b);
}

ggml_tensor* ggml_div(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace)
{
	GGML_ASSERT(ggml_can_repeat(b, a));
	return build(inplace, ctx, a, GGML_OP_DIV, a, b);
}

ggml_tensor* ggml_add1(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace) {
	GGML_ASSERT(ggml_is_scalar(b));
	GGML_ASSERT(ggml_is_padded_1d(a));

	return build(inplace, ctx, a, GGML_OP_ADD1, a, b);
}

ggml_tensor* ggml_scale(
	ggml_context* ctx,
	ggml_tensor* a,
	float s,
	bool inplace)
{
	return ggml_scale_impl(ctx, a, s, 0.0, inplace);
}

// normalize along rows
ggml_tensor* ggml_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps,
	bool inplace)
{
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_NORM, a);
	result->op_params[0] = std::bit_cast<uint32_t>(eps);
	return result;
}

ggml_tensor* ggml_rms_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps,
	bool inplace)
{
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_RMS_NORM, a);
	result->op_params[0] = std::bit_cast<uint32_t>(eps);
	return result;
}

ggml_tensor* ggml_ssm_conv(
	ggml_context* ctx,
	ggml_tensor* sx,
	ggml_tensor* c)
{
	GGML_ASSERT(ggml_is_3d(sx));
	GGML_ASSERT(ggml_is_matrix(c));

	const int64_t d_conv = c->ne[0];
	const int64_t d_inner = c->ne[1];
	const int64_t n_t = sx->ne[0] - d_conv + 1; // tokens per sequence
	const int64_t n_s = sx->ne[2];

	// TODO: maybe support other strides than 1?
	GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
	GGML_ASSERT(sx->ne[1] == d_inner);
	GGML_ASSERT(n_t >= 0);

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, d_inner, n_t, n_s);

	result->op = GGML_OP_SSM_CONV;
	result->src.push_back(sx);
	result->src.push_back(c);

	return result;
}

ggml_tensor* ggml_ssm_scan(
	ggml_context* ctx,
	ggml_tensor* s,
	ggml_tensor* x,
	ggml_tensor* dt,
	ggml_tensor* A,
	ggml_tensor* B,
	ggml_tensor* C,
	ggml_tensor* ids)
{
	GGML_ASSERT(ggml_is_contiguous(s));
	GGML_ASSERT(ggml_is_contiguous(dt));
	GGML_ASSERT(ggml_is_contiguous(A));
	GGML_ASSERT(x->nb[0] == ggml_type_size(x->type));
	GGML_ASSERT(B->nb[0] == ggml_type_size(B->type));
	GGML_ASSERT(C->nb[0] == ggml_type_size(C->type));
	GGML_ASSERT(x->nb[1] == x->ne[0] * x->nb[0]);
	GGML_ASSERT(B->nb[1] == B->ne[0] * B->nb[0]);
	GGML_ASSERT(C->nb[1] == C->ne[0] * C->nb[0]);
	GGML_ASSERT(ggml_are_same_shape(B, C));
	GGML_ASSERT(ids->type == GGML_TYPE_I32);

	{
		const int64_t d_state = s->ne[0];
		const int64_t head_dim = x->ne[0];
		const int64_t n_head = x->ne[1];
		const int64_t n_seq_tokens = x->ne[2];
		const int64_t n_seqs = x->ne[3];

		GGML_ASSERT(dt->ne[0] == n_head);
		GGML_ASSERT(dt->ne[1] == n_seq_tokens);
		GGML_ASSERT(dt->ne[2] == n_seqs);
		GGML_ASSERT(ggml_is_3d(dt));
		GGML_ASSERT(s->ne[1] == head_dim);
		GGML_ASSERT(s->ne[2] == n_head);
		GGML_ASSERT(B->ne[0] == d_state);
		GGML_ASSERT(B->ne[2] == n_seq_tokens);
		GGML_ASSERT(B->ne[3] == n_seqs);
		GGML_ASSERT(ids->ne[0] == n_seqs);
		GGML_ASSERT(ggml_is_vector(ids));
		GGML_ASSERT(A->ne[1] == n_head);
		GGML_ASSERT(ggml_is_matrix(A));

		if (A->ne[0] != 1) {
			// Mamba-1 has more granular decay factors
			GGML_ASSERT(A->ne[0] == d_state);
		}
	}

	// concatenated y + ssm_states
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, x->nelements() + s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]);

	result->op = GGML_OP_SSM_SCAN;
	result->src.push_back(s);
	result->src.push_back(x);
	result->src.push_back(dt);
	result->src.push_back(A);
	result->src.push_back(B);
	result->src.push_back(C);
	result->src.push_back(ids);

	return result;
}

ggml_tensor* ggml_rwkv_wkv6(
	ggml_context* ctx,
	ggml_tensor* k,
	ggml_tensor* v,
	ggml_tensor* r,
	ggml_tensor* tf,
	ggml_tensor* td,
	ggml_tensor* state)
{
	GGML_ASSERT(ggml_is_contiguous(k));
	GGML_ASSERT(ggml_is_contiguous(v));
	GGML_ASSERT(ggml_is_contiguous(r));
	GGML_ASSERT(ggml_is_contiguous(tf));
	GGML_ASSERT(ggml_is_contiguous(td));
	GGML_ASSERT(ggml_is_contiguous(state));

	const int64_t S = k->ne[0];
	const int64_t H = k->ne[1];
	const int64_t n_tokens = k->ne[2];
	const int64_t n_seqs = state->ne[1];
	{
		GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
		GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
		GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
		GGML_ASSERT(state->nelements() == S * S * H * n_seqs);
	}

	// concat output and new_state
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, S * H, n_tokens + S * n_seqs, 1, 1);

	result->op = GGML_OP_RWKV_WKV6;
	result->src.push_back(k);
	result->src.push_back(v);
	result->src.push_back(r);
	result->src.push_back(tf);
	result->src.push_back(td);
	result->src.push_back(state);

	return result;
}

ggml_tensor * ggml_gated_linear_attn(
        ggml_context * ctx,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * q,
        ggml_tensor  * g,
        ggml_tensor  * state,
        float scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    const int64_t S = k->ne[0];
    const int64_t H = k->ne[1];
    const int64_t n_tokens = k->ne[2];
    const int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(state->nelements() == S * S * H * n_seqs);
    }

    // concat output and new_state
    ggml_tensor * result = ctx->create(GGML_TYPE_F32, S * H, n_tokens + S * n_seqs, 1, 1);

	result->op_params[0] = std::bit_cast<uint32_t>(scale);

    result->op     = GGML_OP_GATED_LINEAR_ATTN;
    result->src.push_back(k);
    result->src.push_back(v);
    result->src.push_back(q);
    result->src.push_back(g);
    result->src.push_back(state);

    return result;
}

// ggml_mul_mat_id

/*
	c = ggml_mul_mat_id(ctx, as, b, ids);

	as  -> [cols, rows, n_expert]
	ids -> [n_experts_used, n_tokens] (i32)
	b   -> [cols, n_expert_used, n_tokens]
	c   -> [rows, n_expert_used, n_tokens]

	in b, n_experts_used can be broadcasted to match the n_expert_used of ids

	c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
ggml_tensor* ggml_mul_mat_id(
	ggml_context* ctx,
	ggml_tensor* as,
	ggml_tensor* b,
	ggml_tensor* ids)
{
	GGML_ASSERT(!ggml_is_transposed(as));
	GGML_ASSERT(ids->type == GGML_TYPE_I32);

	GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
	GGML_ASSERT(b->ne[3] == 1); // b is 3d
	GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
	GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
	GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
	GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, as->ne[1], ids->ne[0], b->ne[2], 1);

	result->op = GGML_OP_MUL_MAT_ID;
	result->src.push_back(as);
	result->src.push_back(b);
	result->src.push_back(ids);

	return result;
}

static inline bool ggml_can_out_prod(const ggml_tensor* t0, const ggml_tensor* t1) {
	return (t0->ne[1] == t1->ne[1]) &&
		(t1->ne[2] % t0->ne[2] == 0) && // verify t0 is broadcastable
		(t1->ne[3] % t0->ne[3] == 0);
}

ggml_tensor* ggml_out_prod(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(ggml_can_out_prod(a, b));
	GGML_ASSERT(!ggml_is_transposed(a));

	// a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, a->ne[0], b->ne[0], b->ne[2], b->ne[3]);

	result->op = GGML_OP_OUT_PROD;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_sqr(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return build(inplace, ctx, a, GGML_OP_SQR, a);
}

ggml_tensor* ggml_sqrt(
	ggml_context* ctx,
	ggml_tensor* a) {
	return build(false, ctx, a, GGML_OP_SQRT, a);
}

ggml_tensor* ggml_log(
	ggml_context* ctx,
	ggml_tensor* a) {
	return build(false, ctx, a, GGML_OP_LOG, a);
}

ggml_tensor* ggml_sin(
	ggml_context* ctx,
	ggml_tensor* a) {
	return build(false, ctx, a, GGML_OP_SIN, a);
}

ggml_tensor* ggml_cos(
	ggml_context* ctx,
	ggml_tensor* a) {
	return build(false, ctx, a, GGML_OP_COS, a);
}

ggml_tensor* ggml_clamp(
	ggml_context* ctx,
	ggml_tensor* a,
	float min,
	float max)
{
	// TODO: when implement backward, fix this:
	ggml_tensor* result = ggml_view_tensor(ctx, a);

	float params[] = { min, max };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_CLAMP;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_diag_mask_inf(
	ggml_context* ctx,
	ggml_tensor* a,
	int n_past,
	bool inplace)
{
	auto result = build(inplace, ctx, a, GGML_OP_DIAG_MASK_INF, a);
	result->op_params[0] = std::bit_cast<uint32_t>(n_past);
	return result;
}

static ggml_tensor* ggml_soft_max_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* mask,
	float scale,
	float max_bias,
	bool inplace) {
	GGML_ASSERT(ggml_is_contiguous(a));

	if (mask) {
		GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
		GGML_ASSERT(ggml_is_contiguous(mask));
		GGML_ASSERT(mask->ne[0] == a->ne[0]);
		GGML_ASSERT(mask->ne[1] >= a->ne[1]);
		GGML_ASSERT(a->ne[2] % mask->ne[2] == 0);
		GGML_ASSERT(a->ne[3] % mask->ne[3] == 0);
	}

	if (max_bias > 0.0f) {
		GGML_ASSERT(mask);
	}

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	float params[] = { scale, max_bias };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_SOFT_MAX;
	result->src.push_back(a);
	result->src.push_back(mask);
	result->src.push_back(nullptr); // placeholder for sink

	return result;
}

ggml_tensor* ggml_soft_max(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace,
	ggml_tensor* mask,
	float scale,
	float max_bias)
{
	return ggml_soft_max_impl(ctx, a, mask, scale, max_bias, inplace);
}

static struct ggml_tensor* ggml_rope_impl(
	struct ggml_context* ctx,
	struct ggml_tensor* a,
	struct ggml_tensor* b,
	struct ggml_tensor* c,
	int                   n_dims,
	int                   sections[GGML_MROPE_SECTIONS],
	int                   mode,
	int                   n_ctx_orig,
	float                 freq_base,
	float                 freq_scale,
	float                 ext_factor,
	float                 attn_factor,
	float                 beta_fast,
	float                 beta_slow,
	bool                  inplace) {
	GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

	GGML_ASSERT(ggml_is_vector(b));
	GGML_ASSERT(b->type == GGML_TYPE_I32);

	bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
	if (mrope_used) {
		GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
	}
	else {
		GGML_ASSERT(a->ne[2] == b->ne[0]);
	}

	if (c) {
		GGML_ASSERT(c->type == GGML_TYPE_F32);
		GGML_ASSERT(c->ne[0] >= n_dims / 2);
	}

	struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
	memcpy(params + 5, &freq_base, sizeof(float));
	memcpy(params + 6, &freq_scale, sizeof(float));
	memcpy(params + 7, &ext_factor, sizeof(float));
	memcpy(params + 8, &attn_factor, sizeof(float));
	memcpy(params + 9, &beta_fast, sizeof(float));
	memcpy(params + 10, &beta_slow, sizeof(float));
	if (mrope_used && sections) {
		memcpy(params + 11, sections, sizeof(int32_t) * GGML_MROPE_SECTIONS);
	}
	else {
		memset(params + 11, 0, sizeof(int32_t) * GGML_MROPE_SECTIONS);
	}
	memcpy(result->op_params, params, sizeof(params));

	result->op = GGML_OP_ROPE;
	result->src.push_back(a);
	result->src.push_back(b);
	result->src.push_back(c);

	return result;
}

ggml_tensor* ggml_rope_multi(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* c,
	int n_dims,
	int sections[GGML_MROPE_SECTIONS],
	int mode,
	int n_ctx_orig,
	float freq_base,
	float freq_scale,
	float ext_factor,
	float attn_factor,
	float beta_fast,
	float beta_slow,
	bool inplace) {
	return ggml_rope_impl(
		ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
		ext_factor, attn_factor, beta_fast, beta_slow, inplace
	);
}

ggml_tensor* ggml_rope_ext(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* c,
	int n_dims,
	int mode,
	int n_ctx_orig,
	float freq_base,
	float freq_scale,
	float ext_factor,
	float attn_factor,
	float beta_fast,
	float beta_slow,
	bool inplace) {
	return ggml_rope_impl(
		ctx, a, b, c, n_dims, NULL, mode, n_ctx_orig, freq_base, freq_scale,
		ext_factor, attn_factor, beta_fast, beta_slow, inplace
	);
}

ggml_tensor* ggml_concat(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int dim)
{
	GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
	GGML_ASSERT(a->type == b->type);

	int64_t ne[GGML_MAX_DIMS];
	for (int d = 0; d < GGML_MAX_DIMS; ++d) {
		if (d == dim) {
			ne[d] = a->ne[d] + b->ne[d];
			continue;
		}
		GGML_ASSERT(a->ne[d] == b->ne[d]);
		ne[d] = a->ne[d];
	}

	ggml_tensor* result = ctx->create(a->type, ne);

	result->op_params[0] = std::bit_cast<uint32_t>(dim);
	result->op = GGML_OP_CONCAT;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_argsort(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_sort_order order)
{
	GGML_ASSERT(a->ne[0] <= std::numeric_limits<int32_t>::max());
	ggml_tensor* result = ctx->create(GGML_TYPE_I32, a->ne);

	result->op_params[0] = std::bit_cast<int32_t>(order);

	result->op = GGML_OP_ARGSORT;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_sum(
	ggml_context* ctx,
	ggml_tensor* a) {

	ggml_tensor* result = ctx->create(a->type, 1);

	result->op = GGML_OP_SUM;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_sum_rows(
	ggml_context* ctx,
	ggml_tensor* a)
{
	ggml_tensor* result = ctx->create(a->type, 1, a->ne[1], a->ne[2], a->ne[3]);

	result->op = GGML_OP_SUM_ROWS;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_mean(
	ggml_context* ctx,
	ggml_tensor* a) {
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, 1, a->ne[1], a->ne[2], a->ne[3]);

	result->op = GGML_OP_MEAN;
	result->src.push_back(a);

	return result;
}

static ggml_tensor* ggml_upscale_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	int ne0,
	int ne1,
	int ne2,
	int ne3,
	ggml_scale_mode mode) {
	GGML_ASSERT(a->ne[0] <= ne0);
	GGML_ASSERT(a->ne[1] <= ne1);
	GGML_ASSERT(a->ne[2] <= ne2);
	GGML_ASSERT(a->ne[3] <= ne3);

	ggml_tensor* result = ctx->create(a->type, ne0, ne1, ne2, ne3);
	result->op_params[0] = std::bit_cast<uint32_t>(mode);

	result->op = GGML_OP_UPSCALE;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_upscale(
	ggml_context* ctx,
	ggml_tensor* a,
	int scale_factor,
	ggml_scale_mode mode) {
	return ggml_upscale_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

ggml_tensor* ggml_group_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	int n_groups,
	float eps,
	bool inplace)
{
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_GROUP_NORM, a);

	result->op_params[0] = std::bit_cast<uint32_t>(n_groups);
	result->op_params[1] = std::bit_cast<uint32_t>(eps);
	return result;
}

ggml_tensor* ggml_acc(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	std::initializer_list<int32_t> offset,
	bool inplace) {
	GGML_ASSERT(b->nelements() <= a->nelements());
	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(a->type == GGML_TYPE_F32);
	GGML_ASSERT(b->type == GGML_TYPE_F32);
	GGML_ASSERT(offset.size() == 4);

	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_ACC, a, b);
	cpp26::inplace_vector<int32_t, 4> offsets(offset.begin(), offset.end());
	uint32_t params[] = {
		static_cast<uint32_t>(offsets[0]),
		static_cast<uint32_t>(offsets[1]),
		static_cast<uint32_t>(offsets[2]),
		static_cast<uint32_t>(offsets[3])
	};
	ggml_set_op_params(*result, params, sizeof(params));
	return result;
}

ggml_tensor* ggml_pad(
	ggml_context* ctx,
	ggml_tensor* a,
	int p0,
	int p1,
	int p2,
	int p3,
	bool circular) {
	return ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3, circular);
}

ggml_tensor* ggml_timestep_embedding(
	ggml_context* ctx,
	ggml_tensor* timesteps,
	int dim,
	int max_period) {

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, dim, timesteps->ne[0]);

	result->op_params[0] = std::bit_cast<uint32_t>(dim);
	result->op_params[1] = std::bit_cast<uint32_t>(max_period);
	result->op = GGML_OP_TIMESTEP_EMBEDDING;
	result->src.push_back(timesteps);

	return result;
}

ggml_tensor* ggml_leaky_relu(
	ggml_context* ctx,
	ggml_tensor* a,
	float negative_slope,
	bool inplace) {
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_LEAKY_RELU, a);

	result->op_params[0] = std::bit_cast<uint32_t>(negative_slope);
	return result;
}

static inline bool ggml_can_mul_mat(const ggml_tensor& t0, const ggml_tensor& t1) {
	return (t0.ne[0] == t1.ne[0]) &&
		(t1.ne[2] % t0.ne[2] == 0) && // verify t0 is broadcastable
		(t1.ne[3] % t0.ne[3] == 0);
}

ggml_tensor* ggml_flash_attn_ext(
	ggml_context* ctx,
	ggml_tensor* q,
	ggml_tensor* k,
	ggml_tensor* v,
	ggml_tensor* mask,
	float scale,
	float max_bias,
	float logit_softcap)
{
	GGML_ASSERT(ggml_can_mul_mat(*k, *q));
	// TODO: check if vT can be multiplied by (k*qT)

	GGML_ASSERT(q->ne[3] == k->ne[3]);
	GGML_ASSERT(q->ne[3] == v->ne[3]);

	if (mask) {
		GGML_ASSERT(ggml_is_contiguous(mask));
		//GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

		GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
		GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
	}

	if (max_bias > 0.0f) {
		GGML_ASSERT(mask);
	}

	// permute(0, 2, 1, 3)
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, v->ne[0], q->ne[2], q->ne[1], q->ne[3]);

	float params[] = { scale, max_bias, logit_softcap };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_FLASH_ATTN_EXT;
	result->src.push_back(q);
	result->src.push_back(k);
	result->src.push_back(v);
	result->src.push_back(mask);
	result->src.push_back(nullptr); // placeholder for sink

	return result;
}

ggml_tensor* ggml_mul_mat(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	GGML_ASSERT(ggml_can_mul_mat(*a, *b));
	GGML_ASSERT(!ggml_is_transposed(a));

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3]);

	result->op = GGML_OP_MUL_MAT;
	result->src.emplace_back(a);
	result->src.emplace_back(b);

	return result;
}

ggml_tensor* ggml_conv_1d(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int stride_w,
	int padding_w,
	int dilation_w) {
	ggml_tensor* im2col = ggml_im2col(ctx, a, b, { 0, stride_w }, { 0, padding_w }, { 0, dilation_w }, false, GGML_TYPE_F16); // [N, OL, IC * K]

	ggml_tensor* result =
		ggml_mul_mat(ctx,
			ggml_reshape(ctx, im2col, { im2col->ne[0], (im2col->ne[2] * im2col->ne[1]) }), // [N, OL, IC * K] => [N*OL, IC * K]
			ggml_reshape(ctx, a, { (a->ne[0] * a->ne[1]), a->ne[2] }));                    // [OC¡AIC, K] => [OC, IC * K]

	result = ggml_reshape(ctx, result, { im2col->ne[1], a->ne[2], im2col->ne[2] }); // [N, OC, OL]

	return result;
}

static int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
	return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

ggml_tensor* ggml_conv_2d(
	ggml_context* ctx,
	ggml_tensor* kernel,
	ggml_tensor* input,
	std::pair<int, int> stride,
	std::pair<int, int> padding,
	std::pair<int, int> dilation,
	bool direct)
{
	if (direct) {
		GGML_ASSERT(kernel->ne[2] == input->ne[2]);

		const int64_t N = input->ne[3];
		const int64_t IH = input->ne[1];
		const int64_t IW = input->ne[0];
		const int64_t COut = kernel->ne[3];
		const int64_t KH = kernel->ne[1];
		const int64_t KW = kernel->ne[0];

		auto [stride_h, stride_w] = stride;;
		auto [padding_h, padding_w] = padding;
		auto [dilation_h, dilation_w] = dilation;

		int64_t ne[4];
		ne[0] = ggml_calc_conv_output_size(IW, KW, stride_w, padding_w, dilation_w);
		ne[1] = ggml_calc_conv_output_size(IH, KH, stride_h, padding_h, dilation_h);
		ne[2] = COut;
		ne[3] = N;

		ggml_tensor* result = ctx->create(input->type, ne);

		result->op_params[0] = std::bit_cast<int32_t>(stride_w);
		result->op_params[1] = std::bit_cast<int32_t>(stride_h);
		result->op_params[2] = std::bit_cast<int32_t>(padding_w);
		result->op_params[3] = std::bit_cast<int32_t>(padding_h);
		result->op_params[4] = std::bit_cast<int32_t>(dilation_w);
		result->op_params[5] = std::bit_cast<int32_t>(dilation_h);

		result->op = GGML_OP_CONV_2D;
		result->src.push_back(kernel);
		result->src.push_back(input);

		return result;
	}
	else {
		ggml_tensor* im2col = ggml_im2col(ctx, kernel, input, stride, padding, dilation, true, kernel->type); // [N, OH, OW, IC * KH * KW]

		ggml_tensor* result =
			ggml_mul_mat(ctx,
				ggml_reshape(ctx, im2col, { im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1] }), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
				ggml_reshape(ctx, kernel, { (kernel->ne[0] * kernel->ne[1] * kernel->ne[2]), kernel->ne[3] }));                       // [OC¡AIC, KH, KW] => [OC, IC * KH * KW]

		result = ggml_reshape(ctx, result, { im2col->ne[1], im2col->ne[2], im2col->ne[3], kernel->ne[3] }); // [OC, N, OH, OW]
		result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]

		return result;
	}
}

ggml_tensor* ggml_cross_entropy_loss(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(ggml_are_same_shape(a, b));

	ggml_tensor* result = ctx->create(a->type, 1);

	result->op = GGML_OP_CROSS_ENTROPY_LOSS;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_opt_step_adamw(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* grad,
	ggml_tensor* m,
	ggml_tensor* v,
	ggml_tensor* adamw_params) {
	GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
	GGML_ASSERT(ggml_are_same_shape(a, grad));
	GGML_ASSERT(ggml_are_same_shape(a, m));
	GGML_ASSERT(ggml_are_same_shape(a, v));
	GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
	GGML_ASSERT(adamw_params->nelements() == 7);

	ggml_tensor* result = ggml_view_tensor(ctx, a);

	result->op = GGML_OP_OPT_STEP_ADAMW;
	result->src.push_back(a);
	result->src.push_back(grad);
	result->src.push_back(m);
	result->src.push_back(v);
	result->src.push_back(adamw_params);

	return result;
}

ggml_tensor* ggml_reshape(
	ggml_context* ctx,
	ggml_tensor* a,
	std::initializer_list<int64_t> ne) {
	int64_t nelements = 1;
	for (auto v : ne) nelements *= v;
	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(a->nelements() == nelements);

	ggml_tensor* result = ctx->create(a->type, ne, a, 0);
	result->set_name("{} (reshaped)", a->name);

	result->op = GGML_OP_RESHAPE;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_reshape(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(ggml_is_contiguous(a));
	// as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
	GGML_ASSERT(a->nelements() == b->nelements());

	ggml_tensor* result = ctx->create(a->type, b->ne, a, 0);
	result->set_name("{} (reshaped)", a->name);

	result->op = GGML_OP_RESHAPE;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_cont(
	ggml_context* ctx,
	ggml_tensor* a,
	std::initializer_list<int64_t> ne) {

	int64_t nelements = 1;
	for (auto v : ne) nelements *= v;
	GGML_ASSERT(a->nelements() == nelements);

	ggml_tensor* result = ctx->create(a->type, ne);
	result->set_name("{} (cont)", a->name);

	result->op = GGML_OP_CONT;
	result->src.push_back(a);

	return result;
}

static ggml_tensor* ggml_unary_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_unary_op op,
	bool inplace) {
	GGML_ASSERT(ggml_is_contiguous_rows(a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<int32_t>(op);

	result->op = GGML_OP_UNARY;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_silu(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_SILU, inplace);
}

ggml_tensor* ggml_gelu(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_GELU, inplace);
}

ggml_tensor* ggml_gelu_quick(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_GELU_QUICK, inplace);
}

static ggml_tensor* ggml_l2_norm_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps,
	bool inplace) {
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<uint32_t>(eps);

	result->op = GGML_OP_L2_NORM;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_l2_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps) {
	return ggml_l2_norm_impl(ctx, a, eps, false);
}

ggml_tensor* ggml_rwkv_wkv7(
	ggml_context* ctx,
	ggml_tensor* r,
	ggml_tensor* w,
	ggml_tensor* k,
	ggml_tensor* v,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* state) {
	GGML_ASSERT(ggml_is_contiguous(r));
	GGML_ASSERT(ggml_is_contiguous(w));
	GGML_ASSERT(ggml_is_contiguous(k));
	GGML_ASSERT(ggml_is_contiguous(v));
	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(ggml_is_contiguous(b));
	GGML_ASSERT(ggml_is_contiguous(state));

	const int64_t S = k->ne[0];
	const int64_t H = k->ne[1];
	const int64_t n_tokens = k->ne[2];
	const int64_t n_seqs = state->ne[1];
	{
		GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
		GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
		GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
		GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
		GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
		GGML_ASSERT(state->nelements() == S * S * H * n_seqs);
	}

	// concat output and new_state
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, S * H, n_tokens + S * n_seqs, 1, 1);

	result->op = GGML_OP_RWKV_WKV7;
	result->src.push_back(r);
	result->src.push_back(w);
	result->src.push_back(k);
	result->src.push_back(v);
	result->src.push_back(a);
	result->src.push_back(b);
	result->src.push_back(state);
	return result;
}

ggml_tensor* ggml_sub(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace)
{
	GGML_ASSERT(ggml_can_repeat(b, a));
	return build(inplace, ctx, a, GGML_OP_SUB, a, b);;
}

ggml_tensor* ggml_repeat(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	GGML_ASSERT(ggml_can_repeat(a, b));

	ggml_tensor* result = ctx->create(a->type, b->ne);

	result->op = GGML_OP_REPEAT;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_repeat_4d(
	ggml_context* ctx,
	ggml_tensor* a,
	int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
{
	const bool can_repeat = ggml_is_empty(a) || (
		(ne0 % a->ne[0] == 0) &&
		(ne1 % a->ne[1] == 0) &&
		(ne2 % a->ne[2] == 0) &&
		(ne3 % a->ne[3] == 0)
	);
	GGML_ASSERT(can_repeat);

	ggml_tensor* result = ctx->create(a->type, ne0, ne1, ne2, ne3);

	result->op = GGML_OP_REPEAT;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_transpose(
	ggml_context* ctx,
	ggml_tensor* a)
{
	ggml_tensor* result = ggml_view_tensor(ctx, a);
	result->set_name("{} (transposed)", a->name);

	result->ne[0] = a->ne[1];
	result->ne[1] = a->ne[0];

	result->nb[0] = a->nb[1];
	result->nb[1] = a->nb[0];

	result->op = GGML_OP_TRANSPOSE;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_cont(
	ggml_context* ctx,
	ggml_tensor* a)
{
	ggml_tensor* result = ggml_dup_tensor(ctx, a);
	result->set_name("{} (cont)", a->name);

	result->op = GGML_OP_CONT;
	result->src.emplace_back(a);

	return result;
}

ggml_tensor* ggml_gelu_erf(
	ggml_context* ctx,
	ggml_tensor* a)
{
	return ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

ggml_tensor* ggml_unary(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_unary_op    op) {
	return ggml_unary_impl(ctx, a, op, false);
}

ggml_tensor* ggml_permute(
	ggml_context* ctx,
	ggml_tensor* a,
	int axis0,
	int axis1,
	int axis2,
	int axis3)
{
	GGML_ASSERT(axis0 >= 0 && axis0 < GGML_MAX_DIMS);
	GGML_ASSERT(axis1 >= 0 && axis1 < GGML_MAX_DIMS);
	GGML_ASSERT(axis2 >= 0 && axis2 < GGML_MAX_DIMS);
	GGML_ASSERT(axis3 >= 0 && axis3 < GGML_MAX_DIMS);

	GGML_ASSERT(axis0 != axis1);
	GGML_ASSERT(axis0 != axis2);
	GGML_ASSERT(axis0 != axis3);
	GGML_ASSERT(axis1 != axis2);
	GGML_ASSERT(axis1 != axis3);
	GGML_ASSERT(axis2 != axis3);

	ggml_tensor* result = ggml_view_tensor(ctx, a);
	result->set_name("{} (permuted)", a->name);

	int ne[GGML_MAX_DIMS];
	int nb[GGML_MAX_DIMS];

	ne[axis0] = a->ne[0];
	ne[axis1] = a->ne[1];
	ne[axis2] = a->ne[2];
	ne[axis3] = a->ne[3];

	nb[axis0] = a->nb[0];
	nb[axis1] = a->nb[1];
	nb[axis2] = a->nb[2];
	nb[axis3] = a->nb[3];

	result->ne[0] = ne[0];
	result->ne[1] = ne[1];
	result->ne[2] = ne[2];
	result->ne[3] = ne[3];

	result->nb[0] = nb[0];
	result->nb[1] = nb[1];
	result->nb[2] = nb[2];
	result->nb[3] = nb[3];

	result->op = GGML_OP_PERMUTE;
	result->src.push_back(a);

	int32_t params[] = { axis0, axis1, axis2, axis3 };
	ggml_set_op_params(*result, params, sizeof(params));

	return result;
}

static ggml_tensor* ggml_sub_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace) {
	GGML_ASSERT(ggml_can_repeat(b, a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_SUB;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_abs(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ggml_tensor* ggml_sgn(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

ggml_tensor* ggml_step(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

ggml_tensor* ggml_rope(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int n_dims,
	int mode,
	bool inplace) {
	return ggml_rope_impl(
		ctx, a, b, nullptr, n_dims, nullptr, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, inplace
	);
}

ggml_tensor* ggml_conv_2d_dw(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	std::pair<int, int> stride,
	std::pair<int, int>	padding,
	std::pair<int, int> dilation,
	bool direct) {

	if (direct) {
		GGML_ASSERT(a->ne[2] == 1);
		GGML_ASSERT(a->ne[3] == b->ne[2]);

		auto [stride_h, stride_w] = stride;
		auto [padding_h, padding_w] = padding;
		auto [dilation_h, dilation_w] = dilation;

		ggml_tensor* result = ctx->create(b->type,
			ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride_w, padding_w, dilation_w),
			ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride_h, padding_h, dilation_h),
			b->ne[2],
			b->ne[3]
		);

		if (ggml_is_contiguous_channels(b)) {
			// Result will be permuted the same way as input (NHWC order)
			const int64_t type_size = ggml_type_size(result->type);
			GGML_ASSERT(ggml_blck_size(result->type) == 1);
			result->nb[0] = result->ne[2] * type_size;
			result->nb[1] = result->ne[0] * result->nb[0];
			result->nb[2] = type_size;
		}

		int32_t params[] = { stride_w, stride_h, padding_w, padding_h, dilation_w, dilation_h };
		ggml_set_op_params(*result, params, sizeof(params));

		result->op = GGML_OP_CONV_2D_DW;
		result->src.push_back(a);
		result->src.push_back(b);
		return result;
	}
	else {
		ggml_tensor* new_a = ggml_reshape(ctx, a, { a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3] });
		ggml_tensor* im2col = ggml_im2col(ctx, new_a,
			ggml_reshape(ctx, b, { b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3] }),
			stride, padding, dilation, true, GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
		ggml_tensor* new_b = ggml_reshape(ctx, im2col, { im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3] }); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

		new_a = ggml_reshape(ctx, new_a, { (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3], 1 });                       // [OC¡A1, KH, KW] => [1, OC, 1, KH * KW]
		ggml_tensor* result = ggml_mul_mat(ctx, new_a, new_b);
		result = ggml_reshape(ctx, result, { im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3] }); // [N, OC, OH, OW]
		return result;
	}
}

ggml_tensor* ggml_conv_1d_dw(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int stride_w,
	int padding_w,
	int dilation_w) {
	ggml_tensor* new_a = ggml_reshape(ctx, a, { a->ne[0], 1, a->ne[1], a->ne[2] });
	ggml_tensor* new_b = ggml_reshape(ctx, b, { b->ne[0], 1, b->ne[1], b->ne[2] });

	ggml_tensor* im2col = ggml_im2col(ctx, new_a, new_b, { 0, stride_w }, { 0, padding_w }, { 0, dilation_w }, false, GGML_TYPE_F16);

	ggml_tensor* result = ggml_mul_mat(ctx, im2col, a);

	result = ggml_reshape(ctx, result, { b->ne[0], b->ne[1], 1 });

	return result;
}

ggml_tensor* ggml_roll(
	ggml_context* ctx,
	ggml_tensor* a,
	int                   shift0,
	int                   shift1,
	int                   shift2,
	int                   shift3)
{
	GGML_ASSERT(a->nb[0] == ggml_type_size(a->type));
	GGML_ASSERT(abs(shift0) < a->ne[0]);
	GGML_ASSERT(abs(shift1) < a->ne[1]);
	GGML_ASSERT(abs(shift2) < a->ne[2]);
	GGML_ASSERT(abs(shift3) < a->ne[3]);

	ggml_tensor* result = ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<int32_t>(shift0);
	result->op_params[1] = std::bit_cast<int32_t>(shift1);
	result->op_params[2] = std::bit_cast<int32_t>(shift2);
	result->op_params[3] = std::bit_cast<int32_t>(shift3);

	result->op = GGML_OP_ROLL;
	result->src.push_back(a);

	return result;
}

static ggml_tensor* ggml_glu_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	enum ggml_glu_op op,
	bool swapped) {
	GGML_ASSERT(ggml_is_contiguous_1(a));

	if (b) {
		GGML_ASSERT(ggml_is_contiguous_1(b));
		GGML_ASSERT(ggml_are_same_shape(a, b));
		GGML_ASSERT(a->type == b->type);
	}

	ggml_tensor* result = ctx->create(a->type,
		b ? a->ne[0] : a->ne[0] / 2,
		a->ne[1],
		a->ne[2],
		a->ne[3]
	);

	result->op_params[0] = std::bit_cast<int32_t>(op);
	result->op_params[1] = swapped;

	result->op = GGML_OP_GLU;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_glu(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_glu_op op,
	bool swapped) {
	return ggml_glu_impl(ctx, a, NULL, op, swapped);
}

ggml_tensor* ggml_glu_split(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	enum ggml_glu_op op) {
	return ggml_glu_impl(ctx, a, b, op, false);
}

ggml_tensor* ggml_reglu(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, NULL, GGML_GLU_OP_REGLU, false);
}

ggml_tensor* ggml_reglu_swapped(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, NULL, GGML_GLU_OP_REGLU, true);
}

ggml_tensor* ggml_reglu_split(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

ggml_tensor* ggml_geglu(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, NULL, GGML_GLU_OP_GEGLU, false);
}

ggml_tensor* ggml_geglu_swapped(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, NULL, GGML_GLU_OP_GEGLU, true);
}

ggml_tensor* ggml_geglu_split(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

ggml_tensor* ggml_swiglu(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, NULL, GGML_GLU_OP_SWIGLU, false);
}

ggml_tensor* ggml_swiglu_swapped(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, NULL, GGML_GLU_OP_SWIGLU, true);
}

ggml_tensor* ggml_swiglu_split(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

ggml_tensor* ggml_set_rows(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* c)
{
	GGML_ASSERT(a->ne[0] == b->ne[0]);
	GGML_ASSERT(a->ne[2] == b->ne[2]);
	GGML_ASSERT(a->ne[3] == b->ne[3]);
	GGML_ASSERT(b->ne[1] == c->ne[0]);
	GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
	GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
	GGML_ASSERT(c->ne[3] == 1);
	GGML_ASSERT(b->type == GGML_TYPE_F32);
	GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

	GGML_ASSERT(ggml_is_contiguous_rows(a));
	GGML_ASSERT(ggml_is_contiguous_rows(b));

	ggml_tensor* result = ggml_view_tensor(ctx, a);

	result->op = GGML_OP_SET_ROWS;
	result->src.push_back(b);
	result->src.push_back(c);
	result->src.push_back(a); // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

	return result;
}

static ggml_tensor* ggml_interpolate_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	int64_t ne0,
	int64_t ne1,
	int64_t ne2,
	int64_t ne3,
	uint32_t mode) {
	GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_COUNT);
	// TODO: implement antialias for modes other than bilinear
	GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);

	ggml_tensor* result = ctx->create(a->type, ne0, ne1, ne2, ne3);

	result->op_params[0] = std::bit_cast<int32_t>(mode);

	result->op = GGML_OP_UPSCALE;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_interpolate(
	ggml_context* ctx,
	ggml_tensor* a,
	int64_t ne0,
	int64_t ne1,
	int64_t ne2,
	int64_t ne3,
	uint32_t mode)
{
	return ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

ggml_tensor* ggml_arange(
	ggml_context* ctx,
	float                 start,
	float                 stop,
	float                 step) {
	GGML_ASSERT(stop > start);

	const int64_t steps = (int64_t)ceilf((stop - start) / step);

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, steps);

	result->op_params[0] = std::bit_cast<int32_t>(start);
	result->op_params[1] = std::bit_cast<int32_t>(stop);
	result->op_params[2] = std::bit_cast<int32_t>(step);

	result->op = GGML_OP_ARANGE;

	return result;
}

ggml_tensor* ggml_get_rows(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	GGML_ASSERT(a->ne[2] == b->ne[1]);
	GGML_ASSERT(a->ne[3] == b->ne[2]);
	GGML_ASSERT(b->ne[3] == 1);
	GGML_ASSERT(b->type == GGML_TYPE_I32);

	// TODO: implement non F32 return
	enum ggml_type type = GGML_TYPE_F32;
	if (a->type == GGML_TYPE_I32) {
		type = a->type;
	}
	ggml_tensor* result = ctx->create(type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

	result->op = GGML_OP_GET_ROWS;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

static int64_t ggml_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
	return (ins + 2 * p - ks) / s + 1;
}

ggml_tensor* ggml_pool_1d(
	ggml_context* ctx,
	ggml_tensor* a,
	enum ggml_op_pool op,
	int k0,
	int s0,
	int p0)
{
	const int64_t ne0 = ggml_calc_pool_output_size(a->ne[0], k0, s0, p0);
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, ne0, a->ne[1], a->ne[2], a->ne[3]);
	GGML_ASSERT(ne0 > 0);

	int32_t params[] = { op, k0, s0, p0 };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_POOL_1D;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_pool_2d(
	ggml_context* ctx,
	ggml_tensor* a,
	enum ggml_op_pool op,
	int k0,
	int k1,
	int s0,
	int s1,
	int32_t p0,
	int32_t p1)
{
	const int64_t ne0 = ggml_calc_pool_output_size(a->ne[0], k0, s0, p0);
	const int64_t ne1 = ggml_calc_pool_output_size(a->ne[1], k1, s1, p1);
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, ne0, ne1, a->ne[2], a->ne[3]);
	GGML_ASSERT(ne0 > 0);
	GGML_ASSERT(ne1 > 0);

	int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_POOL_2D;
	result->src.push_back(a);

	return result;
}

static ggml_tensor* ggml_view_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	std::initializer_list<int64_t> ne,
	size_t offset) {

	ggml_tensor* result = ctx->create(a->type, ne, a, offset);
	result->set_name("{} (view)", a->name);

	ggml_set_op_params(*result, &offset, sizeof(offset));

	result->op = GGML_OP_VIEW;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_view(
	ggml_context* ctx,
	ggml_tensor* a,
	std::initializer_list<int64_t> ne,
	std::initializer_list<size_t> nb,
	size_t offset) {

	assert(nb.size() + 1 == ne.size());
	ggml_tensor* result = ggml_view_impl(ctx, a, ne, offset);

	if (nb.size() == 1) {
		auto it = nb.begin();
		result->nb[1] = *it;
		result->nb[2] = result->nb[1] * result->ne[1];
		result->nb[3] = result->nb[2];
	}
	else if (nb.size() == 2) {
		auto it = nb.begin();
		result->nb[1] = *it++;
		result->nb[2] = *it++;
		result->nb[3] = result->nb[2] * result->ne[2];
	}
	else if (nb.size() == 3) {
		auto it = nb.begin();
		result->nb[1] = *it++;
		result->nb[2] = *it++;
		result->nb[3] = *it++;
	}

	return result;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC¡AIC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
ggml_tensor* ggml_im2col(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	std::pair<int, int> stride,
	std::pair<int, int> padding,
	std::pair<int, int> dilation,
	bool is_2D,
	enum ggml_type dst_type) {
	if (is_2D) {
		GGML_ASSERT(a->ne[2] == b->ne[2]);
	}
	else {
		//GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
		GGML_ASSERT(b->ne[1] == a->ne[1]);
		GGML_ASSERT(b->ne[3] == 1);
	}
	auto [stride_h, stride_w] = stride;
	auto [padding_h, padding_w] = padding;
	auto [dilation_h, dilation_w] = dilation;
	const int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride_h, padding_h, dilation_h) : 0;
	const int64_t OW = ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride_w, padding_w, dilation_w);

	GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
	GGML_ASSERT((OW > 0) && "b too small compared to a");

	ggml_tensor* result = ctx->create(dst_type,
		is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
		OW,
		is_2D ? OH : b->ne[2],
		is_2D ? b->ne[3] : 1
	);
	int32_t params[] = { stride_w, stride_h, padding_w, padding_h, dilation_w, dilation_h, (is_2D ? 1 : 0) };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_IM2COL;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_conv_transpose_1d(
	ggml_context* ctx,
	ggml_tensor* a, // [CIn, COut, K]
	ggml_tensor* b, // [CIn, L]
	int stride,
	int padding,
	int dilation)
{
	GGML_ASSERT(ggml_is_matrix(b));
	GGML_ASSERT(a->ne[2] == b->ne[1]);
	GGML_ASSERT(a->ne[3] == 1);

	const int64_t COut = a->ne[1];
	const int64_t L = b->ne[0];
	const int64_t K = a->ne[0];
	const int64_t LOut = (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, LOut, COut, 1, 1);

	int32_t params[] = { stride, padding, dilation };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_CONV_TRANSPOSE_1D;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_conv_transpose_2d(
	ggml_context* ctx,
	ggml_tensor* kernel,
	ggml_tensor* input,
	int stride,
	int padding,
	int dilation)
{
	return ggml_conv_transpose_2d(
		ctx,
		kernel,
		input,
		{ stride, stride },
		{ padding, padding },
		{ dilation, dilation });
}

ggml_tensor* ggml_conv_transpose_2d(
	ggml_context* ctx,
	ggml_tensor* kernel, // [CIn, COut, Kh, Kw]
	ggml_tensor* input, // [N, CIn, HIn, WIn]
	std::pair<int, int> stride, // (stride_h, stride_w)
	std::pair<int, int> padding, // (padding_h, padding_w)
	std::pair<int, int> dilation) // (dilation_h, dilation_w)	
{
	GGML_ASSERT(kernel->ne[3] == input->ne[2]);
	auto [stride_h, stride_w] = stride;
	auto [padding_h, padding_w] = padding;
	auto [dilation_h, dilation_w] = dilation;

	const int64_t CIn = kernel->ne[3];
	const int64_t COut = kernel->ne[2];
	const int64_t Kh = kernel->ne[1];
	const int64_t Kw = kernel->ne[0];
	const int64_t N = input->ne[3];
	const int64_t HIn = input->ne[1];
	const int64_t WIn = input->ne[0];

	ggml_tensor* result = ctx->create(GGML_TYPE_F32,
		(WIn - 1) * stride_w - 2 * padding_w + dilation_w * (Kw - 1) + 1,
		(HIn - 1) * stride_h - 2 * padding_h + dilation_h * (Kh - 1) + 1,
		COut, N
	);

	result->op_params[0] = std::bit_cast<uint32_t>(stride_w);
	result->op_params[1] = std::bit_cast<uint32_t>(stride_h);
	result->op_params[2] = std::bit_cast<uint32_t>(padding_w);
	result->op_params[3] = std::bit_cast<uint32_t>(padding_h);
	result->op_params[4] = std::bit_cast<uint32_t>(dilation_w);
	result->op_params[5] = std::bit_cast<uint32_t>(dilation_h);

	result->op = GGML_OP_CONV_TRANSPOSE_2D;
	result->src.push_back(kernel);
	result->src.push_back(input);

	return result;
}

ggml_tensor* ggml_pad_reflect_1d(
	ggml_context* ctx,
	ggml_tensor* a,
	int p0,
	int p1)
{
	GGML_ASSERT(p0 >= 0);
	GGML_ASSERT(p1 >= 0);

	GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
	GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(a->type == GGML_TYPE_F32);

	ggml_tensor* result = ctx->create(a->type,
		a->ne[0] + p0 + p1,
		a->ne[1],
		a->ne[2],
		a->ne[3]
	);

	int32_t params[] = { p0, p1 };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_PAD_REFLECT_1D;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_argmax(
	ggml_context* ctx,
	ggml_tensor* a) {
	GGML_ASSERT(ggml_is_matrix(a));
	GGML_ASSERT(a->ne[0] <= INT32_MAX);

	ggml_tensor* result = ctx->create(GGML_TYPE_I32, a->ne[1]);

	result->op = GGML_OP_ARGMAX;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_geglu_erf(
	ggml_context* ctx,
	ggml_tensor* a)
{
	return ggml_glu_impl(ctx, a, nullptr, GGML_GLU_OP_GEGLU_ERF, false);
}

ggml_tensor* ggml_geglu_erf_swapped(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, nullptr, GGML_GLU_OP_GEGLU_ERF, true);
}

ggml_tensor* ggml_geglu_quick(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, nullptr, GGML_GLU_OP_GEGLU_QUICK, false);
}

ggml_tensor* ggml_geglu_quick_swapped(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_glu_impl(ctx, a, nullptr, GGML_GLU_OP_GEGLU_QUICK, true);
}

ggml_tensor* ggml_geglu_erf_split(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

ggml_tensor* ggml_geglu_quick_split(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

ggml_tensor* ggml_scale_bias(
	ggml_context* ctx,
	ggml_tensor* a,
	float s,
	float b,
	bool inplace) {
	return ggml_scale_impl(ctx, a, s, b, inplace);
}

ggml_tensor* ggml_get_rel_pos(
	ggml_context* ctx,
	ggml_tensor* a,
	int qh,
	int kh)
{
	GGML_ASSERT(qh == kh);
	GGML_ASSERT(2 * std::max(qh, kh) - 1 == a->ne[1]);

	ggml_tensor* result = ctx->create(GGML_TYPE_F16, a->ne[0], kh, qh);

	result->op = GGML_OP_GET_REL_POS;
	result->src.push_back(a);

	return result;
}

static ggml_tensor* ggml_add_rel_pos_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* pw,
	ggml_tensor* ph,
	bool inplace) {
	GGML_ASSERT(ggml_are_same_shape(pw, ph));
	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(ggml_is_contiguous(pw));
	GGML_ASSERT(ggml_is_contiguous(ph));
	GGML_ASSERT(ph->type == GGML_TYPE_F32);
	GGML_ASSERT(pw->type == GGML_TYPE_F32);
	GGML_ASSERT(pw->ne[3] == a->ne[2]);
	GGML_ASSERT(pw->ne[0] * pw->ne[0] == a->ne[0]);
	GGML_ASSERT(pw->ne[1] * pw->ne[2] == a->ne[1]);

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result->op_params[0] = std::bit_cast<int32_t>(inplace ? 1 : 0);

	result->op = GGML_OP_ADD_REL_POS;
	result->src.push_back(a);
	result->src.push_back(pw);
	result->src.push_back(ph);

	return result;
}

ggml_tensor* ggml_add_rel_pos(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* pw,
	ggml_tensor* ph,
	bool inplace) {
	return ggml_add_rel_pos_impl(ctx, a, pw, ph, inplace);
}

ggml_tensor* ggml_map_custom(
	ggml_context* ctx,
	std::initializer_list<ggml_tensor*> srcs,
	bool inplace,
	ggml_custom_op_cb fun)
{
	assert(srcs.size() > 0);
	ggml_tensor* a = *srcs.begin();
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result->hook = std::move(fun);
	result->op = GGML_OP_CUSTOM;
	std::copy(srcs.begin(), srcs.end(), std::back_inserter(result->src));

	return result;
}

ggml_tensor* ggml_custom(
	ggml_context* ctx,
	ggml_type type,
	std::initializer_list<int64_t> ne,
	std::initializer_list<ggml_tensor*> srcs,
	ggml_custom_op_cb fun)
{
	assert(srcs.size() > 0);
	ggml_tensor* result = ctx->create(type, ne);
	result->hook = std::move(fun);
	result->op = GGML_OP_CUSTOM;
	std::copy(srcs.begin(), srcs.end(), std::back_inserter(result->src));
	return result;
}


ggml_tensor* ggml_neg(ggml_context* ctx, ggml_tensor* a)
{
	return ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

ggml_tensor* ggml_exp(ggml_context* ctx, ggml_tensor* a)
{
	return ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

ggml_tensor* ggml_cast(
	ggml_context* ctx,
	ggml_tensor* a,
	enum ggml_type type) {
	ggml_tensor* result = ctx->create(type, a->ne);
	result->set_name(std::format("{} (copy)", a->name));

	result->op = GGML_OP_CPY;
	result->src.push_back(a);
	result->src.push_back(result); // note: this self-reference might seem redundant, but it's actually needed by so
	                               //       backends for consistency with ggml_cpy_impl() above
	return result;
}

ggml_tensor* ggml_tanh(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_TANH, inplace);
}

ggml_tensor* ggml_set_1d(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	size_t offset) // in bytes
{
	return ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

ggml_tensor* ggml_sigmoid(ggml_context* ctx, ggml_tensor* a)
{
	return ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

ggml_tensor* ggml_conv_1d_ph(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s,
	int d) {
	return ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

ggml_tensor* ggml_conv_1d_dw_ph(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s0,
	int d0)
{
	return ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

ggml_tensor* ggml_relu(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_RELU, inplace);
}

ggml_tensor* ggml_top_k(
	ggml_context* ctx,
	ggml_tensor* a,
	int k)
{
	GGML_ASSERT(a->ne[0] >= k);

	ggml_tensor* result = ctx->create(GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

	result->op = GGML_OP_TOP_K;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_swiglu_oai(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	float alpha,
	float limit) {
	ggml_tensor* result = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
	result->op_params[2] = std::bit_cast<float>(alpha);
	result->op_params[3] = std::bit_cast<float>(limit);
	return result;
}

ggml_tensor* ggml_add_id(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* ids)
{
	GGML_ASSERT(a->ne[0] == b->ne[0]);
	GGML_ASSERT(a->ne[1] == ids->ne[0]);
	GGML_ASSERT(a->ne[2] == ids->ne[1]);
	GGML_ASSERT(ids->type == GGML_TYPE_I32);

	ggml_tensor* result = ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_ADD_ID;
	result->src.push_back(a);
	result->src.push_back(b);
	result->src.push_back(ids);

	return result;
}

ggml_tensor* ggml_opt_step_sgd(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* grad,
	ggml_tensor* params) {
	GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
	GGML_ASSERT(ggml_are_same_shape(a, grad));
	GGML_ASSERT(params->type == GGML_TYPE_F32);
	GGML_ASSERT(params->nelements() == 2);

	ggml_tensor* result = ggml_view_tensor(ctx, a);

	result->op = GGML_OP_OPT_STEP_SGD;
	result->src.push_back(a);
	result->src.push_back(grad);
	result->src.push_back(params);

	return result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
ggml_tensor* ggml_im2col_3d(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int64_t IC,
	int s0, // stride width
	int s1, // stride height
	int s2, // stride depth
	int p0, // padding width
	int p1, // padding height
	int p2, // padding depth
	int d0, // dilation width
	int d1, // dilation height
	int d2, // dilation depth
	ggml_type dst_type) {
	const int64_t N = b->ne[3] / IC;
	const int64_t ID = b->ne[2];
	const int64_t IH = b->ne[1];
	const int64_t IW = b->ne[0];

	const int64_t OC = a->ne[3] / IC;
	const int64_t KD = a->ne[2];
	const int64_t KH = a->ne[1];
	const int64_t KW = a->ne[0];
	const int64_t OD = ggml_calc_conv_output_size(ID, KD, s2, p2, d2);
	const int64_t OH = ggml_calc_conv_output_size(IH, KH, s1, p1, d1);
	const int64_t OW = ggml_calc_conv_output_size(IW, KW, s0, p0, d0);

	GGML_ASSERT((OD > 0) && "b too small compared to a");
	GGML_ASSERT((OH > 0) && "b too small compared to a");
	GGML_ASSERT((OW > 0) && "b too small compared to a");

	ggml_tensor* result = ctx->create(dst_type, KW * KH * KD * IC, OW, OH, OD * N);
	int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_IM2COL_3D;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_conv_3d_direct(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s0,
	int s1,
	int s2,
	int p0,
	int p1,
	int p2,
	int d0,
	int d1,
	int d2,
	int c,
	int n,
	int oc) {

	GGML_ASSERT(a->ne[3] == (int64_t)c * oc);
	GGML_ASSERT(b->ne[3] == (int64_t)c * n);

	int64_t ne[4];
	ne[0] = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
	ne[1] = ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
	ne[2] = ggml_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
	ne[3] = (int64_t)oc * n;

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, ne);

	result->op_params[0] = s0;
	result->op_params[1] = s1;
	result->op_params[2] = s2;
	result->op_params[3] = p0;
	result->op_params[4] = p1;
	result->op_params[5] = p2;
	result->op_params[6] = d0;
	result->op_params[7] = d1;
	result->op_params[8] = d2;
	result->op_params[9] = c;
	result->op_params[10] = n;
	result->op_params[11] = oc;

	result->op = GGML_OP_CONV_3D;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_pad_ext(
	ggml_context* ctx,
	ggml_tensor* a,
	int lp0,
	int rp0,
	int lp1,
	int rp1,
	int lp2,
	int rp2,
	int lp3,
	int rp3,
	bool circular
) {
	ggml_tensor* result = ctx->create(a->type,
		a->ne[0] + lp0 + rp0,
		a->ne[1] + lp1 + rp1,
		a->ne[2] + lp2 + rp2,
		a->ne[3] + lp3 + rp3);

	result->op_params[0] = lp0;
	result->op_params[1] = rp0;
	result->op_params[2] = lp1;
	result->op_params[3] = rp1;
	result->op_params[4] = lp2;
	result->op_params[5] = rp2;
	result->op_params[6] = lp3;
	result->op_params[7] = rp3;
	result->op_params[8] = circular;

	result->op = GGML_OP_PAD;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_xielu(
	ggml_context* ctx,
	ggml_tensor* a,
	float alpha_n,
	float alpha_p,
	float beta,
	float eps) {
	ggml_tensor* result = ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<int32_t>(GGML_UNARY_OP_XIELU);
	result->op_params[1] = std::bit_cast<int32_t>(beta + ggml_compute_softplus_f32(alpha_n));
	result->op_params[2] = std::bit_cast<int32_t>(ggml_compute_softplus_f32(alpha_p));
	result->op_params[3] = std::bit_cast<int32_t>(beta);
	result->op_params[4] = std::bit_cast<int32_t>(eps);

	result->op = GGML_OP_UNARY;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_floor(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_FLOOR, inplace);
}

ggml_tensor* ggml_ceil(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_CEIL, inplace);
}

ggml_tensor* ggml_round(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_ROUND, inplace);
}

ggml_tensor* ggml_trunc(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_TRUNC, inplace);
}

ggml_tensor* ggml_cumsum(
	ggml_context* ctx,
	ggml_tensor* a)
{
	GGML_ASSERT(a->type == GGML_TYPE_F32);

	ggml_tensor* result = ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_CUMSUM;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_tri(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tri_type type)
{
	GGML_ASSERT(a->type == GGML_TYPE_F32);

	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(a->ne[0] == a->ne[1]);

	ggml_tensor* result = ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<int32_t>(type);

	result->op = GGML_OP_TRI;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_fill(
	ggml_context* ctx,
	ggml_tensor* a,
	float c,
	bool inplace) {
	GGML_ASSERT(a->type == GGML_TYPE_F32);
	GGML_ASSERT(ggml_is_contiguous(a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<int32_t>(c);

	result->op = GGML_OP_FILL;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_solve_tri(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool left,
	bool lower,
	bool uni) {
	GGML_ASSERT(a->type == GGML_TYPE_F32);
	GGML_ASSERT(b->type == GGML_TYPE_F32);

	// A must be square and lower diagonal
	GGML_ASSERT(a->ne[0] == a->ne[1]);
	// B must have same outer dimension as A
	GGML_ASSERT(a->ne[1] == b->ne[1]);

	// batch dimensions must be equal
	GGML_ASSERT(a->ne[2] == b->ne[2]);
	GGML_ASSERT(a->ne[3] == b->ne[3]);

	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(ggml_is_contiguous(b));

	GGML_ASSERT(lower && left && !uni); // TODO: support other variants

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, b->ne);

	result->op = GGML_OP_SOLVE_TRI;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_expm1(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_EXPM1, inplace);
}

ggml_tensor* ggml_softplus(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace)
{
	return ggml_unary_impl(ctx, a, GGML_UNARY_OP_SOFTPLUS, inplace);
}

ggml_tensor* ggml_argsort_top_k(
	ggml_context* ctx,
	ggml_tensor* a,
	int k) {
	GGML_ASSERT(a->ne[0] >= k);

	ggml_tensor* result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

	result = ggml_view(ctx, result,
		{ k, result->ne[1], result->ne[2], result->ne[3] },
		{ result->nb[1], result->nb[2], result->nb[3] },
		0);

	return result;
}

ggml_tensor* ggml_diag(
	ggml_context* ctx,
	ggml_tensor* a) {
	GGML_ASSERT(a->ne[1] == 1);

	ggml_tensor* result = ctx->create(a->type, a->ne[0], a->ne[0], a->ne[2], a->ne[3]);

	result->op = GGML_OP_DIAG;
	result->src.push_back(a);

	return result;
}