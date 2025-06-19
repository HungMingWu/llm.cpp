module;
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <bit>
#include <initializer_list>
#include <limits>
#include <type_traits>
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
	GGML_ASSERT(ggml_can_repeat(*b, *a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_ADD;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

static ggml_tensor* ggml_mul_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool                  inplace) {
	GGML_ASSERT(ggml_can_repeat(*b, *a));

	struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_MUL;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

static struct ggml_tensor* ggml_div_impl(
	struct ggml_context* ctx,
	struct ggml_tensor* a,
	struct ggml_tensor* b,
	bool                  inplace) {
	GGML_ASSERT(ggml_can_repeat(*b, *a));

	struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_DIV;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_add1_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool                  inplace) {
	GGML_ASSERT(ggml_is_scalar(b));
	GGML_ASSERT(ggml_is_padded_1d(a));

	return build(inplace, ctx, a, GGML_OP_ADD1, a, b);
}

static ggml_tensor* ggml_scale_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	float s,
	bool inplace) {
	GGML_ASSERT(ggml_is_padded_1d(a));
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_SCALE, a);
	result->op_params[0] = std::bit_cast<uint32_t>(s);
	return result;
}

ggml_tensor* ggml_dup(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_dup_impl(ctx, a, false);
}

ggml_tensor* ggml_count_equal(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(ggml_are_same_shape(a, b));

	ggml_tensor* result = ctx->create(GGML_TYPE_I64, { 1 });

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
	size_t                offset) {
	return ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
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
	ggml_tensor* b)
{
	return ggml_add_impl(ctx, a, b, false);
}

ggml_tensor* ggml_mul(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	return ggml_mul_impl(ctx, a, b, false);
}

ggml_tensor* ggml_div(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	return ggml_div_impl(ctx, a, b, false);
}

ggml_tensor* ggml_add1(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_add1_impl(ctx, a, b, false);
}

ggml_tensor* ggml_scale(
	ggml_context* ctx,
	ggml_tensor* a,
	float s)
{
	return ggml_scale_impl(ctx, a, s, false);
}

static ggml_tensor* ggml_norm_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps,
	bool inplace) {
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_NORM, a);
	result->op_params[0] = std::bit_cast<uint32_t>(eps);
	return result;
}

// normalize along rows
ggml_tensor* ggml_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps)
{
	return ggml_norm_impl(ctx, a, eps, false);
}

static ggml_tensor* ggml_rms_norm_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps,
	bool inplace) {
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_RMS_NORM, a);
	result->op_params[0] = std::bit_cast<uint32_t>(eps);
	return result;
}

ggml_tensor* ggml_rms_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps)
{
	return ggml_rms_norm_impl(ctx, a, eps, false);
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
	// FIXME: this is always true?
	GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
	GGML_ASSERT(sx->ne[1] == d_inner);
	GGML_ASSERT(n_t >= 0);

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { d_inner, n_t, n_s });

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
	ggml_tensor* C)
{
	GGML_ASSERT(ggml_is_contiguous(s));
	GGML_ASSERT(ggml_is_contiguous(x));
	GGML_ASSERT(ggml_is_contiguous(dt));
	GGML_ASSERT(ggml_is_contiguous(A));
	GGML_ASSERT(ggml_is_matrix(A));
	GGML_ASSERT(ggml_is_3d(B));
	GGML_ASSERT(ggml_is_3d(s));
	GGML_ASSERT(B->nb[0] == ggml_type_size(B->type));
	GGML_ASSERT(C->nb[0] == ggml_type_size(C->type));
	GGML_ASSERT(ggml_are_same_shape(x, dt));
	GGML_ASSERT(ggml_are_same_shape(B, C));

	{
		const int64_t d_state = s->ne[0];
		const int64_t d_inner = s->ne[1];
		const int64_t n_seq_tokens = x->ne[1];
		const int64_t n_seqs = x->ne[2];

		GGML_ASSERT(s->ne[2] == n_seqs);
		GGML_ASSERT(x->ne[0] == d_inner);
		GGML_ASSERT(A->ne[0] == d_state);
		GGML_ASSERT(A->ne[1] == d_inner);
		GGML_ASSERT(B->ne[0] == d_state);
		GGML_ASSERT(B->ne[1] == n_seq_tokens);
		GGML_ASSERT(B->ne[2] == n_seqs);
	}

	// concatenated y + ssm_states
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { x->nelements() + s->nelements() });

	result->op = GGML_OP_SSM_SCAN;
	result->src.push_back(s);
	result->src.push_back(x);
	result->src.push_back(dt);
	result->src.push_back(A);
	result->src.push_back(B);
	result->src.push_back(C);

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
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { S * H, n_tokens + S * n_seqs, 1, 1 });

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
    ggml_tensor * result = ctx->create(GGML_TYPE_F32, {S * H, n_tokens + S * n_seqs, 1, 1});

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

	struct ggml_tensor* result = ctx->create(GGML_TYPE_F32, { as->ne[1], ids->ne[0], b->ne[2], 1 });

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
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { a->ne[0], b->ne[0], b->ne[2], b->ne[3] });

	result->op = GGML_OP_OUT_PROD;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_sqr(
	ggml_context* ctx,
	ggml_tensor* a)
{
	return build(false, ctx, a, GGML_OP_SQR, a);
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
	int n_past) {
	auto result = build(false, ctx, a, GGML_OP_DIAG_MASK_INF, a);
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
		GGML_ASSERT(ggml_is_matrix(mask));
		GGML_ASSERT(mask->ne[0] == a->ne[0]);
		GGML_ASSERT(mask->ne[1] >= a->ne[1]);
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

	return result;
}

ggml_tensor* ggml_soft_max(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* mask,
	float scale,
	float max_bias)
{
	return ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

ggml_tensor* ggml_rope_multi(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* c,
	int n_dims,
	int sections[4],
	int mode,
	int n_ctx_orig,
	float freq_base,
	float freq_scale,
	float ext_factor,
	float attn_factor,
	float beta_fast,
	float beta_slow) {
	// Multimodal Rotary Position Embedding
	GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

	GGML_ASSERT(ggml_is_vector(*b));
	GGML_ASSERT(b->type == GGML_TYPE_I32);
	GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token

	if (c) {
		GGML_ASSERT(c->type == GGML_TYPE_F32);
		GGML_ASSERT(c->ne[0] >= n_dims / 2);
	}

	ggml_tensor* result = ggml_dup_tensor(ctx, a);

	int32_t params[11 + 4] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
	memcpy(params + 5, &freq_base, sizeof(float));
	memcpy(params + 6, &freq_scale, sizeof(float));
	memcpy(params + 7, &ext_factor, sizeof(float));
	memcpy(params + 8, &attn_factor, sizeof(float));
	memcpy(params + 9, &beta_fast, sizeof(float));
	memcpy(params + 10, &beta_slow, sizeof(float));
	memcpy(&params[11], sections, sizeof(int) * 4);
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_ROPE;
	result->src.push_back(a);
	result->src.push_back(b);
	result->src.push_back(c);

	return result;
}

static constexpr size_t GGML_MROPE_SECTIONS = 4;

static ggml_tensor* ggml_rope_impl(
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
	GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

	GGML_ASSERT(ggml_is_vector(*b));
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

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
	memcpy(params + 5, &freq_base, sizeof(float));
	memcpy(params + 6, &freq_scale, sizeof(float));
	memcpy(params + 7, &ext_factor, sizeof(float));
	memcpy(params + 8, &attn_factor, sizeof(float));
	memcpy(params + 9, &beta_fast, sizeof(float));
	memcpy(params + 10, &beta_slow, sizeof(float));
	if (mrope_used)
		memcpy(params + 11, sections, sizeof(int32_t) * GGML_MROPE_SECTIONS);
	else
		memset(params + 11, 0, sizeof(int32_t) * GGML_MROPE_SECTIONS);
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_ROPE;
	result->src.push_back(a);
	result->src.push_back(b);
	result->src.push_back(c);

	return result;
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
	float beta_slow) {
	return ggml_rope_impl(
		ctx, a, b, c, n_dims, nullptr, mode, n_ctx_orig, freq_base, freq_scale,
		ext_factor, attn_factor, beta_fast, beta_slow, false
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

	ggml_tensor* result = ctx->create(a->type, { ne[0], ne[1], ne[2], ne[3] });

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
	ggml_tensor* result = ctx->create(GGML_TYPE_I32, { a->ne[0], a->ne[1], a->ne[2], a->ne[3] });

	result->op_params[0] = std::bit_cast<int32_t>(order);

	result->op = GGML_OP_ARGSORT;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_sum(
	ggml_context* ctx,
	ggml_tensor* a) {

	ggml_tensor* result = ctx->create(a->type, { 1 });

	result->op = GGML_OP_SUM;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_sum_rows(
	ggml_context* ctx,
	ggml_tensor* a)
{
	ggml_tensor* result = ctx->create(a->type, { 1, a->ne[1], a->ne[2], a->ne[3] });

	result->op = GGML_OP_SUM_ROWS;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_mean(
	ggml_context* ctx,
	ggml_tensor* a) {
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { 1, a->ne[1], a->ne[2], a->ne[3] });

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

	ggml_tensor* result = ctx->create(a->type, { ne0, ne1, ne2, ne3 });
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

ggml_tensor* ggml_upscale_ext(
	ggml_context* ctx,
	ggml_tensor* a,
	int ne0,
	int ne1,
	int ne2,
	int ne3,
	ggml_scale_mode mode) {
	return ggml_upscale_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

static ggml_tensor* ggml_group_norm_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	int n_groups,
	float eps,
	bool inplace) {
	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_GROUP_NORM, a);

	result->op_params[0] = std::bit_cast<uint32_t>(n_groups);
	result->op_params[1] = std::bit_cast<uint32_t>(eps);
	return result;
}

ggml_tensor* ggml_group_norm(
	ggml_context* ctx,
	ggml_tensor* a,
	int n_groups,
	float eps)
{
	return ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

static ggml_tensor* ggml_acc_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	size_t nb1,
	size_t nb2,
	size_t nb3,
	size_t offset,
	bool inplace) {
	GGML_ASSERT(b->nelements() <= a->nelements());
	GGML_ASSERT(ggml_is_contiguous(a));
	GGML_ASSERT(a->type == GGML_TYPE_F32);
	GGML_ASSERT(b->type == GGML_TYPE_F32);

	ggml_tensor* result = build(inplace, ctx, a, GGML_OP_ACC, a, b);

	uint32_t params[] = {
		static_cast<uint32_t>(nb1),
		static_cast<uint32_t>(nb2),
		static_cast<uint32_t>(nb3),
		static_cast<uint32_t>(offset),
		static_cast<uint32_t>(inplace ? 1 : 0)
	};
	ggml_set_op_params(*result, params, sizeof(params));
	return result;
}

ggml_tensor* ggml_acc(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	size_t nb1,
	size_t nb2,
	size_t nb3,
	size_t offset) {
	return ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

ggml_tensor* ggml_pad(
	ggml_context* ctx,
	ggml_tensor* a,
	int p0,
	int p1,
	int p2,
	int p3) {
	ggml_tensor* result = ctx->create(a->type,
		{ a->ne[0] + p0,
		a->ne[1] + p1,
		a->ne[2] + p2,
		a->ne[3] + p3 });

	result->op = GGML_OP_PAD;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_timestep_embedding(
	ggml_context* ctx,
	ggml_tensor* timesteps,
	int dim,
	int max_period) {
	int actual_dim = dim;
	if (dim % 2 != 0) {
		actual_dim = dim + 1;
	}

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { actual_dim, timesteps->ne[0] });

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

	if (mask) {
		GGML_ASSERT(ggml_is_contiguous(mask));
		GGML_ASSERT(mask->ne[2] == 1);
		GGML_ASSERT(mask->ne[3] == 1);
		GGML_ASSERT(mask->ne[1] >= GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD) &&
			"the Flash-Attention kernel requires the mask to be padded to GGML_KQ_MASK_PAD and at least n_queries big");
		//GGML_ASSERT(ggml_can_repeat_rows(mask, qk));
	}

	if (max_bias > 0.0f) {
		GGML_ASSERT(mask);
	}

	// permute(0, 2, 1, 3)
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { q->ne[0], q->ne[2], q->ne[1], q->ne[3] });

	float params[] = { scale, max_bias, logit_softcap };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_FLASH_ATTN_EXT;
	result->src.push_back(q);
	result->src.push_back(k);
	result->src.push_back(v);
	result->src.push_back(mask);

	return result;
}

ggml_tensor* ggml_mul_mat(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	GGML_ASSERT(ggml_can_mul_mat(*a, *b));
	GGML_ASSERT(!ggml_is_transposed(a));

	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { a->ne[1], b->ne[1], b->ne[2], b->ne[3] });

	result->op = GGML_OP_MUL_MAT;
	result->src.emplace_back(a);
	result->src.emplace_back(b);

	return result;
}

ggml_tensor* ggml_conv_1d(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s0,
	int p0,
	int d0) {
	ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F16); // [N, OL, IC * K]

	ggml_tensor* result =
		ggml_mul_mat(ctx,
			ggml_reshape(ctx, im2col, { im2col->ne[0], (im2col->ne[2] * im2col->ne[1]) }), // [N, OL, IC * K] => [N*OL, IC * K]
			ggml_reshape(ctx, a, { (a->ne[0] * a->ne[1]), a->ne[2] }));                    // [OC¡AIC, K] => [OC, IC * K]

	result = ggml_reshape(ctx, result, { im2col->ne[1], a->ne[2], im2col->ne[2] }); // [N, OC, OL]

	return result;
}

ggml_tensor* ggml_conv_2d(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s0,
	int s1,
	int p0,
	int p1,
	int d0,
	int d1) {
	ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type); // [N, OH, OW, IC * KH * KW]

	ggml_tensor* result =
		ggml_mul_mat(ctx,
			ggml_reshape(ctx, im2col, { im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1] }), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
			ggml_reshape(ctx, a, { (a->ne[0] * a->ne[1] * a->ne[2]), a->ne[3] }));                       // [OC¡AIC, KH, KW] => [OC, IC * KH * KW]

	result = ggml_reshape(ctx, result, { im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3] }); // [OC, N, OH, OW]
	result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]

	return result;
}

ggml_tensor* ggml_cross_entropy_loss(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	GGML_ASSERT(ggml_are_same_shape(a, b));

	ggml_tensor* result = ctx->create(a->type, { 1 });

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

	ggml_tensor* result = ctx->create(a->type, { b->ne[0], b->ne[1], b->ne[2], b->ne[3] }, a, 0);
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

ggml_tensor* ggml_silu(
	ggml_context* ctx,
	ggml_tensor* a)
{
	return ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

ggml_tensor* ggml_gelu(
	ggml_context* ctx,
	ggml_tensor* a)
{
	return ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
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
	ggml_tensor* result = ctx->create(GGML_TYPE_F32, { S * H, n_tokens + S * n_seqs, 1, 1 });

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
	ggml_tensor* b)
{
	GGML_ASSERT(ggml_can_repeat(*b, *a));
	return build(false, ctx, a, GGML_OP_SUB, a, b);;
}

ggml_tensor* ggml_conv_2d_dw_direct(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int stride0,
	int stride1,
	int pad0,
	int pad1,
	int dilation0,
	int dilation1) {
	GGML_ASSERT(a->ne[2] == 1);
	GGML_ASSERT(a->ne[3] == b->ne[2]);

	ggml_tensor* result = ctx->create(b->type, {
		ggml_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0),
		ggml_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1),
		b->ne[2],
		b->ne[3]
	});

	if (ggml_is_contiguous_channels(b)) {
		// Result will be permuted the same way as input (CWHN order)
		const int64_t type_size = ggml_type_size(result->type);
		GGML_ASSERT(ggml_blck_size(result->type) == 1);
		result->nb[0] = result->ne[2] * type_size;
		result->nb[1] = result->ne[0] * result->nb[0];
		result->nb[2] = type_size;
	}

	int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_CONV_2D_DW;
	result->src.push_back(a);
	result->src.push_back(b);
	return result;
}

ggml_tensor* ggml_repeat(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b)
{
	GGML_ASSERT(ggml_can_repeat(*a, *b));

	ggml_tensor* result = ctx->create(a->type, { b->ne[0], b->ne[1], b->ne[2], b->ne[3] });

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

static ggml_tensor* ggml_unary_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_unary_op    op,
	bool inplace) {
	GGML_ASSERT(ggml_is_contiguous_1(a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op_params[0] = std::bit_cast<int32_t>(op);

	result->op = GGML_OP_UNARY;
	result->src.push_back(a);

	return result;
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

static ggml_tensor* ggml_unary_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	enum ggml_unary_op op) {
	return ggml_unary_impl(ctx, a, op, true);
}

ggml_tensor* ggml_gelu_inplace(
	ggml_context* ctx,
	ggml_tensor* a)
{
	return ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

ggml_tensor* ggml_silu_inplace(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

ggml_tensor* ggml_tanh_inplace(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

ggml_tensor* ggml_relu_inplace(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

static ggml_tensor* ggml_sqr_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	bool inplace) {
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_SQR;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_sqr_inplace(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_sqr_impl(ctx, a, true);
}

ggml_tensor* ggml_scale_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	float s) {
	return ggml_scale_impl(ctx, a, s, true);
}

ggml_tensor* ggml_add_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_add_impl(ctx, a, b, true);
}

static ggml_tensor* ggml_sub_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	bool inplace) {
	GGML_ASSERT(ggml_can_repeat(*b, *a));

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	result->op = GGML_OP_SUB;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_sub_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_sub_impl(ctx, a, b, true);
}

ggml_tensor* ggml_norm_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps) {
	return ggml_norm_impl(ctx, a, eps, true);
}

ggml_tensor* ggml_rms_norm_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	float eps) {
	return ggml_rms_norm_impl(ctx, a, eps, true);
}

ggml_tensor* ggml_soft_max_inplace(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_soft_max_impl(ctx, a, nullptr, 1.0f, 0.0f, true);
}

ggml_tensor* ggml_abs(
	ggml_context* ctx,
	ggml_tensor* a) {
	return ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

ggml_tensor* ggml_rope(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int n_dims,
	int mode) {
	return ggml_rope_impl(
		ctx, a, b, nullptr, n_dims, nullptr, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
	);
}

ggml_tensor* ggml_rope_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int n_dims,
	int mode) {
	return ggml_rope_impl(
		ctx, a, b, nullptr, n_dims, nullptr, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
	);
}

ggml_tensor* ggml_mul_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b) {
	return ggml_mul_impl(ctx, a, b, true);
}

static ggml_tensor* ggml_diag_mask_inf_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	int n_past,
	bool inplace) {
	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	int32_t params[] = { n_past };
	ggml_set_op_params(*result, params, sizeof(params));

	result->op = GGML_OP_DIAG_MASK_INF;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_diag_mask_inf_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	int n_past) {
	return ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

ggml_tensor* ggml_conv_2d_dw(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s0,
	int s1,
	int p0,
	int p1,
	int d0,
	int d1) {
	ggml_tensor* new_a = ggml_reshape(ctx, a, { a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3] });
	ggml_tensor* im2col = ggml_im2col(ctx, new_a,
		ggml_reshape(ctx, b, { b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3] }),
		s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
	ggml_tensor* new_b = ggml_reshape(ctx, im2col, { im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3] }); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

	new_a = ggml_reshape(ctx, new_a, { (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3], 1 });                       // [OC¡A1, KH, KW] => [1, OC, 1, KH * KW]
	ggml_tensor* result = ggml_mul_mat(ctx, new_a, new_b);
	result = ggml_reshape(ctx, result, { im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3] }); // [N, OC, OH, OW]

	return result;
}

ggml_tensor* ggml_conv_1d_dw(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	int s0,
	int p0,
	int d0) {
	ggml_tensor* new_a = ggml_reshape(ctx, a, { a->ne[0], 1, a->ne[1], a->ne[2] });
	ggml_tensor* new_b = ggml_reshape(ctx, b, { b->ne[0], 1, b->ne[1], b->ne[2] });

	ggml_tensor* im2col = ggml_im2col(ctx, new_a, new_b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F16);

	ggml_tensor* result = ggml_mul_mat(ctx, im2col, a);

	result = ggml_reshape(ctx, result, { b->ne[0], b->ne[1], 1 });

	return result;
}

#define GGML_N_TASKS_MAX (-1)

static ggml_tensor* ggml_map_custom1_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	const ggml_custom1_op_t   fun,
	int n_tasks,
	void* userdata,
	bool inplace) {
	GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

	struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	struct ggml_map_custom1_op_params params = {
		/*.fun      =*/ fun,
		/*.n_tasks  =*/ n_tasks,
		/*.userdata =*/ userdata
	};
	ggml_set_op_params(*result, &params, sizeof(params));

	result->op = GGML_OP_MAP_CUSTOM1;
	result->src.push_back(a);

	return result;
}

ggml_tensor* ggml_map_custom1(
	ggml_context* ctx,
	ggml_tensor* a,
	const ggml_custom1_op_t fun,
	int n_tasks,
	void* userdata) {
	return ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

ggml_tensor* ggml_map_custom1_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	 ggml_custom1_op_t fun,
	int n_tasks,
	void* userdata) {
	return ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

static struct ggml_tensor* ggml_map_custom2_impl(
	struct ggml_context* ctx,
	struct ggml_tensor* a,
	struct ggml_tensor* b,
	const  ggml_custom2_op_t   fun,
	int n_tasks,
	void* userdata,
	bool inplace) {
	GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

	struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	struct ggml_map_custom2_op_params params = {
		/*.fun      =*/ fun,
		/*.n_tasks  =*/ n_tasks,
		/*.userdata =*/ userdata
	};
	ggml_set_op_params(*result, &params, sizeof(params));

	result->op = GGML_OP_MAP_CUSTOM2;
	result->src.push_back(a);
	result->src.push_back(b);

	return result;
}

ggml_tensor* ggml_map_custom2(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	const ggml_custom2_op_t fun,
	int n_tasks,
	void* userdata) {
	return ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

ggml_tensor* ggml_map_custom2_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	const ggml_custom2_op_t fun,
	int n_tasks,
	void* userdata) {
	return ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

static struct ggml_tensor* ggml_map_custom3_impl(
	struct ggml_context* ctx,
	struct ggml_tensor* a,
	struct ggml_tensor* b,
	struct ggml_tensor* c,
	const  ggml_custom3_op_t   fun,
	int n_tasks,
	void* userdata,
	bool inplace) {
	GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

	ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	struct ggml_map_custom3_op_params params = {
		/*.fun      =*/ fun,
		/*.n_tasks  =*/ n_tasks,
		/*.userdata =*/ userdata
	};
	ggml_set_op_params(*result, &params, sizeof(params));

	result->op = GGML_OP_MAP_CUSTOM3;
	result->src.push_back(a);
	result->src.push_back(b);
	result->src.push_back(c);

	return result;
}

ggml_tensor* ggml_map_custom3(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* c,
	const ggml_custom3_op_t fun,
	int n_tasks,
	void* userdata) {
	return ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

ggml_tensor* ggml_map_custom3_inplace(
	ggml_context* ctx,
	ggml_tensor* a,
	ggml_tensor* b,
	ggml_tensor* c,
	const ggml_custom3_op_t fun,
	int n_tasks,
	void* userdata) {
	return ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

ggml_tensor* ggml_custom_4d(
	ggml_context* ctx,
	enum ggml_type type,
	int64_t               ne0,
	int64_t               ne1,
	int64_t               ne2,
	int64_t               ne3,
	ggml_tensor** args,
	int n_args,
	ggml_custom_op_t fun,
	int n_tasks,
	void* userdata) {

	GGML_ASSERT(n_args < GGML_MAX_SRC);

	ggml_tensor* result = ctx->create(type, { ne0, ne1, ne2, ne3 });

	struct ggml_custom_op_params params = {
		/*.fun      =*/ fun,
		/*.n_tasks  =*/ n_tasks,
		/*.userdata =*/ userdata
	};
	ggml_set_op_params(*result, &params, sizeof(params));

	result->op = GGML_OP_CUSTOM;
	for (int i = 0; i < n_args; i++) {
		result->src.push_back(args[i]);
	}

	return result;
}

ggml_tensor* ggml_rope_ext_inplace(
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
	float beta_slow)
{
	return ggml_rope_impl(
		ctx, a, b, c, n_dims, nullptr, mode, n_ctx_orig, freq_base, freq_scale,
		ext_factor, attn_factor, beta_fast, beta_slow, true
	);
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