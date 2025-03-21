module;
#include <format>
#include <assert.h>
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
		result->set_name(std::format("{} (copy of {})", b->name, a->name));
	}
	else {
		result->set_name(std::format("{} (copy)", a->name));
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

ggml_tensor* ggml_soft_max_ext(
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

static ggml_tensor* ggml_rope_impl(
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
	GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

	GGML_ASSERT(ggml_is_vector(*b));
	GGML_ASSERT(b->type == GGML_TYPE_I32);
	GGML_ASSERT(a->ne[2] == b->ne[0]);

	if (c) {
		GGML_ASSERT(c->type == GGML_TYPE_F32);
		GGML_ASSERT(c->ne[0] >= n_dims / 2);
	}

	int sections[4] = { 0, 0, 0, 0 };

	struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

	int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
	memcpy(params + 5, &freq_base, sizeof(float));
	memcpy(params + 6, &freq_scale, sizeof(float));
	memcpy(params + 7, &ext_factor, sizeof(float));
	memcpy(params + 8, &attn_factor, sizeof(float));
	memcpy(params + 9, &beta_fast, sizeof(float));
	memcpy(params + 10, &beta_slow, sizeof(float));
	memcpy(params + 11, &sections, sizeof(int) * 4);
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
		ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
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