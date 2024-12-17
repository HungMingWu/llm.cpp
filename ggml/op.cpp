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