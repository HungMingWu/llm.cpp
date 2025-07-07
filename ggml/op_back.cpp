module;
#include <assert.h>
#include <stdint.h>
#include <bit>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;

ggml_tensor* ggml_rms_norm_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    float eps) {
    ggml_tensor* result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(*result, &eps, sizeof(eps));

    result->op = GGML_OP_RMS_NORM_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

ggml_tensor* ggml_silu_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b) {
    ggml_tensor* result = ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_SILU_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

static ggml_tensor* ggml_soft_max_ext_back_impl(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    float scale,
    float max_bias,
    bool inplace) {
    ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_SOFT_MAX_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    result->op_params[0] = std::bit_cast<uint32_t>(scale);
    result->op_params[1] = std::bit_cast<uint32_t>(max_bias);

    return result;
}

ggml_tensor* ggml_soft_max_ext_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    float scale,
    float max_bias) {
    return ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

ggml_tensor* ggml_rope_multi_back(
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
    ggml_tensor* result = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    return result;
}

ggml_tensor* ggml_rope_ext_back(
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
    ggml_tensor* result = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    result->op = GGML_OP_ROPE_BACK;
    return result;
}

ggml_tensor* ggml_cross_entropy_loss_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    ggml_tensor* c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    ggml_tensor* result = ggml_dup_tensor(ctx, b);

    result->op = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src.push_back(a);
    result->src.push_back(b);
    result->src.push_back(c);

    return result;
}

ggml_tensor* ggml_repeat_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ggml_tensor* result = ctx->create(a->type, { b->ne[0], b->ne[1], b->ne[2], b->ne[3] });

    result->op = GGML_OP_REPEAT_BACK;
    result->src.push_back(a);

    return result;
}

ggml_tensor* ggml_get_rows_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    ggml_tensor* c)
{
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    ggml_tensor* result = ctx->create(GGML_TYPE_F32, { c->ne[0], c->ne[1] });

    result->op = GGML_OP_GET_ROWS_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}