export module ggml:op_back;
import :ds;

export {
    ggml_tensor* ggml_rms_norm_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b,
        float eps);

    ggml_tensor* ggml_silu_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b);

    ggml_tensor* ggml_soft_max_ext_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b,
        float scale,
        float max_bias);

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
        float beta_slow);

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
        float beta_slow);

    ggml_tensor* ggml_cross_entropy_loss_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b,
        ggml_tensor* c);

    ggml_tensor* ggml_repeat_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b);
}