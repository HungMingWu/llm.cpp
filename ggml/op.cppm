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

	ggml_tensor* ggml_mul_mat_id(
		ggml_context* ctx,
		ggml_tensor* as,
		ggml_tensor* b,
		ggml_tensor* ids);

	ggml_tensor* ggml_out_prod(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_sqr(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_sqrt(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_log(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_sin(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_cos(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_clamp(
		ggml_context* ctx,
		ggml_tensor* a,
		float min,
		float max);

	ggml_tensor* ggml_diag_mask_inf(
		ggml_context* ctx,
		ggml_tensor* a,
		int n_past);

	ggml_tensor* ggml_soft_max_ext(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* mask,
		float scale,
		float max_bias);

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
		float beta_slow);

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
		float beta_slow);

	ggml_tensor* ggml_concat(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b,
		int dim);

	ggml_tensor* ggml_argsort(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_sort_order order);

	ggml_tensor* ggml_sum(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_sum_rows(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_mean(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_upscale(
		ggml_context* ctx,
		ggml_tensor* a,
		int scale_factor,
		ggml_scale_mode mode);

	ggml_tensor* ggml_upscale_ext(
		ggml_context* ctx,
		ggml_tensor* a,
		int ne0,
		int ne1,
		int ne2,
		int ne3,
		ggml_scale_mode mode);

	ggml_tensor* ggml_group_norm(
		ggml_context* ctx,
		ggml_tensor* a,
		int n_groups,
		float eps);

	ggml_tensor* ggml_acc(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b,
		size_t nb1,
		size_t nb2,
		size_t nb3,
		size_t offset);

	ggml_tensor* ggml_pad(
		ggml_context* ctx,
		ggml_tensor* a,
		int p0,
		int p1,
		int p2,
		int p3);

	ggml_tensor* ggml_timestep_embedding(
		ggml_context* ctx,
		ggml_tensor* timesteps,
		int dim,
		int max_period);

	ggml_tensor* ggml_leaky_relu(
		ggml_context* ctx,
		ggml_tensor* a,
		float negative_slope,
		bool inplace);

	ggml_tensor* ggml_flash_attn_ext(
		ggml_context* ctx,
		ggml_tensor* q,
		ggml_tensor* k,
		ggml_tensor* v,
		ggml_tensor* mask,
		float scale,
		float max_bias,
		float logit_softcap);

	ggml_tensor* ggml_mul_mat(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_conv_1d(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b,
		int s0,
		int p0,
		int d0);

	ggml_tensor* ggml_conv_2d(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b,
		int s0,
		int s1,
		int p0,
		int p1,
		int d0,
		int d1);

	ggml_tensor* ggml_cross_entropy_loss(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_soft_max(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_opt_step_adamw(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* grad,
		ggml_tensor* m,
		ggml_tensor* v,
		ggml_tensor* adamw_params);

	ggml_tensor* ggml_reshape_2d(
		ggml_context* ctx,
		ggml_tensor* a,
		int64_t ne0,
		int64_t ne1);

	ggml_tensor* ggml_reshape_3d(
		ggml_context* ctx,
		ggml_tensor* a,
		int64_t ne0,
		int64_t ne1,
		int64_t ne2);

	ggml_tensor* ggml_cont_2d(
		ggml_context* ctx,
		ggml_tensor* a,
		int64_t ne0,
		int64_t ne1);

	ggml_tensor* ggml_cont_4d(
		ggml_context* ctx,
		ggml_tensor* a,
		int64_t ne0,
		int64_t ne1,
		int64_t ne2,
		int64_t ne3);

	ggml_tensor* ggml_silu(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_gelu(
		ggml_context* ctx,
		ggml_tensor* a);

	ggml_tensor* ggml_l2_norm(
		ggml_context* ctx,
		ggml_tensor* a,
		float eps);

	ggml_tensor* ggml_rwkv_wkv7(
		ggml_context* ctx,
		ggml_tensor* r,
		ggml_tensor* w,
		ggml_tensor* k,
		ggml_tensor* v,
		ggml_tensor* a,
		ggml_tensor* b,
		ggml_tensor* state);

	ggml_tensor* ggml_sub(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b);

	ggml_tensor* ggml_conv_2d_dw_direct(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b,
		int stride0,
		int stride1,
		int pad0,
		int pad1,
		int dilation0,
		int dilation1);
}
