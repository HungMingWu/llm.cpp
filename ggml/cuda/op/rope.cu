#include <algorithm>
#include <numbers>
#include "cuda_func.h"
#include "common.cuh"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "launch.cuh"

#define GGML_ABORT(...)

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template <bool forward>
static __device__ std::pair<float, float> rope_yarn(
    const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
    float mscale) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float cos_theta = cosf(theta) * mscale;
    float sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
    return { cos_theta, sin_theta };
}

template <bool forward, typename src_t, typename dst_t>
void rope_neox_cuda(const rope_context& ctx, const rope_corr_dims corr_dims, cudaStream_t stream) {
    assert(ctx.src_ne[0] % 2 == 0);
    const float theta_scale = powf(ctx.freq_base, -2.0f / ctx.n_dims);

    auto src_data = make_strided_mdspan<3>(static_cast<const src_t*>(ctx.src_d), ctx.src_ne, ctx.src_nb);
    auto dst_data = make_strided_mdspan<3>(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src_ne[2], ctx.src_ne[1], ctx.src_ne[0] / 2),
        [=] __device__(int64_t channel_x, int64_t row_x, int64_t i0) {
            i0 *= 2;

            auto get_dst_channel = [&](int64_t channel_x) {
                if (ctx.row_indices != nullptr)
                    return ctx.row_indices[channel_x];
                return channel_x;
            };

            if (i0 >= ctx.n_dims) {
                dst_data(get_dst_channel(channel_x), row_x, i0) = ggml_cuda_cast<dst_t>(src_data(channel_x, row_x, i0 + 0));
                dst_data(get_dst_channel(channel_x), row_x, i0 + 1) = ggml_cuda_cast<dst_t>(src_data(channel_x, row_x, i0 + 1));
                return;
            }

            const float theta_base = ctx.pos[channel_x] * powf(theta_scale, i0 / 2.0f);
            const float freq_factor = (ctx.freq_factors != nullptr) ? ctx.freq_factors[i0 / 2] : 1.0f;
            const auto [cos_theta, sin_theta] = 
                rope_yarn<forward>(theta_base / freq_factor, ctx.freq_scale, corr_dims, i0, ctx.ext_factor, ctx.attn_factor);

            const float x0 = src_data(channel_x, row_x, i0 / 2);
            const float x1 = src_data(channel_x, row_x, i0 / 2 + ctx.n_dims / 2);

            dst_data(get_dst_channel(channel_x), row_x, i0 / 2) = ggml_cuda_cast<dst_t>(x0 * cos_theta - x1 * sin_theta);
            dst_data(get_dst_channel(channel_x), row_x, i0 / 2 + ctx.n_dims / 2) = ggml_cuda_cast<dst_t>(x0 * sin_theta + x1 * cos_theta);
        }
    );
}

static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * std::numbers::pi_v<float>)) / (2 * logf(base));
}

static void ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max<float>(0, start);
    dims[1] = std::min<float>(n_dims - 1, end);
}

template<bool forward, typename T>
void rope_multi_cuda(const rope_context& ctx, const rope_corr_dims corr_dims, cudaStream_t stream) {
    assert(ctx.src_ne[0] % 2 == 0);

    const float theta_scale = powf(ctx.freq_base, -2.0f / ctx.n_dims);

    auto src_data = make_strided_mdspan<3>(static_cast<const T*>(ctx.src_d), ctx.src_ne, ctx.src_nb);
    auto dst_data = make_strided_mdspan<3>(static_cast<T*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src_ne[2], ctx.src_ne[1], ctx.src_ne[0] / 2),
        [=] __device__(int64_t channel_x, int64_t row_x, int64_t i0) {
           i0 *= 2;

           if (i0 >= ctx.n_dims) {
               dst_data(channel_x, row_x, i0 + 0) = src_data(channel_x, row_x, i0 + 0);
               dst_data(channel_x, row_x, i0 + 1) = src_data(channel_x, row_x, i0 + 1);

               return;
           }

           const int sect_dims = ctx.sections.v[0] + ctx.sections.v[1] + ctx.sections.v[2] + ctx.sections.v[3];
           const int sec_w = ctx.sections.v[1] + ctx.sections.v[0];
           const int sector = (i0 / 2) % sect_dims;

           float theta_base = 0.0;
           if (ctx.is_imrope) {
               if (sector % 3 == 1 && sector < 3 * ctx.sections.v[1]) { // h
                   theta_base = ctx.pos[channel_x + ctx.src_ne[2] * 1] * powf(theta_scale, i0 / 2.0f);
               }
               else if (sector % 3 == 2 && sector < 3 * ctx.sections.v[2]) { // w
                   theta_base = ctx.pos[channel_x + ctx.src_ne[2] * 2] * powf(theta_scale, i0 / 2.0f);
               }
               else if (sector % 3 == 0 && sector < 3 * ctx.sections.v[0]) { // t
                   theta_base = ctx.pos[channel_x] * powf(theta_scale, i0 / 2.0f);
               }
               else {
                   theta_base = ctx.pos[channel_x + ctx.src_ne[2] * 3] * powf(theta_scale, i0 / 2.0f);
               }
           }
           else {
               if (sector < ctx.sections.v[0]) {
                   theta_base = ctx.pos[channel_x] * powf(theta_scale, i0 / 2.0f);
               }
               else if (sector >= ctx.sections.v[0] && sector < sec_w) {
                   theta_base = ctx.pos[channel_x + ctx.src_ne[2] * 1] * powf(theta_scale, i0 / 2.0f);
               }
               else if (sector >= sec_w && sector < sec_w + ctx.sections.v[2]) {
                   theta_base = ctx.pos[channel_x + ctx.src_ne[2] * 2] * powf(theta_scale, i0 / 2.0f);
               }
               else if (sector >= sec_w + ctx.sections.v[2]) {
                   theta_base = ctx.pos[channel_x + ctx.src_ne[2] * 3] * powf(theta_scale, i0 / 2.0f);
               }
           }

           const float freq_factor = (ctx.freq_factors != nullptr) ? ctx.freq_factors[i0 / 2] : 1.0f;

           const auto [cos_theta, sin_theta] = 
               rope_yarn<forward>(theta_base / freq_factor, ctx.freq_scale, corr_dims, i0, ctx.ext_factor, ctx.attn_factor);

           const float x0 = src_data(channel_x, row_x, i0 / 2);
           const float x1 = src_data(channel_x, row_x, i0 / 2 + ctx.n_dims / 2);

           dst_data(channel_x, row_x, i0 / 2) = x0 * cos_theta - x1 * sin_theta;
           dst_data(channel_x, row_x, i0 / 2 + ctx.n_dims / 2) = x0 * sin_theta + x1 * cos_theta;
        }
    );
}

template<bool forward, typename T>
void rope_vision_cuda(const rope_context& ctx, const rope_corr_dims corr_dims, cudaStream_t stream) {
    assert(ctx.src_ne[0] % 2 == 0);
    const float theta_scale = powf(ctx.freq_base, -2.0f / ctx.n_dims);

    auto src_data = make_strided_mdspan<3>(static_cast<const T*>(ctx.src_d), ctx.src_ne, ctx.src_nb);
    auto dst_data = make_strided_mdspan<3>(static_cast<T*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src_ne[2], ctx.src_ne[1], ctx.src_ne[0] / 2),
        [=] __device__(int64_t channel_x, int64_t row_x, int64_t i0) {
            i0 *= 2;

            const int sect_dims = ctx.sections.v[0] + ctx.sections.v[1];
            const int sec_w = ctx.sections.v[1] + ctx.sections.v[0];
            const int sector = (i0 / 2) % sect_dims;

            float theta_base = 0.0;
            if (sector < ctx.sections.v[0]) {
                const int p = sector;
                theta_base = ctx.pos[channel_x] * powf(theta_scale, p);
            }
            else if (sector >= ctx.sections.v[0] && sector < sec_w) {
                const int p = sector - ctx.sections.v[0];
                theta_base = ctx.pos[channel_x + ctx.src_ne[2]] * powf(theta_scale, p);
            }

            const float freq_factor = (ctx.freq_factors != nullptr) ? ctx.freq_factors[i0 / 2] : 1.0f;

            const auto [cos_theta, sin_theta] = 
                rope_yarn<forward>(theta_base / freq_factor, ctx.freq_scale, corr_dims, i0, ctx.ext_factor, ctx.attn_factor);

            const float x0 = src_data(channel_x, row_x, i0 / 2);
            const float x1 = src_data(channel_x, row_x, i0 / 2 + ctx.n_dims);

            dst_data(channel_x, row_x, i0 / 2) = x0 * cos_theta - x1 * sin_theta;
            dst_data(channel_x, row_x, i0 / 2 + ctx.n_dims) = x0 * sin_theta + x1 * cos_theta;
        }
    );
}

template <bool forward, typename T, typename D>
void rope_norm_cuda(const rope_context& ctx, const rope_corr_dims corr_dims, cudaStream_t stream) {
    assert(ctx.src_ne[0] % 2 == 0);
    const float theta_scale = powf(ctx.freq_base, -2.0f / ctx.n_dims);
    auto src_data = make_strided_mdspan<3>(static_cast<const T*>(ctx.src_d), ctx.src_ne, ctx.src_nb);
    auto dst_data = make_strided_mdspan<3>(static_cast<D*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    launch_functor(stream, std::make_tuple(ctx.src_ne[2], ctx.src_ne[1], ctx.src_ne[0] / 2),
        [=] __device__(int64_t channel_x, int64_t row_x, int64_t i0) {
            i0 *= 2;

            auto get_dst_channel = [&](int64_t channel_x) {
                if (ctx.row_indices != nullptr)
                    return ctx.row_indices[channel_x];
                return channel_x;
            };

            if (i0 >= ctx.n_dims) {
                dst_data(get_dst_channel(channel_x), row_x, i0) = ggml_cuda_cast<D>(src_data(channel_x, row_x, i0 + 0));
                dst_data(get_dst_channel(channel_x), row_x, i0 + 1) = ggml_cuda_cast<D>(src_data(channel_x, row_x, i0 + 1));
                return;
            }

            const float theta_base = ctx.pos[channel_x] * powf(theta_scale, i0 / 2.0f);

            const float freq_factor = (ctx.freq_factors != nullptr) ? ctx.freq_factors[i0 / 2] : 1.0f;

            const auto [cos_theta, sin_theta] = 
                rope_yarn<forward>(theta_base / freq_factor, ctx.freq_scale, corr_dims, i0, ctx.ext_factor, ctx.attn_factor);

            const float x0 = src_data(channel_x, row_x, i0 + 0);
            const float x1 = src_data(channel_x, row_x, i0 + 1);

            dst_data(get_dst_channel(channel_x), row_x, i0) = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
            dst_data(get_dst_channel(channel_x), row_x, i0 + 1) = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
        }
    );
}

template <bool forward>
void rope_cuda(const rope_context& ctx, cudaStream_t stream)
{
    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(ctx.n_dims, ctx.n_ctx_orig,
        ctx.freq_base, ctx.beta_fast, ctx.beta_slow, corr_dims.v);

    if (ctx.is_neox) {
        if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>(ctx, corr_dims, stream);
        }
        else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>(ctx, corr_dims, stream);
        }
        else if (ctx.src_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>(ctx, corr_dims, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    } else if (ctx.is_mrope && !ctx.is_vision) {
        if (ctx.src_type == internal::GGML_TYPE_F32) {
            rope_multi_cuda<forward, float>(ctx, corr_dims, stream);
        }
        else if (ctx.src_type == internal::GGML_TYPE_F16) {
            rope_multi_cuda<forward, half>(ctx, corr_dims, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    } else if (ctx.is_vision) {
        if (ctx.src_type == internal::GGML_TYPE_F32) {
            rope_vision_cuda<forward, float>(ctx, corr_dims, stream);
        }
        else if (ctx.src_type == internal::GGML_TYPE_F16) {
            rope_vision_cuda<forward, half>(ctx, corr_dims, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>(ctx, corr_dims, stream);
        }
        else if (ctx.src_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>(ctx, corr_dims, stream);
        }
        else if (ctx.src_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>(ctx, corr_dims, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    }
}

void rope_cuda(const rope_context& ctx, cudaStream_t stream)
{
	if (ctx.forward) {
        rope_cuda<true>(ctx, stream);
	}
	else {
        rope_cuda<false>(ctx, stream);
	}
}