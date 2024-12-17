#include <algorithm>
#include <numbers>
#include "cuda_func.h"
#include "common.cuh"
#define GGML_ABORT(...)

static constexpr size_t CUDA_ROPE_BLOCK_SIZE = 256;

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
    const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
    float mscale, float& cos_theta, float& sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_neox(
    const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
    const int32_t* pos, const float freq_scale, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float theta_scale, const float* freq_factors) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix = channel_x * s2 + row_x * s1 + i0 / 2;

    const float theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template<bool forward, typename T>
static void rope_neox_cuda(
    const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr,
    const int32_t* pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float* freq_factors, cudaStream_t stream) {
    //GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false, T> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
    else {
        rope_neox<forward, true, T> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
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

template<bool forward, bool has_ff, typename T>
static __global__ void rope_multi(
    const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2,
    const int n_dims, const int32_t* pos, const float freq_scale, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float theta_scale, const float* freq_factors, const mrope_sections sections) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix = channel_x * s2 + row_x * s1 + i0 / 2;

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        theta_base = pos[channel_x + ne2 * 1] * powf(theta_scale, i0 / 2.0f);
    }
    else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 2] * powf(theta_scale, i0 / 2.0f);
    }
    else if (sector >= sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 3] * powf(theta_scale, i0 / 2.0f);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template<bool forward, typename T>
static void rope_multi_cuda(
    const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
    const int32_t* pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float* freq_factors, const mrope_sections sections, cudaStream_t stream) {
    //GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_multi<forward, false, T> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
    else {
        rope_multi<forward, true, T> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_vision(
    const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims,
    const int32_t* pos, const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
    const float theta_scale, const float* freq_factors, const mrope_sections sections) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    const int row_x = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix = channel_x * s2 + row_x * s1 + i0 / 2;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base = pos[channel_x] * powf(theta_scale, p);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base = pos[channel_x + ne2] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template<bool forward, typename T>
static void rope_vision_cuda(
    const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
    const int32_t* pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float* freq_factors, const mrope_sections sections, cudaStream_t stream) {
    //GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
    else {
        rope_vision<forward, true, T> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_norm(
    const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
    const int32_t* pos, const float freq_scale, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float theta_scale, const float* freq_factors) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0;
    const int ix = channel_x * s2 + row_x * s1 + i0;

    const float theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + 1] = x0 * sin_theta + x1 * cos_theta;
}

template<bool forward, typename T>
static void rope_norm_cuda(
    const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr,
    const int32_t* pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float* freq_factors, cudaStream_t stream) {
    //GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
    else {
        rope_norm<forward, true> << <block_nums, block_dims, 0, stream >> > (
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
}

template <bool forward>
void rope_cuda(const rope_context* ctx, cudaStream_t stream)
{
    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(ctx->n_dims, ctx->n_ctx_orig,
        ctx->freq_base, ctx->beta_fast, ctx->beta_slow, corr_dims.v);

    if (ctx->is_neox) {
        if (ctx->src0_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward>(
                (const float*)ctx->src0_d, (float*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base,
                ctx->ext_factor, ctx->attn_factor, corr_dims, ctx->freq_factors, stream);
        }
        else if (ctx->src0_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward>(
                (const half*)ctx->src0_d, (half*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base,
                ctx->ext_factor, ctx->attn_factor, corr_dims, ctx->freq_factors, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    } else if (ctx->is_mrope && !ctx->is_vision) {
        if (ctx->src0_type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>(
                (const float*)ctx->src0_d, (float*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->ne02, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base,
                ctx->ext_factor, ctx->attn_factor, corr_dims, ctx->freq_factors, ctx->sections, stream);
        }
        else if (ctx->src0_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>(
                (const half*)ctx->src0_d, (half*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->ne02, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base,
                ctx->ext_factor, ctx->attn_factor, corr_dims, ctx->freq_factors, ctx->sections, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    } else if (ctx->is_vision) {
        if (ctx->src0_type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>(
                (const float*)ctx->src0_d, (float*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->ne02, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base,
                ctx->ext_factor, ctx->attn_factor, corr_dims, ctx->freq_factors, ctx->sections, stream);
        }
        else if (ctx->src0_type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>(
                (const half*)ctx->src0_d, (half*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->ne02, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base,
                ctx->ext_factor, ctx->attn_factor, corr_dims, ctx->freq_factors, ctx->sections, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (ctx->src0_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward>(
                (const float*)ctx->src0_d, (float*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base, ctx->ext_factor,
                ctx->attn_factor, corr_dims, ctx->freq_factors, stream);
        }
        else if (ctx->src0_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward>(
                (const half*)ctx->src0_d, (half*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->s01, ctx->s02, ctx->n_dims, ctx->nr,
                ctx->pos, ctx->freq_scale, ctx->freq_base, ctx->ext_factor,
                ctx->attn_factor, corr_dims, ctx->freq_factors, stream);
        }
        else {
            GGML_ABORT("fatal error");
        }
    }
}

void rope_cuda(const rope_context* ctx, cudaStream_t stream)
{
	if (ctx->forward) {
        rope_cuda<true>(ctx, stream);
	}
	else {
        rope_cuda<false>(ctx, stream);
	}
}