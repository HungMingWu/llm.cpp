#include "common.cuh"
#include "cuda_func.h"

static constexpr int DSV4_HC = 4;

static __device__ void dsv4_hc_comb_norm_cols(float* comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC * isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC * isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float* comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC * isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC * isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
    const float* mixes,
    const float* scale,
    const float* base,
    float* dst,
    int64_t n_tokens,
    int64_t sm0,
    int64_t sm1,
    int64_t ss0,
    int64_t sb0,
    int64_t sd0,
    int64_t sd1,
    int64_t sd2,
    float eps,
    int32_t n_iter) {
    constexpr int comb_offset = 2 * DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2 * ss0];
    float comb[DSV4_HC * DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            const float v = mixes[(comb_offset + idx) * sm0 + it * sm1] * scale_comb + base[(comb_offset + idx) * sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            dst[idst * sd0 + isrc * sd1 + it * sd2] = comb[idx];
        }
    }
}

static __global__ void dsv4_hc_pre_f32(
    const float* x,
    const float* weights,
    float* dst,
    int64_t n_embd,
    int64_t hc,
    int64_t n_tokens,
    int64_t sx0,
    int64_t sx1,
    int64_t sx2,
    int64_t sw0,
    int64_t sw1,
    int64_t sd0,
    int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0 * sx0 + it * sx2] * weights[it * sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0 * sx0 + ih * sx1 + it * sx2];
        const float wv = weights[ih * sw0 + it * sw1];
        sum += xv * wv;
    }

    dst[i0 * sd0 + it * sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
    const float* x,
    const float* residual,
    const float* post,
    const float* comb,
    float* dst,
    int64_t n_embd,
    int64_t hc,
    int64_t n_tokens,
    int64_t sx0,
    int64_t sx1,
    int64_t sr0,
    int64_t sr1,
    int64_t sr2,
    int64_t sp0,
    int64_t sp1,
    int64_t sc0,
    int64_t sc1,
    int64_t sc2,
    int64_t sd0,
    int64_t sd1,
    int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it = ir / (n_embd * hc);

    float sum = x[i0 * sx0 + it * sx1] * post[idst * sp0 + it * sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0 * sr0 + isrc * sr1 + it * sr2] * comb[idst * sc0 + isrc * sc1 + it * sc2];
    }

    dst[i0 * sd0 + idst * sd1 + it * sd2] = sum;
}

void dsv4_hc_comb_cuda(const dsv4_hc_comb_context& ctx, cudaStream_t stream) {
    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((ctx.n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
        ctx.mixes_data, ctx.scale_data, ctx.base_data, ctx.dst_data,
        ctx.n_tokens,
        ctx.nbm0 / sizeof(float), ctx.nbm1 / sizeof(float),
        ctx.nbs0 / sizeof(float),
        ctx.nbb0 / sizeof(float),
        ctx.nbd0 / sizeof(float), ctx.nbd1 / sizeof(float), ctx.nbd2 / sizeof(float),
        ctx.eps, ctx.n_iter);
}

void dsv4_hc_pre_cuda(const dsv4_hc_pre_context& ctx, cudaStream_t stream) {
    const int block_size = 256;
    const int64_t nr = ctx.n_embd * ctx.n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
        ctx.x_data, ctx.weights_data, ctx.dst_data,
        ctx.n_embd, ctx.hc, ctx.n_tokens,
        ctx.nbx0 / sizeof(float), ctx.nbx1 / sizeof(float), ctx.nbx2 / sizeof(float),
        ctx.nbw0 / sizeof(float), ctx.nbw1 / sizeof(float),
        ctx.nbd0 / sizeof(float), ctx.nbd1 / sizeof(float));
}

void dsv4_hc_post_cuda(const dsv4_hc_post_context& ctx, cudaStream_t stream) {
    const int block_size = 256;
    const int64_t nr = ctx.n_embd * ctx.hc * ctx.n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
        ctx.x_data, ctx.residual_data,
        ctx.post_data, ctx.comb_data, ctx.dst_data,
        ctx.n_embd, ctx.hc, ctx.n_tokens,
        ctx.nbx0 / sizeof(float), ctx.nbx1 / sizeof(float),
        ctx.nbr0 / sizeof(float), ctx.nbr1 / sizeof(float), ctx.nbr2 / sizeof(float),
        ctx.nbp0 / sizeof(float), ctx.nbp1 / sizeof(float),
        ctx.nbc0 / sizeof(float), ctx.nbc1 / sizeof(float), ctx.nbc2 / sizeof(float),
        ctx.nbd0 / sizeof(float), ctx.nbd1 / sizeof(float), ctx.nbd2 / sizeof(float));
}
