#include <stdint.h>

static __global__ void opt_step_adamw_f32(
    float* __restrict__ x, const float* __restrict__ g, float* __restrict__ g_m, float* __restrict__ g_v,
    const float* __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float alpha = pars[0];
    const float beta1 = pars[1];
    const float beta2 = pars[2];
    const float eps = pars[3];
    const float wd = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[i];
    const float gmi = g_m[i] * beta1 + gi * (1.0f - beta1);
    const float gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    const float mh = gmi * beta1h;
    const float vh = sqrtf(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

void opt_step_adamw_f32_cuda(
    float* x, const float* g, float* g_m,
    float* g_v, const float* pars, const int64_t k, cudaStream_t stream)
{
    static constexpr size_t CUDA_OPT_STEP_ADAMW_BLOCK_SIZE = 256;
    const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    opt_step_adamw_f32 << <block_nums, block_dims, 0, stream >> > (x, g, g_m, g_v, pars, k);
}