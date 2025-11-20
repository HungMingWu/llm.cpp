#include "cuda_func.h"
#include "helper.h"
#include "launch.cuh"
#include <span>

void timestep_embedding_f32_cuda(const timestep_embeddin_ctx& ctx, cudaStream_t stream)
{
    int half_ceil = (ctx.dim + 1) / 2;
    std::span<const float> timesteps(ctx.src0_d, ctx.ne00);
    launch_functor(stream, std::make_tuple(ctx.ne00, half_ceil),
        [=] __device__(int64_t i, int64_t j) {
            float* embed_data = (float*)((char*)ctx.dst_d + i * ctx.nb1);

            int half_dim = ctx.dim / 2;
            if (ctx.dim % 2 != 0 && j == half_dim) {
                embed_data[2 * half_dim] = 0.f;
            }

            if (j >= half_dim) {
                return;
            }

            float timestep = timesteps[i];
            float freq = (float)expf(-logf(ctx.max_period) * j / half_dim);
            float arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half_dim] = sinf(arg);
        }
    );
}