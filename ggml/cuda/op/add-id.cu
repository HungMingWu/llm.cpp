#include <algorithm>
#include "cuda_func.h"

static __global__ void add_id_kernel(
    const float* src0, const float* src1, const int32_t* src2, float* dst,
    int64_t ne0, int64_t ne1,
    size_t nb01, size_t nb02,
    size_t nb11,
    size_t nb21
) {

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;

    const int i11 = *(int32_t*)((char*)src2 + i1 * sizeof(int32_t) + i2 * nb21);

    const size_t nb1 = ne0 * sizeof(float);
    const size_t nb2 = ne1 * nb1;

    float* dst_row = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
    const float* src0_row = (const float*)((char*)src0 + i1 * nb01 + i2 * nb02);
    const float* src1_row = (const float*)((char*)src1 + i11 * nb11);

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

void add_id_cuda(const add_id_context* ctx, cudaStream_t stream)
{
    int threads = std::min((int)ctx->ne00, 768); // cols
    dim3 blocks(ctx->ne01, ctx->ne02); // n_experts_used, n_tokens
    add_id_kernel << <blocks, threads, 0, stream >> > (
        ctx->src0_d, ctx->src1_d, ctx->src2_d, ctx->dst_d,
        ctx->ne0, ctx->ne1,
        ctx->nb01, ctx->nb02,
        ctx->nb11,
        ctx->nb21
        );
}