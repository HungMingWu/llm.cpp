#include "convert.cuh"
#include "cuda_func.h"

template <typename T>
static __global__ void diag_kernel(T* __restrict__ dst,
    const T* __restrict__ src,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    [[maybe_unused]] const int64_t ne3,
    const int64_t total_elements) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= total_elements) {
        return;
    }

    const int64_t i0 = global_idx % ne0;
    const int64_t i1 = (global_idx / ne0) % ne1;
    const int64_t i2 = (global_idx / (ne0 * ne1)) % ne2;
    const int64_t i3 = global_idx / (ne0 * ne1 * ne2);

    const int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    if (i0 == i1) {
        const int64_t batch_idx = i3 * ne2 + i2;
        const int64_t src_idx = batch_idx * ne0 + i0;
        dst[dst_idx] = src[src_idx];
    }
    else {
        dst[dst_idx] = ggml_cuda_cast<T>(0);
    }
}

void diag_cuda(const diag_context& ctx, cudaStream_t stream)
{
    static constexpr size_t CUDA_DIAG_BLOCK_SIZE = 256;
    const int64_t num_blocks = (ctx.n_elems + CUDA_DIAG_BLOCK_SIZE - 1) / CUDA_DIAG_BLOCK_SIZE;
    switch (ctx.dst_type) {
    case internal::GGML_TYPE_F32:
        diag_kernel << <num_blocks, CUDA_DIAG_BLOCK_SIZE, 0, stream >> > ((float*)ctx.dst_d, (const float*)ctx.src0_d, ctx.ne0,
            ctx.ne1, ctx.ne2, ctx.ne3, ctx.n_elems);
        break;
    case internal::GGML_TYPE_F16:
        diag_kernel << <num_blocks, CUDA_DIAG_BLOCK_SIZE, 0, stream >> > ((half*)ctx.dst_d, (const half*)ctx.src0_d, ctx.ne0,
            ctx.ne1, ctx.ne2, ctx.ne3, ctx.n_elems);
        break;
    default:
        GGML_ABORT("unsupported type");
    }
}
