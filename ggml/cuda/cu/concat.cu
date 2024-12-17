#include "common.cuh"
#include "cuda_func.h"

static constexpr size_t CUDA_CONCAT_BLOCK_SIZE = 256;

// contiguous kernels
static __global__ void concat_f32_dim0(const float* x, const float* y, float* dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    }
    else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float* x, const float* y, float* dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    }
    else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float* x, const float* y, float* dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    }
    else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float* x, const float* y, float* dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0 << <gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream >> > (x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1 << <gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream >> > (x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2 << <gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream >> > (x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
concat_f32_non_cont(
    const char* src0,
    const char* src1,
    char* dst,
    int64_t   ne00,
    int64_t   ne01,
    int64_t   ne02,
    int64_t   ne03,
    uint64_t   nb00,
    uint64_t   nb01,
    uint64_t   nb02,
    uint64_t   nb03,
    int64_t /*ne10*/,
    int64_t /*ne11*/,
    int64_t /*ne12*/,
    int64_t /*ne13*/,
    uint64_t   nb10,
    uint64_t   nb11,
    uint64_t   nb12,
    uint64_t   nb13,
    int64_t   ne0,
    int64_t /*ne1*/,
    int64_t /*ne2*/,
    int64_t /*ne3*/,
    uint64_t   nb0,
    uint64_t   nb1,
    uint64_t   nb2,
    uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float* x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float*)(src0 + (i3)*nb03 + (i2)*nb02 + (i1)*nb01 + (i0)*nb00);
        }
        else {
            if constexpr (dim == 0) {
                x = (const float*)(src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            }
            else if constexpr (dim == 1) {
                x = (const float*)(src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            }
            else if constexpr (dim == 2) {
                x = (const float*)(src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            }
            else if constexpr (dim == 3) {
                x = (const float*)(src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float* y = (float*)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

        *y = *x;
    }
}

void concat_cuda(const concat_context* ctx, cudaStream_t stream)
{
    if (ctx->src0_is_contiguous && ctx->src1_is_contiguous) {
        if (ctx->dim != 3) {
            for (int i3 = 0; i3 < ctx->ne3; i3++) {
                concat_f32_cuda(
                    ctx->src0_d + i3 * (ctx->nb03 / 4),
                    ctx->src1_d + i3 * (ctx->nb13 / 4),
                    ctx->dst_d + i3 * (ctx->nb3 / 4),
                    ctx->ne00, ctx->ne01, ctx->ne02,
                    ctx->ne0, ctx->ne1, ctx->ne2, ctx->dim, stream);
            }
        }
        else {
            CUDA_CHECK(cudaMemcpyAsync(ctx->dst_d, ctx->src0_d, ctx->src0_size, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(ctx->dst_d + ctx->src0_size / 4, ctx->src1_d, ctx->src1_size, cudaMemcpyDeviceToDevice, stream));
        }
    }
    else {
        dim3 grid_dim(ctx->ne1, ctx->ne2, ctx->ne3);

        auto launch_kernel = [&](auto dim) {
            concat_f32_non_cont<dim> << <grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream >> > (
                (const char*)ctx->src0_d, (const char*)ctx->src1_d, (char*)ctx->dst_d,
                ctx->ne00, ctx->ne01, ctx->ne02, ctx->ne03,
                ctx->nb00, ctx->nb01, ctx->nb02, ctx->nb03,
                ctx->ne10, ctx->ne11, ctx->ne12, ctx->ne13,
                ctx->nb10, ctx->nb11, ctx->nb12, ctx->nb13,
                ctx->ne0, ctx->ne1, ctx->ne2, ctx->ne3,
                ctx->nb0, ctx->nb1, ctx->nb2, ctx->nb3);
        };
        switch (ctx->dim) {
        case 0:
            launch_kernel(std::integral_constant<int, 0>{});
            break;
        case 1:
            launch_kernel(std::integral_constant<int, 1>{});
            break;
        case 2:
            launch_kernel(std::integral_constant<int, 2>{});
            break;
        case 3:
            launch_kernel(std::integral_constant<int, 3>{});
            break;
        default:
            GGML_ABORT("Invalid dim: %d", dim);
            break;
        }
    }
}