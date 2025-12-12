#include "common.cuh"
#include "convert.cuh"
#include "cuda_func.h"

#define GGML_ABORT(...)

template<typename T, bool prefix_keep, int add_to_split>
static __global__ void tri_kernel(
    const T* src, T* dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    [[maybe_unused]] const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    [[maybe_unused]] const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    const int64_t split_point = i1 + add_to_split;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const T* src_row = src + i1 * nb01 + i2 * nb02 + i3 * nb03;
    T* dst_row = dst + i1 * nb1 + i2 * nb2 + i3 * nb3;

    if constexpr (prefix_keep) {
        for (int64_t i0 = threadIdx.x; i0 < split_point; i0 += blockDim.x) {
            dst_row[i0] = src_row[i0];
        }
        for (int64_t i0 = threadIdx.x + split_point; i0 < ne00; i0 += blockDim.x) {
            dst_row[i0] = ggml_cuda_cast<T, float>(0.0f);
        }
    }
    else {
        for (int64_t i0 = threadIdx.x; i0 < split_point; i0 += blockDim.x) {
            dst_row[i0] = ggml_cuda_cast<T, float>(0.0f);
        }
        for (int64_t i0 = threadIdx.x + split_point; i0 < ne00; i0 += blockDim.x) {
            dst_row[i0] = src_row[i0];
        }
    }
}

template<typename T>
static void tri_cuda(
    const T* src, T* dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3,
    const internal::ggml_tri_type ttype,
    cudaStream_t stream) {

    static constexpr size_t CUDA_TRI_BLOCK_SIZE = 256;
    dim3 block_dims(CUDA_TRI_BLOCK_SIZE, 1, 1);
    dim3 grid_dims(ne01, ne02, ne03);
    const size_t type_size = sizeof(T);

    const int add_to_split = (ttype == internal::GGML_TRI_TYPE_LOWER_DIAG || ttype == internal::GGML_TRI_TYPE_UPPER) ? 1 : 0;
    const bool prefix_keep = (ttype == internal::GGML_TRI_TYPE_LOWER || ttype == internal::GGML_TRI_TYPE_LOWER_DIAG);

    if (prefix_keep) {
        if (add_to_split == 0) {
            tri_kernel<T, true, 0> << <grid_dims, block_dims, 0, stream >> > (
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
                );
        }
        else { // only 0 and 1 supported
            tri_kernel<T, true, 1> << <grid_dims, block_dims, 0, stream >> > (
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
                );
        }
    }
    else {
        if (add_to_split == 0) {
            tri_kernel<T, false, 0> << <grid_dims, block_dims, 0, stream >> > (
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
                );
        }
        else {
            tri_kernel<T, false, 1> << <grid_dims, block_dims, 0, stream >> > (
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
                );
        }
    }
}

void tri_cuda(const tri_context& ctx, cudaStream_t  stream)
{
    switch (ctx.src0_type) {
    case internal::GGML_TYPE_F32:
    {
        tri_cuda(
            (const float*)ctx.src0_d, (float*)ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_ne[1], ctx.src0_ne[2], ctx.src0_ne[3],
            ctx.src0_nb[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3],
            ctx.dst_nb[0], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3],
            ctx.ttype, stream
        );
    } break;
    case internal::GGML_TYPE_F16:
    {
        tri_cuda(
            (const half*)ctx.src0_d, (half*)ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_ne[1], ctx.src0_ne[2], ctx.src0_ne[3],
            ctx.src0_nb[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3],
            ctx.dst_nb[0], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3],
            ctx.ttype, stream
        );
    } break;
    case internal::GGML_TYPE_BF16:
    {
        tri_cuda(
            (const nv_bfloat16*)ctx.src0_d, (nv_bfloat16*)ctx.dst_d,
            ctx.src0_ne[0], ctx.src0_ne[1], ctx.src0_ne[2], ctx.src0_ne[3],
            ctx.src0_nb[0], ctx.src0_nb[1], ctx.src0_nb[2], ctx.src0_nb[3],
            ctx.dst_nb[0], ctx.dst_nb[1], ctx.dst_nb[2], ctx.dst_nb[3],
            ctx.ttype, stream
        );
    } break;
    default:
        GGML_ABORT("fatal error");
    }
}
