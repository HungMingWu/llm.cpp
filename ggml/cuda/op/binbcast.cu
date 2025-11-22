#include <array>
#include "cuda_func.h"
#include "common.cuh"
#include "helper.h"
#include "convert.cuh"
#include "launch.cuh"

#define GGML_ASSERT(...)

__device__ __forceinline__ float op_repeat(const float, const float b) {
    return b;
}

__device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

__device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

__device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

__device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template<class op>
static void ggml_cuda_op_bin_bcast(const bin_bcast_context &ctx, cudaStream_t stream) {
    GGML_ASSERT(src1->type == internal::GGML_TYPE_F32 || src1->type == internal::GGML_TYPE_F16);
    if (ctx.src0_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F32) {
        op().template operator() < float, float, float > (ctx, stream);
    }
    else if (ctx.src0_type == internal::GGML_TYPE_F16 && ctx.src1_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F16) {
        op().template operator() < half, half, half > (ctx, stream);
    }
    else if (ctx.src0_type == internal::GGML_TYPE_F16 && ctx.src1_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F16) {
        op().template operator() < half, float, half > (ctx, stream);
    }
    else if (ctx.src0_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F32) {
        op().template operator() < half, float, float > (ctx, stream);
    }
    else if (ctx.src0_type == internal::GGML_TYPE_BF16 && ctx.src1_type == internal::GGML_TYPE_BF16) {
        op().template operator() < nv_bfloat16, nv_bfloat16, nv_bfloat16 > (ctx, stream);
    }
    else if (ctx.src0_type == internal::GGML_TYPE_BF16 && ctx.src1_type == internal::GGML_TYPE_F32) {
        op().template operator() < nv_bfloat16, float, nv_bfloat16 > (ctx, stream);
    }
    else {
        //fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            //internal::GGML_TYPE_name(dst->type), internal::GGML_TYPE_name(src0->type), internal::GGML_TYPE_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t, size_t... I>
void launch_bin_bcast_pack(const bin_bcast_context &ctx, cudaStream_t stream, std::index_sequence<I...>) {
    const uint3 ne10 = init_fastdiv_values((uint32_t)ctx.src1_ne[0]);
    const uint3 ne11 = init_fastdiv_values((uint32_t)ctx.src1_ne[1]);
    const uint3 ne12 = init_fastdiv_values((uint32_t)ctx.src1_ne[2]);
    const uint3 ne13 = init_fastdiv_values((uint32_t)ctx.src1_ne[3]);
    auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(ctx.src_data[0]), ctx.src0_ne, ctx.src0_nb);
    auto src1_datas = std::array{
        (make_strided_mdspan(static_cast<const src1_t*>(ctx.src_data[1 + I]), ctx.src1_ne, ctx.src1_nb))... };
    auto dst_data = make_strided_mdspan(static_cast<dst_t*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);
    launch_functor(stream, std::make_tuple(ctx.dst_ne[3], ctx.dst_ne[2], ctx.dst_ne[1], ctx.dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            const int i11 = fastmodulo(i1, ne11);
            const int i12 = fastmodulo(i2, ne12);
            const int i13 = fastmodulo(i3, ne13);
            const int i10 = fastmodulo(i0, ne10);

            float result = ctx.src_data[0] ? ggml_cuda_cast<float>(src0_data(i3, i2, i1, i0)) : 0.0f;
            result = (..., (result = bin_op(result, ggml_cuda_cast<float>(src1_datas[I](i13, i12, i11, i10)))));
            dst_data(i3, i2, i1, i0) = ggml_cuda_cast<dst_t>(result);
        }
    );
}

template <float (*bin_op)(const float, const float), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const bin_bcast_context &ctx, cudaStream_t stream) {
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            ctx, stream, std::make_index_sequence<n_fuse>{});
    }
};

void repeat_cuda(const bin_bcast_context &ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(ctx, stream);
}

void add_cuda(const bin_bcast_context &ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(ctx, stream);
}

void sub_cuda(const bin_bcast_context &ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(ctx, stream);
}

void mul_cuda(const bin_bcast_context &ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(ctx, stream);
}

void div_cuda(const bin_bcast_context &ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(ctx, stream);
}

template <float (*op)(const float, const float), int n_fuse>
static void ggml_cuda_op_fused_binbcast_impl(const bin_bcast_context &ctx, cudaStream_t stream) {
    if (ctx.src0_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, float, float, float>(ctx, stream, std::make_index_sequence<n_fuse>{});
    }
    else if (ctx.src0_type == internal::GGML_TYPE_F16 && ctx.src1_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, half, half>(ctx, stream, std::make_index_sequence<n_fuse>{});
    }
    else if (ctx.src0_type == internal::GGML_TYPE_F16 && ctx.src1_type == internal::GGML_TYPE_F32 && ctx.dst_type == internal::GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, float, half>(ctx, stream, std::make_index_sequence<n_fuse>{});
    }
    else if (ctx.src0_type == internal::GGML_TYPE_F16 && ctx.dst_type == internal::GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, half, float, float>(ctx, stream, std::make_index_sequence<n_fuse>{});
    }
    else {
#if 0
        fprintf(stderr,
            "%s: unsupported types for fusion: dst: %s, src0: %s, src1: %s\n",
            __func__, internal::GGML_TYPE_name(ctx.dst_type), internal::GGML_TYPE_name(ctx.src0_type), internal::GGML_TYPE_name(ctx.src1_type));
        GGML_ABORT("fatal error");
#endif
    }
}

template <typename T>
static __global__ void k_repeat_back(
    const T* __restrict__ src, T* __restrict__ dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {

    const int64_t tid0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t tid1 = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    const int64_t tid23 = int64_t(blockIdx.z) * blockDim.z + threadIdx.z;
    const int64_t tid2 = tid23 % ne2;
    const int64_t tid3 = tid23 / ne2;

    if (tid0 >= ne0) {
        return;
    }

    T sum = 0;
    for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
        for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
            for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                    sum += src[i3 * s03 + i2 * s02 + i1 * s01 + i0 * s00];
                }
            }
        }
    }
    dst[tid3 * ne2 * ne1 * ne0 + tid2 * ne1 * ne0 + tid1 * ne0 + tid0] = sum;
}

template <typename T>
static void repeat_back_cuda(
    const T* src, T* dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums((ne0 + WARP_SIZE - 1) / WARP_SIZE, ne1, ne2 * ne3);
    k_repeat_back<T> << <block_nums, block_dims, 0, stream >> >
        (src, dst, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3);
}

void repeat_back_cuda(const repeat_back_context &ctx, cudaStream_t stream)
{
    const size_t s00 = ctx.nb00 / ctx.src0_ts;
    const size_t s01 = ctx.nb01 / ctx.src0_ts;
    const size_t s02 = ctx.nb02 / ctx.src0_ts;
    const size_t s03 = ctx.nb03 / ctx.src0_ts;
    switch (ctx.dst_type) {
    case internal::GGML_TYPE_F32: {
        const float* src0_d = (const float*)ctx.src0_d;
        float* dst_d = (float*)ctx.dst_d;
        repeat_back_cuda(src0_d, dst_d,
            ctx.ne00, ctx.ne01, ctx.ne02, ctx.ne03,
            s00, s01, s02, s03,
            ctx.ne0, ctx.ne1, ctx.ne2, ctx.ne3, stream);
    } break;
    default: {
        GGML_ASSERT(false);
    } break;
    }
}

void fused_add_cuda(const bin_bcast_context &ctx, int n_fuse, cudaStream_t stream) {
    GGML_ASSERT(2 <= n_fuse && n_fuse <= 8);

    switch (n_fuse) {
    case 2:
        ggml_cuda_op_fused_binbcast_impl<op_add, 2>(ctx, stream);
        break;
    case 3:
        ggml_cuda_op_fused_binbcast_impl<op_add, 3>(ctx, stream);
        break;
    case 4:
        ggml_cuda_op_fused_binbcast_impl<op_add, 4>(ctx, stream);
        break;
    case 5:
        ggml_cuda_op_fused_binbcast_impl<op_add, 5>(ctx, stream);
        break;
    case 6:
        ggml_cuda_op_fused_binbcast_impl<op_add, 6>(ctx, stream);
        break;
    case 7:
        ggml_cuda_op_fused_binbcast_impl<op_add, 7>(ctx, stream);
        break;
    case 8:
        ggml_cuda_op_fused_binbcast_impl<op_add, 8>(ctx, stream);
        break;
    default:
        GGML_ASSERT(false && "Unsupported n_fuse value");
    }
}