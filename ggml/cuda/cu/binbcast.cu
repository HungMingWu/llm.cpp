#include "cuda_func.h"
#include "common.cuh"
#define GGML_ASSERT(...)

static __device__ __forceinline__ float op_repeat(const float, const float b) {
    return b;
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast_unravel(const src0_t* src0, const src1_t* src1, dst_t* dst,
    int ne0, int ne1, int ne2, int ne3,
    int ne10, int ne11, int ne12, int ne13,
    /*int s0, */ int s1, int s2, int s3,
    /*int s00,*/ int s01, int s02, int s03,
    /*int s10,*/ int s11, int s12, int s13) {

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    const int i3 = i / (ne2 * ne1 * ne0);
    const int i2 = (i / (ne1 * ne0)) % ne2;
    const int i1 = (i / ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_dst = i3 * s3 + i2 * s2 + i1 * s1;

    const src0_t* src0_row = src0 + i_src0;
    const src1_t* src1_row = src1 + i_src1;
    dst_t* dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast(const src0_t* src0, const src1_t* src1, dst_t* dst,
    int ne0, int ne1, int ne2, int ne3,
    int ne10, int ne11, int ne12, int ne13,
    /*int s0, */ int s1, int s2, int s3,
    /*int s00,*/ int s01, int s02, int s03,
    /*int s10,*/ int s11, int s12, int s13) {
    const int i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const int i1 = (blockDim.y * blockIdx.y + threadIdx.y);
    const int i2 = (blockDim.z * blockIdx.z + threadIdx.z) / ne3;
    const int i3 = (blockDim.z * blockIdx.z + threadIdx.z) % ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_dst = i3 * s3 + i2 * s2 + i1 * s1;

    const src0_t* src0_row = src0 + i_src0;
    const src1_t* src1_row = src1 + i_src1;
    dst_t* dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_cuda {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(const bin_bcast_context* ctx, cudaStream_t stream) {

        int nr0 = ctx->ne10 / ctx->ne0;
        int nr1 = ctx->ne11 / ctx->ne1;
        int nr2 = ctx->ne12 / ctx->ne2;
        int nr3 = ctx->ne13 / ctx->ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = { ctx->ne0, ctx->ne1, ctx->ne2, ctx->ne3 };
        int64_t cne0[] = { ctx->ne00, ctx->ne01, ctx->ne02, ctx->ne03 };
        int64_t cne1[] = { ctx->ne10, ctx->ne11, ctx->ne12, ctx->ne13 };

        size_t cnb[] = { ctx->nb0, ctx->nb1, ctx->nb2, ctx->nb3 };
        size_t cnb0[] = { ctx->nb00, ctx->nb01, ctx->nb02, ctx->nb03 };
        size_t cnb1[] = { ctx->nb10, ctx->nb11, ctx->nb12, ctx->nb13 };

        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (ctx->src0_is_contiguous && ctx->src1_is_contiguous && ctx->dst_is_contiguous) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }

        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
            //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
            //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
            //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s00 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0 / 2LL, 1LL);

            dim3 block_dims;
            block_dims.x = std::min<unsigned int>(hne0, block_size);
            block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
            block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

            dim3 block_nums(
                (hne0 + block_dims.x - 1) / block_dims.x,
                (ne1 + block_dims.y - 1) / block_dims.y,
                (ne2 * ne3 + block_dims.z - 1) / block_dims.z
            );

            if (block_nums.z > 65535) {
                // this is the maximum number of blocks in z dimension, fallback to 1D grid kernel
                int block_num = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
                k_bin_bcast_unravel<bin_op> << <block_num, block_size, 0, stream >> > (
                    (const src0_t*)ctx->src0_d, (const src1_t *)ctx->src1_d, (dst_t *)ctx->dst_d,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            }
            else {
                k_bin_bcast<bin_op> << <block_nums, block_dims, 0, stream >> > (
                    (const src0_t*)ctx->src0_d, (const src1_t*)ctx->src1_d, (dst_t*)ctx->dst_d,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            }
        }
    }
};

template<class op>
static void ggml_cuda_op_bin_bcast(const bin_bcast_context* ctx, cudaStream_t stream) {
    if (ctx->src0_type == GGML_TYPE_F32 && ctx->dst_type == GGML_TYPE_F32) {
        op().template operator()<float, float, float>(ctx, stream);
    }
    else if (ctx->src0_type == GGML_TYPE_F16 && ctx->dst_type == GGML_TYPE_F16) {
        op().template operator()<half, float, half>(ctx, stream);
    }
    else if (ctx->src0_type == GGML_TYPE_F16 && ctx->dst_type == GGML_TYPE_F32) {
        op().template operator()<half, float, float>(ctx, stream);
    }
    else {
        //fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            //ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

void repeat_cuda(const bin_bcast_context* ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(ctx, stream);
}

void add_cuda(const bin_bcast_context* ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(ctx, stream);
}

void mul_cuda(const bin_bcast_context* ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(ctx, stream);
}

void div_cuda(const bin_bcast_context* ctx, cudaStream_t stream)
{
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(ctx, stream);
}