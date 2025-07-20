module;
#include <assert.h>
#include <bit>
#include <vector>
#include "common.h"
#include "cu/cuda_func.h"
#include "vendor_constant.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml:cuda.op;
import :ds;
import :func;
import :tensor;
import :cuda.buffer;

static dup_context create(const ggml_tensor* src0, ggml_tensor* dst)
{
   return dup_context {
       .src_d = src0->data,
       .dst_d = dst->data,
       .src_type = src0->type,
       .dst_type = dst->type,
       .ne = src0->nelements(),
       .src_length = src0->nbytes(),
       .ne00 = src0->ne[0],
       .ne01 = src0->ne[1],
       .ne02 = src0->ne[2],
       .ne03 = src0->ne[3],
       .nb00 = src0->nb[0],
       .nb01 = src0->nb[1],
       .nb02 = src0->nb[2],
       .nb03 = src0->nb[3],
       .ne10 = dst->ne[0],
       .ne11 = dst->ne[1],
       .ne12 = dst->ne[2],
       .ne13 = dst->ne[3],
       .nb10 = dst->nb[0],
       .nb11 = dst->nb[1],
       .nb12 = dst->nb[2],
       .nb13 = dst->nb[3],
       .src_is_contiguous = ggml_is_contiguous(src0),
       .dst_is_contiguous = ggml_is_contiguous(dst)
    };
}

static unary_context create(const ggml_tensor* src0, ggml_tensor* dst, cudaStream_t stream)
{
    return {
        .stream = stream,
        .src0_type = src0->type,
        .dst_type = dst->type,
        .src0_d = src0->data,
        .dst_d = dst->data,
        .nelements = src0->nelements()
    };
}

static bin_bcast_context create_bcast_context(const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst)
{
    return bin_bcast_context{
        .src0_d = src0->data,
        .src1_d = src1->data,
        .dst_d = dst->data,
        .src0_type = src0->type,
        .src1_type = src1->type,
        .dst_type = dst->type,
        .ne00 = src0->ne[0],
        .ne01 = src0->ne[1],
        .ne02 = src0->ne[2],
        .ne03 = src0->ne[3],
        .nb00 = src0->nb[0],
        .nb01 = src0->nb[1],
        .nb02 = src0->nb[2],
        .nb03 = src0->nb[3],
        .ne10 = src1->ne[0],
        .ne11 = src1->ne[1],
        .ne12 = src1->ne[2],
        .ne13 = src1->ne[3],
        .nb10 = src1->nb[0],
        .nb11 = src1->nb[1],
        .nb12 = src1->nb[2],
        .nb13 = src1->nb[3],
        .ne0 = dst->ne[0],
        .ne1 = dst->ne[1],
        .ne2 = dst->ne[2],
        .ne3 = dst->ne[3],
        .nb0 = dst->nb[0],
        .nb1 = dst->nb[1],
        .nb2 = dst->nb[2],
        .nb3 = dst->nb[3],
        .src0_is_contiguous = ggml_is_contiguous(src0),
        .src1_is_contiguous = ggml_is_contiguous(src1),
        .dst_is_contiguous = ggml_is_contiguous(dst),
    };
}

namespace op
{
    void arange(cudaStream_t stream, ggml_tensor* dst)
    {
        float* dst_d = (float*)dst->data;
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const float start = std::bit_cast<float>(dst->op_params[0]);
        const float stop = std::bit_cast<float>(dst->op_params[1]);
        const float step = std::bit_cast<float>(dst->op_params[2]);

        int64_t steps = (int64_t)ceil((stop - start) / step);
        GGML_ASSERT(dst->nelements() == steps);

        arange_f32_cuda(dst_d, (size_t)dst->ne[0], start, step, stream);
    }

    void conv_transpose_1d(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;

        const ggml_tensor* src1 = dst->src[1];
        const float* src1_d = (const float*)src1->data;

        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));

        const int s0 = std::bit_cast<int>(dst->op_params[0]);

        const int64_t output_size = dst->nelements();

        conv_transpose_1d_f32_f32_cuda(s0,
            src0->ne[0], src0->ne[1], src0->ne[2],
            src1->ne[0], src1->ne[1],
            dst->ne[0], dst->ne[1],
            src0_d, src1_d, dst_d, (size_t)output_size, stream);
    }

    void mul_mat_vec(cudaStream_t stream, const ggml_tensor* ids, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const size_t ts_src0 = ggml_type_size(src0->type);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const size_t ts_dst = ggml_type_size(dst->type);

        GGML_ASSERT(!ids || src1->ne[2] == 1); // Implementation is only correct for  batch size 1.
        GGML_ASSERT(src1->ne[3] == dst->ne[3]);

        GGML_ASSERT(src0->nb[0] == ts_src0);
        GGML_ASSERT(src1->nb[0] == ts_src1);
        GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
        GGML_ASSERT(dst->nb[0] == ts_dst);

        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

        const int64_t s01 = src0->nb[1] / ts_src0;
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s1 = dst->nb[1] / ts_dst;
        const int64_t s02 = src0->nb[2] / ts_src0;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s2 = dst->nb[2] / ts_dst;
        const int64_t s03 = src0->nb[3] / ts_src0;
        const int64_t s13 = src1->nb[3] / ts_src1;
        const int64_t s3 = dst->nb[3] / ts_dst;

        // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
        const int64_t ncols_dst = ids ? dst->ne[2] : dst->ne[1];
        const int64_t nchannels_y = ids ? src1->ne[1] : src1->ne[2];
        const int64_t nchannels_dst = ids ? dst->ne[1] : dst->ne[2];
        const int64_t stride_channel_dst = ids ? s1 : s2;
        const int64_t stride_channel_y = ids ? s11 : s12;

        GGML_ASSERT(!ids || ncols_dst == 1);

        mul_mat_vec_context ctx{
            .src0_type = src0->type,
            .prec = prec,
            .src0_d = src0->data,
            .src1_d = (const float*)src1->data,
            .ids_d = ids ? (const int32_t*)ids->data : nullptr,
            .dst_d = (float*)dst->data,
            .ncols = src0->ne[0],
            .nrows = src0->ne[1],
			.ncols_dst = ncols_dst,
            .stride_row = s01,
            .stride_col_y = s11,
			.stride_col_dst = s1,
            .nchannels_x = src0->ne[2],
            .nchannels_y = nchannels_y,
            .nchannels_dst = nchannels_dst,
            .stride_channel_x = s02,
            .stride_channel_y = stride_channel_y,
            .stride_channel_dst = stride_channel_dst,
			.nsamples_x = src0->ne[3],
            .nsamples_dst = dst->ne[3],
            .stride_sample_x = s03,
            .stride_sample_y = s13,
            .stride_sample_dst = s3,
        };
		mul_mat_vec_cuda(&ctx, stream);
    }

    void pool2d(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int32_t* opts = (const int32_t*)dst->op_params;
        enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
        const int k0 = opts[1];
        const int k1 = opts[2];
        const int s0 = opts[3];
        const int s1 = opts[4];
        const int p0 = opts[5];
        const int p1 = opts[6];

        const int64_t IH = src0->ne[1];
        const int64_t IW = src0->ne[0];

        const int64_t N = dst->ne[3];
        const int64_t OC = dst->ne[2];
        const int64_t OH = dst->ne[1];
        const int64_t OW = dst->ne[0];

        const int parallel_elements = N * OC * OH * OW;

        pool2d_nchw_kernel_f32_f32_cuda(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0, parallel_elements, src0_d, dst_d, op, stream);
    }

    void im2col(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

        const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
        const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
        const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
        const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
        const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
        const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

        const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

        const int64_t IC = src1->ne[is_2D ? 2 : 1];
        const int64_t IH = is_2D ? src1->ne[1] : 1;
        const int64_t IW = src1->ne[0];

        const int64_t KH = is_2D ? src0->ne[1] : 1;
        const int64_t KW = src0->ne[0];

        const int64_t OH = is_2D ? dst->ne[2] : 1;
        const int64_t OW = dst->ne[1];

        const size_t  delta_offset = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
        const int64_t batch = src1->ne[is_2D ? 3 : 2];
        const size_t  batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

        if (dst->type == GGML_TYPE_F16) {
            im2col_cuda_f16(src1_d, (half*)dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
        }
        else {
            im2col_cuda_f32(src1_d, (float*)dst_d, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
        }
    }

    void cpy(cudaStream_t stream, ggml_tensor* dst) {
        dup_context context = create(dst->src[0], dst->src[1]);
        dup_cuda(&context, stream);
    }

    void dup(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(src0->nelements() == dst->nelements());
        GGML_ASSERT(src0->nbytes() <= INT_MAX);
        GGML_ASSERT(dst->nbytes() <= INT_MAX);
        //GGML_ASSERT(src0->ne[3] == 1);
        //GGML_ASSERT(dst->ne[3] == 1);
        dup_context context = create(src0, dst);
        if (src0->type == dst->type && context.src_is_contiguous && context.dst_is_contiguous) {
            GGML_ASSERT(src0->nbytes() == dst->nbytes());
        }
        dup_cuda(&context, stream);
    }

    void unary(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));

        unary_context ctx = create(src0, dst, stream);
        switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_ABS:
            return abs_cuda(&ctx);
        case GGML_UNARY_OP_SGN:
            return sgn_cuda(&ctx);
        case GGML_UNARY_OP_NEG:
            return neg_cuda(&ctx);
        case GGML_UNARY_OP_STEP:
            return step_cuda(&ctx);
        case GGML_UNARY_OP_GELU:
            return gelu_cuda(&ctx);
        case GGML_UNARY_OP_SILU:
            return silu_cuda(&ctx);
        case GGML_UNARY_OP_GELU_ERF:
            return gelu_erf_cuda(&ctx);
        case GGML_UNARY_OP_GELU_QUICK:
            return gelu_quick_cuda(&ctx);
        case GGML_UNARY_OP_TANH:
            return tanh_cuda(&ctx);
        case GGML_UNARY_OP_RELU:
            return relu_cuda(&ctx);
        case GGML_UNARY_OP_SIGMOID:
            return sigmoid_cuda(&ctx);
        case GGML_UNARY_OP_HARDSIGMOID:
            return hardsigmoid_cuda(&ctx);
        case GGML_UNARY_OP_HARDSWISH:
            return hardswish_cuda(&ctx);
        case GGML_UNARY_OP_EXP:
            return exp_cuda(&ctx);
        case GGML_UNARY_OP_ELU:
            return elu_cuda(&ctx);
        default:
            assert(false);
        }
    }

    void get_rows(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(src1->ne[3] == 1);

        GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
        GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
        GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

        get_row_context ctx{
            .stream = stream,
            .src0_d = src0->data,
            .src0_type = src0->type,
            .src1_d = (const int32_t*)src1->data,
            .dst_d = dst->data,
            .dst_type = dst->type,
            .ne00 = src0->ne[0],
            .nb01 = src0->nb[1],
            .nb02 = src0->nb[2],
            .nb03 = src0->nb[3],
            .ne10 = src1->ne[0],
            .ne11 = src1->ne[1],
            .ne12 = src1->ne[2],
            .nb10 = src1->nb[0],
            .nb11 = src1->nb[1],
            .nb12 = src1->nb[2],
            .nb1 = dst->nb[1],
            .nb2 = dst->nb[2],
            .nb3 = dst->nb[3]
        };
        get_rows_cuda(&ctx);
    }

    void get_rows_back(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0]; // gradients of forward pass output
        const ggml_tensor* src1 = dst->src[1]; // src1 in forward pass

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(src0->ne[2] * src0->ne[3] == 1);
        GGML_ASSERT(src1->ne[2] * src1->ne[3] == 1);
        GGML_ASSERT(dst->ne[2] * dst->ne[3] == 1);
        get_row_back_context context{
            .src0_d = (const float*)src0->data,
            .src1_d = (const int32_t*)src1->data,
            .dst_d = (float*)dst->data,
            .ne00 = src0->ne[0],
            .ne10 = src1->ne[0],
            .ne1 = dst->ne[1],
        };
        get_rows_back_cuda(&context, stream);
    }

    void argmax(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_I32);

        GGML_ASSERT(ggml_is_contiguous(src0));

        argmax_context context{
            .src0_d = (const float*)src0->data,
            .dst_d = (int32_t*)dst->data,
            .ne00 = src0->ne[0],
            .nrows = ggml_nrows(src0)
        };
        argmax_cuda(&context, stream);
    }

    void count_equal(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == src1->type);
        GGML_ASSERT(dst->type == GGML_TYPE_I64);

        GGML_ASSERT(ggml_are_same_shape(src0, src1));
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(src0->type == GGML_TYPE_I32);

        const int64_t ne = src0->nelements();
        GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
        count_equal_context context{
            .src0_d = (const int*)src0->data,
            .src1_d = (const int*)src1->data,
            .dst_d = (int64_t*)dst->data,
            .dst_size = dst->nbytes(),
            .ne = ne,
        };
        count_equal_cuda(&context, stream);
    }

    void repeat(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        bin_bcast_context context{
            .src0_d = nullptr,
            .src1_d = src0->data,
            .dst_d = dst->data,
            .src0_type = dst->type,
            .src1_type = src0->type,
            .dst_type = dst->type,
            .ne00 = dst->ne[0],
            .ne01 = dst->ne[1],
            .ne02 = dst->ne[2],
            .ne03 = dst->ne[3],
            .nb00 = dst->nb[0],
            .nb01 = dst->nb[1],
            .nb02 = dst->nb[2],
            .nb03 = dst->nb[3],
            .ne10 = src0->ne[0],
            .ne11 = src0->ne[1],
            .ne12 = src0->ne[2],
            .ne13 = src0->ne[3],
            .nb10 = src0->nb[0],
            .nb11 = src0->nb[1],
            .nb12 = src0->nb[2],
            .nb13 = src0->nb[3],
            .ne0 = dst->ne[0],
            .ne1 = dst->ne[1],
            .ne2 = dst->ne[2],
            .ne3 = dst->ne[3],
            .nb0 = dst->nb[0],
            .nb1 = dst->nb[1],
            .nb2 = dst->nb[2],
            .nb3 = dst->nb[3],
            .src0_is_contiguous = ggml_is_contiguous(dst),
            .src1_is_contiguous = ggml_is_contiguous(src0),
            .dst_is_contiguous = ggml_is_contiguous(dst),
        };
        repeat_cuda(&context, stream);
    }

    void add(cudaStream_t stream, ggml_tensor* dst) {
        bin_bcast_context context = create_bcast_context(dst->src[0], dst->src[1], dst);
        add_cuda(&context, stream);
    }

    void sub(cudaStream_t stream, ggml_tensor* dst) {
        bin_bcast_context context = create_bcast_context(dst->src[0], dst->src[1], dst);
        sub_cuda(&context, stream);
    }

    void mul(cudaStream_t stream, ggml_tensor* dst) {
        bin_bcast_context context = create_bcast_context(dst->src[0], dst->src[1], dst);
        mul_cuda(&context, stream);
    }

    void div(cudaStream_t stream, ggml_tensor* dst) {
        bin_bcast_context context = create_bcast_context(dst->src[0], dst->src[1], dst);
        div_cuda(&context, stream);
    }

    void scale(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const float scale = std::bit_cast<float>(dst->op_params[0]);
        const float bias = std::bit_cast<float>(dst->op_params[1]);

        scale_f32_cuda(src0_d, dst_d, scale, bias, src0->nelements(), stream);
    }

    void norm(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        float eps = std::bit_cast<float>(dst->op_params[0]);
        GGML_ASSERT(eps >= 0.0f);

        const size_t ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        const int64_t s01 = src0->nb[1] / ts0;
        const int64_t s02 = src0->nb[2] / ts0;
        const int64_t s03 = src0->nb[3] / ts0;

        norm_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }

    void rms_norm(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        float eps = std::bit_cast<float>(dst->op_params[0]);
        GGML_ASSERT(eps >= 0.0f);

        const size_t ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        const int64_t s01 = src0->nb[1] / ts0;
        const int64_t s02 = src0->nb[2] / ts0;
        const int64_t s03 = src0->nb[3] / ts0;

        rms_norm_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }

    void rms_norm_back(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* grad = dst->src[0]; // gradients
        const ggml_tensor* src0f = dst->src[1]; // src0 from forward pass

        const float* grad_d = (const float*)grad->data;
        const float* src0f_d = (const float*)src0f->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(ggml_is_contiguous(grad));

        GGML_ASSERT(grad->type == GGML_TYPE_F32);
        GGML_ASSERT(src0f->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int64_t ne00 = src0f->ne[0];
        const int64_t nrows = ggml_nrows(src0f);

        float eps = std::bit_cast<float>(dst->op_params[0]);
        GGML_ASSERT(eps >= 0.0f);

        rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
    }

    void silu_back(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0]; // input from forward pass
        const ggml_tensor* src1 = dst->src[1]; // grads of forward pass output

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(ggml_is_contiguous(src0));

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        silu_back_f32_cuda(src0_d, src1_d, dst_d, src0->nelements(), stream);
    }

    void rwkv_wkv6(cudaStream_t stream, ggml_tensor* dst) {
        const float* k_d = (const float*)dst->src[0]->data;
        const float* v_d = (const float*)dst->src[1]->data;
        const float* r_d = (const float*)dst->src[2]->data;
        const float* tf_d = (const float*)dst->src[3]->data;
        const float* td_d = (const float*)dst->src[4]->data;
        const float* s_d = (const float*)dst->src[5]->data;

        const int64_t B = dst->src[5]->ne[1];
        const int64_t T = dst->src[0]->ne[2];
        const int64_t C = dst->ne[0];
        const int64_t H = dst->src[0]->ne[1];

        float* dst_d = (float*)dst->data;
        GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
        GGML_ASSERT(C % H == 0);
        
        rwkv_wkv_cuda(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d, stream);
    }

    void rwkv_wkv7(cudaStream_t stream, ggml_tensor* dst) {
        const float* r_d = (const float*)dst->src[0]->data;
        const float* w_d = (const float*)dst->src[1]->data;
        const float* k_d = (const float*)dst->src[2]->data;
        const float* v_d = (const float*)dst->src[3]->data;
        const float* a_d = (const float*)dst->src[4]->data;
        const float* b_d = (const float*)dst->src[5]->data;
        const float* s_d = (const float*)dst->src[6]->data;

        const int64_t B = dst->src[6]->ne[1];
        const int64_t T = dst->src[0]->ne[2];
        const int64_t C = dst->ne[0];
        const int64_t H = dst->src[0]->ne[1];

        float* dst_d = (float*)dst->data;

        GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
        GGML_ASSERT(C % H == 0);
        rwkv_wkv7_cuda(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, stream);
    }

    void gated_linear_attn(cudaStream_t stream, ggml_tensor* dst) {
        const int64_t C = dst->ne[0];
        const int64_t H = dst->src[0]->ne[1];

        GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
        GGML_ASSERT(C % H == 0);
        GGML_ASSERT(C / H == 64 || C / H == 128);
        gla_context ctx{
            .B = dst->src[4]->ne[1],
            .T = dst->src[0]->ne[2],
            .C = dst->ne[0],
            .H = dst->src[0]->ne[1],
            .scale = std::bit_cast<float>(dst->op_params[0]),
            .k = (const float*)dst->src[0]->data,
            .v = (const float*)dst->src[1]->data,
            .r = (const float*)dst->src[2]->data,
            .td = (const float*)dst->src[3]->data,
            .s = (const float*)dst->src[4]->data,
            .dst = (float*)dst->data
        };
        gated_linear_attn_cuda(&ctx, stream);
    }

    void mul_mat_vec_q(cudaStream_t stream, const ggml_tensor* ids, ggml_tensor* dst, ggml_cuda_pool& pool)
    {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

        const size_t ts_src0 = ggml_type_size(src0->type);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const size_t ts_dst = ggml_type_size(dst->type);

        GGML_ASSERT(src0->nb[0] == ts_src0);
        GGML_ASSERT(src1->nb[0] == ts_src1);
        GGML_ASSERT(dst->nb[0] == ts_dst);
        GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

        GGML_ASSERT(!ids || src1->ne[2] == 1); // Implementation is only correct for  batch size 1.

        const float* src1_d = (const float*)src1->data;
        const int32_t* ids_d = ids ? (const int32_t*)ids->data : nullptr;
        float* dst_d = (float*)dst->data;

        // If src0 is a temporary compute buffer, clear any potential padding.
        if (src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size_data = src0->nbytes();
            const size_t size_alloc = src0->buffer->get_alloc_size(src0);
            if (size_alloc > size_data) {
                GGML_ASSERT(ggml_is_contiguously_allocated(src0));
                GGML_ASSERT(!src0->view_src);
                CUDA_CHECK(cudaMemsetAsync((char*)src0->data + size_data, 0, size_alloc - size_data, stream));
            }
        }

        const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
        ggml_cuda_pool_alloc<char> src1_q8_1(pool,
            src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1);
        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
                src1->ne[0], s11, s12, s13, ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);
        }

        const int64_t s01 = src0->nb[1] / ts_src0;
        const int64_t s11 = ne10_padded / QK8_1;
        const int64_t s1 = dst->nb[1] / ts_dst;
        const int64_t s02 = src0->nb[2] / ts_src0;
        const int64_t s2 = dst->nb[2] / ts_dst;
        const int64_t s03 = src0->nb[3] / ts_src0;
        const int64_t s3 = dst->nb[3] / ts_dst;

        const int64_t s12 = src1->ne[1] * s11;
        const int64_t s13 = src1->ne[2] * s12;

        // For MUL_MAT_ID the memory layout is different than for MUL_MAT:

        mat_vec_q_switch_context ctx{
            .type_x = src0->type,
            .vx = src0->data,
            .vy = src1_q8_1.get(),
            .ids = ids_d,
            .dst = dst_d,
            .ncols_x = src0->ne[0],
            .nrows_x = src0->ne[1],
            .ncols_dst = ids ? dst->ne[2] : dst->ne[1],
            .stride_row_x = s01,
            .stride_col_y = ids ? s12 : s11,
            .stride_col_dst = ids ? s2 : s1,
            .nchannels_x = src0->ne[2],
            .nchannels_y = ids ? src1->ne[1] : src1->ne[2],
            .nchannels_dst = ids ? dst->ne[1] : dst->ne[2],
            .stride_channel_x = s02,
            .stride_channel_y = ids ? s11 : s12,
            .stride_channel_dst = ids ? s1 : s2,
            .nsamples_x = src0->ne[3],
            .nsamples_dst = dst->ne[3],
            .stride_sample_x = s03,
            .stride_sample_y = s13,
            .stride_sample_dst = s3
        };

        mul_mat_vec_q_switch_type(&ctx, stream);
    }

    void mul_mat_q(
        cudaStream_t stream, ggml_cuda_pool& pool, const ggml_tensor* ids, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

        const size_t ts_src0 = ggml_type_size(src0->type);
        const size_t ts_src1 = ggml_type_size(src1->type);
        const size_t ts_dst = ggml_type_size(dst->type);

        GGML_ASSERT(src0->nb[0] == ts_src0);
        GGML_ASSERT(src1->nb[0] == ts_src1);
        GGML_ASSERT(dst->nb[0] == ts_dst);
        GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

        const char* src0_d = (const char*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        // If src0 is a temporary compute buffer, clear any potential padding.
        if (src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size_data = src0->nbytes();
            const size_t size_alloc = src0->buffer->get_alloc_size(src0);
            if (size_alloc > size_data) {
                GGML_ASSERT(ggml_is_contiguously_allocated(src0));
                GGML_ASSERT(!src0->view_src);
                CUDA_CHECK(cudaMemsetAsync((char*)src0->data + size_data, 0, size_alloc - size_data, stream));
            }
        }

        const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);

        const int64_t s01 = src0->nb[1] / ts_src0;
        const int64_t s1 = dst->nb[1] / ts_dst;
        const int64_t s02 = src0->nb[2] / ts_src0;
        const int64_t s2 = dst->nb[2] / ts_dst;
        const int64_t s03 = src0->nb[3] / ts_src0;
        const int64_t s3 = dst->nb[3] / ts_dst;

        const bool use_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;

        if (!ids) {
            const size_t nbytes_src1_q8_1 = src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1 +
                get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
            ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);

            {
                const int64_t s11 = src1->nb[1] / ts_src1;
                const int64_t s12 = src1->nb[2] / ts_src1;
                const int64_t s13 = src1->nb[3] / ts_src1;
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
                    src1->ne[0], s11, s12, s13, ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);
            }

            const int64_t s12 = src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
            const int64_t s13 = src1->ne[2] * s12;

            const mmq_args args = {
                src0_d, src0->type, (const int*)src1_q8_1.ptr, nullptr, nullptr, dst_d,
                src0->ne[0], src0->ne[1], dst->ne[1], s01, src1->ne[1], s1,
                src0->ne[2], src1->ne[2], s02, s12, s2,
                src0->ne[3], src1->ne[3], s03, s13, s3,
                use_stream_k };
            ggml_cuda_mul_mat_q_switch_type(pool, args, stream);
            return;
        }

        GGML_ASSERT(src1->ne[3] == 1);
        GGML_ASSERT(src1->nb[2] % src1->nb[1] == 0);
        GGML_ASSERT(dst->nb[2] % dst->nb[1] == 0);

        const int64_t n_expert_used = ids->ne[0];
        const int64_t ne_get_rows = src1->ne[2] * n_expert_used;

        std::vector<char> ids_host(ids->nbytes());
        std::vector<int32_t> ids_src1_host;
        ids_src1_host.reserve(ne_get_rows);
        std::vector<int32_t> ids_dst_host;
        ids_dst_host.reserve(ne_get_rows);
        std::vector<int32_t> tokens_per_expert_host(src0->ne[2]);
        std::vector<int32_t> expert_bounds_host(src0->ne[2] + 1);
        ggml_cuda_pool_alloc<int32_t> ids_buf_dev(pool);

        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids->nbytes(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) { // expert matrices
            for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) { // tokens
                for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                    const int32_t expert_to_use = *(const int32_t*)(ids_host.data() + i12 * ids->nb[1] + iex * ids->nb[0]);
                    assert(expert_to_use >= 0 && expert_to_use < src0->ne[2]);
                    if (expert_to_use == i02) {
                        ids_src1_host.push_back(i12 * (src1->nb[2] / src1->nb[1]) + iex % src1->ne[1]);
                        ids_dst_host.push_back(i12 * dst->ne[1] + iex);
                        tokens_per_expert_host[i02]++;
                        break;
                    }
                }
            }
        }

        int32_t cumsum = 0;
        for (int64_t i = 0; i < src0->ne[2]; ++i) {
            expert_bounds_host[i] = cumsum;
            cumsum += tokens_per_expert_host[i];
        }
        expert_bounds_host[src0->ne[2]] = cumsum;

        std::vector<int32_t> ids_buf_host;
        ids_buf_host.reserve(ids_src1_host.size() + ids_dst_host.size() + expert_bounds_host.size());
        ids_buf_host.insert(ids_buf_host.end(), ids_src1_host.begin(), ids_src1_host.end());
        ids_buf_host.insert(ids_buf_host.end(), ids_dst_host.begin(), ids_dst_host.end());
        ids_buf_host.insert(ids_buf_host.end(), expert_bounds_host.begin(), expert_bounds_host.end());
        ids_buf_dev.alloc(ids_buf_host.size() + get_mmq_x_max_host(cc)); // Expert bounds are padded on device.
        CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_buf_host.data(), ids_buf_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int32_t* ids_src1_dev = ids_buf_dev.ptr;
        const int32_t* ids_dst_dev = ids_src1_dev + ids_src1_host.size();
        const int32_t* expert_bounds_dev = ids_dst_dev + ids_dst_host.size();

        const size_t nbytes_src1_q8_1 = src1->ne[2] * n_expert_used * ne10_padded * sizeof(block_q8_1) / QK8_1 +
            get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(pool, nbytes_src1_q8_1);

        const int64_t ne11_flat = src1->ne[2] * n_expert_used;
        const int64_t ne12_flat = 1;
        const int64_t ne13_flat = 1;

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[2] / ts_src1;
            quantize_mmq_q8_1_cuda(src1_d, ids_src1_dev, src1_q8_1.get(), src0->type,
                src1->ne[0], s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }

        const int64_t s12 = src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = src1->ne[2] * s12;

        // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
        const mmq_args args = {
            src0_d, src0->type, (const int*)src1_q8_1.ptr, ids_dst_dev, expert_bounds_dev, dst_d,
            src0->ne[0], src0->ne[1], ne_get_rows, s01, ne_get_rows, s1,
            src0->ne[2], src0->ne[2], s02, s12, s2,
            src0->ne[3], src1->ne[3], s03, s13, s3,
            use_stream_k };

        ggml_cuda_mul_mat_q_switch_type(pool, args, stream);
    }

    void mul_mat_id(cudaStream_t stream, ggml_tensor* dst, ggml_cuda_pool& pool, auto mat_mul_cb) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const ggml_tensor* ids = dst->src[2];

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(!to_split_buffer_type(src0->buffer->get_type()) && "mul_mat_id does not support split buffers");

        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (dst->ne[2] == 1) {
                if (ggml_is_quantized(src0->type)) {
                    mul_mat_vec_q(stream, ids, dst, pool);
                }
                else {
                    mul_mat_vec(stream, ids, dst);
                }
                return;
            }

            if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2])) {
                mul_mat_q(stream, pool, ids, dst);
                return;
            }
        }

        GGML_ASSERT(src1->nb[2] % src1->nb[1] == 0);
        GGML_ASSERT(dst->nb[2] % dst->nb[1] == 0);

        const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
            || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
        const ggml_type type_dst_sorted = GGML_TYPE_F32;
        const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
        const size_t ts_dst_sorted = ggml_type_size(type_dst_sorted);

        const int64_t n_expert_used = ids->ne[0];
        const int64_t ne_get_rows = src1->ne[2] * n_expert_used;

        std::vector<int32_t> ids_to_sorted_host;
        ids_to_sorted_host.reserve(2 * ne_get_rows);
        std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

        ggml_cuda_pool_alloc<int32_t> ids_buf_dev(pool, 2 * ne_get_rows);

        std::vector<int32_t> tokens_per_expert(src0->ne[2]);

        ggml_cuda_pool_alloc<char> src1_sorted(pool, src1->ne[2] * n_expert_used * src1->ne[0] * ts_src1_sorted);
        ggml_cuda_pool_alloc<char>  dst_sorted(pool, dst->ne[2] * n_expert_used * dst->ne[0] * ts_dst_sorted);

        std::vector<char> ids_host(ids->nbytes());
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids->nbytes(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) { // expert matrices
            for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) { // tokens
                for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                    const int32_t expert_to_use = *(const int32_t*)(ids_host.data() + i12 * ids->nb[1] + iex * ids->nb[0]);
                    assert(expert_to_use >= 0 && expert_to_use < src0->ne[2]);
                    if (expert_to_use == i02) {
                        ids_from_sorted_host[i12 * n_expert_used + iex] = ids_to_sorted_host.size();
                        ids_to_sorted_host.push_back(i12 * src1->ne[1] + iex % src1->ne[1]);
                        tokens_per_expert[i02]++;
                        break;
                    }
                }
            }
        }
        GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

        ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

        CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2 * ne_get_rows * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int32_t* ids_to_sorted = ids_buf_dev.ptr + 0 * ne_get_rows;
        const int32_t* ids_from_sorted = ids_buf_dev.ptr + 1 * ne_get_rows;

        get_row_context ctx1{
            .stream = stream,
            .src0_d = src1->data,
            .src0_type = src1->type,
            .src1_d = ids_to_sorted,
            .dst_d = src1_sorted.ptr,
            .dst_type = type_src1_sorted,
            .ne00 = src1->ne[0],
            .nb01 = src1->nb[1],
            .nb02 = src1->nb[2],
            .nb03 = src1->nb[3],
            .ne10 = ne_get_rows,
            .ne11 = 1,
            .ne12 = 1,
            .nb10 = sizeof(int32_t),
            .nb11 = ne_get_rows * sizeof(int32_t),
            .nb12 = ne_get_rows * sizeof(int32_t),
            .nb1 = src1->ne[0] * ts_src1_sorted,
            .nb2 = ne_get_rows * src1->ne[0] * ts_src1_sorted,
            .nb3 = ne_get_rows * src1->ne[0] * ts_src1_sorted
        };

        get_rows_cuda(&ctx1);
        CUDA_CHECK(cudaGetLastError());

        char* src1_data_cur = (char*)src1_sorted.ptr;
        char* dst_data_cur = (char*)dst_sorted.ptr;
        for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) {
            if (tokens_per_expert[i02] == 0) {
                continue;
            }

            ggml_tensor src0_slice = *src0;
            src0_slice.ne[2] = 1;
            src0_slice.nb[3] = src0_slice.nb[2];
            src0_slice.op = GGML_OP_VIEW;
            src0_slice.view_src = dst->src[0]; // non-const pointer to src0
            src0_slice.data = (char*)src0->data + i02 * src0->nb[2];

            ggml_tensor src1_slice;
            src1_slice.buffer = src1->buffer;
            src1_slice.type = type_src1_sorted;
            src1_slice.ne[0] = src1->ne[0];
            src1_slice.ne[1] = tokens_per_expert[i02];
            src1_slice.ne[2] = 1;
            src1_slice.ne[3] = 1;
            src1_slice.nb[0] = ts_src1_sorted;
            src1_slice.nb[1] = src1_slice.ne[0] * src1_slice.nb[0];
            src1_slice.nb[2] = src1_slice.ne[1] * src1_slice.nb[1];
            src1_slice.nb[3] = src1_slice.ne[2] * src1_slice.nb[2];
            src1_slice.data = src1_data_cur;

            ggml_tensor dst_slice;
            dst_slice.buffer = dst->buffer;
            dst_slice.type = type_dst_sorted;
            dst_slice.ne[0] = dst->ne[0];
            dst_slice.ne[1] = tokens_per_expert[i02];
            dst_slice.ne[2] = 1;
            dst_slice.ne[3] = 1;
            dst_slice.nb[0] = ts_dst_sorted;
            dst_slice.nb[1] = dst_slice.ne[0] * dst_slice.nb[0];
            dst_slice.nb[2] = dst_slice.ne[1] * dst_slice.nb[1];
            dst_slice.nb[3] = dst_slice.ne[2] * dst_slice.nb[2];
            dst_slice.data = dst_data_cur;
            dst_slice.src.push_back(&src0_slice);
            dst_slice.src.push_back(&src1_slice);

            mat_mul_cb(&dst_slice);
            CUDA_CHECK(cudaGetLastError());

            src1_data_cur += src1_slice.nb[2];
            dst_data_cur += dst_slice.nb[2];
        }

        get_row_context ctx2{
            .stream = stream,
            .src0_d = dst_sorted.ptr,
            .src0_type = type_dst_sorted,
            .src1_d = ids_from_sorted,
            .dst_d = dst->data,
            .dst_type = dst->type,
            .ne00 = dst->ne[0],
            .nb01 = dst->ne[0] * ts_dst_sorted,
            .nb02 = ne_get_rows * dst->ne[0] * ts_dst_sorted,
            .nb03 = ne_get_rows * dst->ne[0] * ts_dst_sorted,
            .ne10 = ne_get_rows,
            .ne11 = 1,
            .ne12 = 1,
            .nb10 = sizeof(int32_t),
            .nb11 = ne_get_rows * sizeof(int32_t),
            .nb12 = ne_get_rows * sizeof(int32_t),
            .nb1 = dst->nb[1],
            .nb2 = dst->nb[2],
            .nb3 = dst->nb[3]
        };
        get_rows_cuda(&ctx2);
    }

    void out_prod(cudaStream_t stream, cublasHandle_t handle, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(src0->ne[1] == src1->ne[1]);
        GGML_ASSERT(dst->ne[0] == src0->ne[0]);
        GGML_ASSERT(dst->ne[1] == src1->ne[0]);

        GGML_ASSERT(dst->ne[2] % src0->ne[2] == 0);
        GGML_ASSERT(dst->ne[3] % src0->ne[3] == 0);

        GGML_ASSERT(dst->ne[2] == src1->ne[2]);
        GGML_ASSERT(dst->ne[3] == src1->ne[3]);

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(handle, stream));

        const int64_t lda = src0->nb[1] / sizeof(float);
        const int64_t ldc = dst->nb[1] / sizeof(float);

        const bool src1_T = ggml_is_transposed(src1);
        const cublasOperation_t src1_cublas_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
        const int64_t           ldb = (src1_T ? src1->nb[0] : src1->nb[1]) / sizeof(float);
        GGML_ASSERT((src1_T ? src1->nb[1] : src1->nb[0]) == sizeof(float));

        // data strides in dimensions 2/3
        const size_t s02 = src0->nb[2] / sizeof(float);
        const size_t s03 = src0->nb[3] / sizeof(float);
        const size_t s12 = src1->nb[2] / sizeof(float);
        const size_t s13 = src1->nb[3] / sizeof(float);
        const size_t s2 = dst->nb[2] / sizeof(float);
        const size_t s3 = dst->nb[3] / sizeof(float);

        // dps == dst per src0, used for group query attention
        const int64_t dps2 = dst->ne[2] / src0->ne[2];
        const int64_t dps3 = dst->ne[3] / src0->ne[3];

        // TODO batched matrix multiplication
        for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
                CUBLAS_CHECK(
                    cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        dst->ne[0], dst->ne[1], src0->ne[1],
                        &alpha, src0_d + (i3 / dps3) * s03 + (i2 / dps2) * s02, lda,
                        src1_d + i3 * s13 + i2 * s12, ldb,
                        &beta, dst_d + i3 * s3 + i2 * s2, ldc));
            }
        }
    }

    void sqr(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));
        unary_context ctx = create(src0, dst, stream);
        sqr_cuda(&ctx);
    }

    void sqrt(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));
        unary_context ctx = create(src0, dst, stream);
        sqrt_cuda(&ctx);
    }

    void sin(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));
        unary_context ctx = create(src0, dst, stream);
        sin_cuda(&ctx);
    }

    void cos(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));
        unary_context ctx = create(src0, dst, stream);
        cos_cuda(&ctx);
    }

    void clamp(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        clamp_context ctx{
            .stream = stream,
			.src0_type = src0->type,
            .dst_type = dst->type,
            .src0_d = src0->data,
            .dst_d = dst->data,
            .nelements = src0->nelements(),
            .min = std::bit_cast<float>(dst->op_params[0]),
            .max = std::bit_cast<float>(dst->op_params[1])
        };
        clamp_cuda(&ctx);
    }

    void log(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(src0));
        unary_context ctx = create(src0, dst, stream);
        log_cuda(&ctx);
    }

    void diag_mask_inf(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int64_t ne00 = src0->ne[0];
        const int64_t ne01 = src0->ne[1];
        const int nrows0 = ggml_nrows(src0);

        const int n_past = std::bit_cast<int>(dst->op_params[0]);

        diag_mask_inf_f32_cuda(src0_d, dst_d, ne00, nrows0, ne01, n_past, stream);
    }

    void soft_max(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        const float* src0_d = (const float*)src0->data;
        const void* src1_d = src1 ? (const void*)src1->data : nullptr;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

        const int64_t nrows_x = ggml_nrows(src0);
        const int64_t nrows_y = src0->ne[1];

        const int64_t ne00 = src0->ne[0];

        float scale = std::bit_cast<float>(dst->op_params[0]);
        float max_bias = std::bit_cast<float>(dst->op_params[1]);

        const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

        const int64_t nb11 = src1 ? src1->nb[1] : 1;
        const int64_t nb12 = src1 ? src1->nb[2] : 1;
        const int64_t nb13 = src1 ? src1->nb[3] : 1;

        const int64_t ne12 = src1 ? src1->ne[2] : 1;
        const int64_t ne13 = src1 ? src1->ne[3] : 1;

        const uint32_t n_head = src0->ne[2];
        const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));

        const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

        softmax_context ctx{
            .src0_d = src0_d,
			.src1_d = src1_d,
            .dst_d = dst_d,
			.ne00 = ne00,
			.nrows_x = nrows_x,
			.nrows_y = nrows_y,
			.scale = scale,
			.max_bias = max_bias,
            .use_f16 = use_f16,
            .params = {
                .nheads = src0->ne[2],
                .n_head_log2 = n_head_log2,
                .ncols = ne00,
                .nrows_x = nrows_x,
                .nrows_y = nrows_y,
                .ne00 = src0->ne[0],
                .ne01 = src0->ne[1],
                .ne02 = src0->ne[2],
                .ne03 = src0->ne[3],
                .nb11 = nb11,
                .nb12 = nb12,
                .nb13 = nb13,
                .ne12 = ne12,
                .ne13 = ne13,
                .scale = scale,
                .max_bias = max_bias,
                .m0 = m0,
                .m1 = m1
            }
        };
        soft_max_f32_cuda(&ctx, stream);
    }

    void soft_max_back(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0]; // grad
        const ggml_tensor* src1 = dst->src[1]; // forward pass output

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int64_t ncols = src0->ne[0];
        const int64_t nrows = ggml_nrows(src0);

        float scale = std::bit_cast<float>(dst->op_params[0]);
        float max_bias = std::bit_cast<float>(dst->op_params[1]);

        GGML_ASSERT(max_bias == 0.0f);

        soft_max_back_f32_cuda(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
    }

    void rope(cudaStream_t stream, ggml_tensor* dst, bool forward) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const ggml_tensor* src2 = dst->src[2];

        GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        GGML_ASSERT(src0->type == dst->type);

        const int mode = std::bit_cast<int>(dst->op_params[2]);

        rope_context ctx{
            .forward = forward,
            .is_neox = static_cast<bool>(mode & GGML_ROPE_TYPE_NEOX),
            .is_mrope = static_cast<bool>(mode & GGML_ROPE_TYPE_MROPE),
            .is_vision = static_cast<bool>(mode == GGML_ROPE_TYPE_VISION),
            .src0_type = src0->type,
            .src0_d = src0->data,
            .dst_d = dst->data,
            .ne00 = src0->ne[0],
            .ne01 = src0->ne[1],
            .ne02 = src0->ne[2],
            .s01 = src0->nb[1] / ggml_type_size(src0->type),
            .s02 = src0->nb[2] / ggml_type_size(src0->type),
            .n_dims = std::bit_cast<int>(dst->op_params[1]),
            .n_ctx_orig = std::bit_cast<int>(dst->op_params[4]),
            .nr = ggml_nrows(src0),
            .pos = (const int32_t*)src1->data,
            // RoPE alteration for extended context
            .freq_base = std::bit_cast<float>(dst->op_params[5]),
            .freq_scale = std::bit_cast<float>(dst->op_params[6]),
            .ext_factor = std::bit_cast<float>(dst->op_params[7]),
            .attn_factor = std::bit_cast<float>(dst->op_params[8]),
            .beta_fast = std::bit_cast<float>(dst->op_params[9]),
            .beta_slow = std::bit_cast<float>(dst->op_params[10]),
            .freq_factors = (src2 != nullptr) ? (const float*)src2->data : nullptr
        };
        memcpy(&ctx.sections.v, (int32_t*)dst->op_params + 11, sizeof(int) * 4);

        if (ctx.is_mrope) {
            GGML_ASSERT(ctx.sections.v[0] > 0 || ctx.sections.v[1] > 0 || ctx.sections.v[2] > 0);
        }
        if (ctx.is_vision) {
            GGML_ASSERT(ctx.n_dims == ctx.ne00 / 2);
        }

        rope_cuda(&ctx, stream);
    }

    void concat(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        concat_context ctx{
            .dim = dst->op_params[0],
            .src0_is_contiguous = ggml_is_contiguous(src0),
			.src1_is_contiguous = ggml_is_contiguous(src1),
            .src0_d = (const float*)src0->data,
			.src1_d = (const float*)src1->data,
			.dst_d = (float*)dst->data,
            .ne00 = src0->ne[0],
			.ne01 = src0->ne[1],
			.ne02 = src0->ne[2],
            .ne03 = src0->ne[3],
            .nb00 = src0->nb[0],
            .nb01 = src0->nb[1],
            .nb02 = src0->nb[2],
            .nb03 = src0->nb[3],
            .ne10 = src1->ne[0],
            .ne11 = src1->ne[1],
            .ne12 = src1->ne[2],
            .ne13 = src1->ne[3],
            .nb10 = src1->nb[0],
            .nb11 = src1->nb[1],
            .nb12 = src1->nb[2],
            .nb13 = src1->nb[3],
            .ne0 = dst->ne[0],
            .ne1 = dst->ne[1],
			.ne2 = dst->ne[2],
            .ne3 = dst->ne[3],
            .nb0 = dst->nb[0],
            .nb1 = dst->nb[1],
            .nb2 = dst->nb[2],
            .nb3 = dst->nb[3],
			.src0_size = src0->nbytes(),
			.src1_size = src1->nbytes()
        };

		concat_cuda(&ctx, stream);
    }

    void argsort(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src0));

        const int64_t ncols = src0->ne[0];
        const int64_t nrows = ggml_nrows(src0);

        ggml_sort_order order = std::bit_cast<ggml_sort_order>(dst->op_params[0]);

        argsort_f32_i32_cuda(src0_d, (int*)dst_d, ncols, nrows, order, stream);
    }

    void sum(ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguously_allocated(src0));

        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        const int64_t ne = src0->nelements();
        sum_f32_cuda(pool, src0_d, dst_d, ne, stream);
    }

    void sum_rows(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src0));

        const int64_t ncols = src0->ne[0];
        const int64_t nrows = ggml_nrows(src0);

        sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
    }

    void upscale(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int mode_flags = dst->op_params[0];
        const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

        float sf0 = (float)dst->ne[0] / src0->ne[0];
        float sf1 = (float)dst->ne[1] / src0->ne[1];
        float sf2 = (float)dst->ne[2] / src0->ne[2];
        const float sf3 = (float)dst->ne[3] / src0->ne[3];

        if (mode == GGML_SCALE_MODE_NEAREST) {
            upscale_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
        }
        else if (mode == GGML_SCALE_MODE_BILINEAR) {
            float pixel_offset = 0.5f;
            if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
                sf0 = (float)(dst->ne[0] - 1) / (src0->ne[0] - 1);
                sf1 = (float)(dst->ne[1] - 1) / (src0->ne[1] - 1);
                pixel_offset = 0.0f;
            }
            upscale_f32_bilinear_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                sf0, sf1, sf2, sf3, pixel_offset, stream);
        }
    }

    void group_norm(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int num_groups = std::bit_cast<int>(dst->op_params[0]);

        const float eps = std::bit_cast<float>(dst->op_params[1]);
        GGML_ASSERT(eps >= 0.0f);

        int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
        group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, src0->nelements(), stream);
    }

    void acc(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
        GGML_ASSERT(ggml_is_contiguously_allocated(dst));

        const int64_t s1 = dst->op_params[0] / sizeof(float);
        const int64_t s2 = dst->op_params[1] / sizeof(float);
        const int64_t s3 = dst->op_params[2] / sizeof(float);
        const int64_t offset = dst->op_params[3] / sizeof(float);

        acc_f32_cuda(src0_d, src1_d, dst_d, dst->nelements(), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], s1, s2, s3, offset, stream);
    }

    void pad(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

        pad_f32_cuda(src0_d, dst_d,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
    }

    void timestep_embedding(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int dim = std::bit_cast<int>(dst->op_params[0]);
        const int max_period = std::bit_cast<int>(dst->op_params[1]);

        timestep_embedding_f32_cuda(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
    }

    void leaky_relu(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const void* src0_d = src0->data;
        void* dst_d = dst->data;

        GGML_ASSERT(ggml_is_contiguous(src0));

        GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        GGML_ASSERT(src0->type == dst->type);

        const float negative_slope = std::bit_cast<float>(dst->op_params[0]);
        leaky_relu_cuda(src0->type == GGML_TYPE_F16, src0_d, dst_d, src0->nelements(), negative_slope, stream);
    }

    void flash_attn_ext(int device, ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* Q = dst->src[0];
        const ggml_tensor* K = dst->src[1];
        const ggml_tensor* V = dst->src[2];
        const ggml_tensor* mask = dst->src[3];

        ggml_cuda_set_device(device);
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
		const ggml_prec prec = std::bit_cast<ggml_prec>(dst->op_params[3]);

        flash_attn_ext_context ctx{
            .device = device,
            .main_stream = stream,
            .pool = &pool,
            .scale = std::bit_cast<float>(dst->op_params[0]),
            .max_bias = std::bit_cast<float>(dst->op_params[1]),
            .logit_softcap = std::bit_cast<float>(dst->op_params[2]),
            .precision = std::bit_cast<ggml_prec>(dst->op_params[3]),
            .Q = {
                .type = Q->type,
                .data = Q->data,
                .ne0 = Q->ne[0],
                .ne1 = Q->ne[1],
                .ne2 = Q->ne[2],
                .ne3 = Q->ne[3],
                .nb0 = Q->nb[0],
                .nb1 = Q->nb[1],
                .nb2 = Q->nb[2],
                .nb3 = Q->nb[3],
                .element_size = ggml_element_size(Q)
            },
            .K = {
				.type = K->type,
                .block_size = ggml_blck_size(K->type),
                .type_size = ggml_type_size(K->type),
                .data = K->data,
				.elements = K->nelements(),
                .ne0 = K->ne[0],
                .ne1 = K->ne[1],
                .ne2 = K->ne[2],
                .ne3 = K->ne[3],
				.nb0 = K->nb[0],
				.nb1 = K->nb[1],
				.nb2 = K->nb[2],
				.nb3 = K->nb[3],
                .bs = ggml_blck_size(K->type),
                .ts = ggml_type_size(K->type),
                .contiguously_allocated = ggml_is_contiguously_allocated(K),
                .element_size = ggml_element_size(K)
            },
            .V = {
                .exist = V != nullptr,
                .type = V ? V->type : GGML_TYPE_F32,
                .block_size = V ? ggml_blck_size(V->type) : 0,
                .type_size = V ? ggml_type_size(V->type) : 0,
                .data = V ? V->data : nullptr,
                .elements = V ? V->nelements() : 0,
                .ne0 = V ? V->ne[0] : 0,
                .ne1 = V ? V->ne[1] : 0,
                .ne2 = V ? V->ne[2] : 0,
                .ne3 = V ? V->ne[3] : 0,
                .nb0 = V ? V->nb[0] : 0,
                .nb1 = V ? V->nb[1] : 0,
                .nb2 = V ? V->nb[2] : 0,
                .nb3 = V ? V->nb[3] : 00,
                .bs = V ? ggml_blck_size(V->type) : 0,
                .ts = V ? ggml_type_size(V->type) : 0,
                .contiguously_allocated = V ? ggml_is_contiguously_allocated(V) : false,
                .element_size = V ? ggml_element_size(V) : 0
            },
            .mask = {
				.exist = mask != nullptr,
                .type = mask ? mask->type : GGML_TYPE_F32,
                .data = mask ? mask->data : nullptr,
                .ne0 = (mask) ? mask->ne[0] : 0,
                .ne1 = (mask) ? mask->ne[1] : 0,
                .ne2 = (mask) ? mask->ne[2] : 0,
                .ne3 = (mask) ? mask->ne[3] : 0,
                .nb0 = (mask) ? mask->nb[0] : 0,
                .nb1 = (mask) ? mask->nb[1] : 0,
                .nb2 = (mask) ? mask->nb[2] : 0,
                .nb3 = (mask) ? mask->nb[3] : 0
            },
            .KQV = {
                .type = dst->type,
                .data = dst->data,
                .elements = dst->nelements(),
                .nrows = ggml_nrows(dst),
                .ne0 = dst->ne[0],
                .ne1 = dst->ne[1],
                .ne2 = dst->ne[2],
                .ne3 = dst->ne[3]
            }
        };

        if (cc >= GGML_CUDA_CC_OFFSET_AMD) {
#if defined(GGML_HIP_ROCWMMA_FATTN)
            if (fp16_mma_available(cc)) {
                ggml_cuda_flash_attn_ext_wmma_f16(ctx);
                return;
            }
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

            // On AMD the tile kernels perform poorly, use the vec kernel instead:
            if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx);
            }
            else {
                ggml_cuda_flash_attn_ext_vec_f32(ctx);
            }
            return;
        }

        if (!fast_fp16_available(cc)) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx);
            }
            else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx);
            }
            return;
        }
        if (!fp16_mma_available(cc)) {
            if (prec == GGML_PREC_DEFAULT) {
                if (Q->ne[1] <= 8) {
                    ggml_cuda_flash_attn_ext_vec_f16(ctx);
                }
                else {
                    ggml_cuda_flash_attn_ext_tile_f16(ctx);
                }
            }
            else {
                if (Q->ne[1] <= 8) {
                    ggml_cuda_flash_attn_ext_vec_f32(ctx);
                }
                else {
                    ggml_cuda_flash_attn_ext_tile_f32(ctx);
                }
            }
            return;
        }
        const int gqa_ratio = Q->ne[2] / K->ne[2];
        const bool mma_fast_for_bs1 = fp16_mma_available(cc) && gqa_ratio % 2 == 0 &&
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && mask;
        if (Q->ne[1] == 1 && Q->ne[0] % (2 * warp_size) == 0 && !mma_fast_for_bs1) {
            if (prec == GGML_PREC_DEFAULT) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx);
                return;
            }
            else if (Q->ne[0] <= 128) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx);
                return;
            }
        }

        // The MMA implementation needs Turing or newer, use the old WMMA code for Volta:
        if (fp16_mma_available(cc) && !new_mma_available(cc)) {
            ggml_cuda_flash_attn_ext_wmma_f16(ctx);
            return;
        }
        ggml_cuda_flash_attn_ext_mma_f16(ctx);
    }

    void cross_entropy_loss(ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT(ggml_is_contiguous(dst));

        const int id = ggml_cuda_get_device();
        cross_entropy_context ctx{
            .id = id,
            .smpbo = ggml_cuda_info().devices[id].smpbo,
            .pool = pool,
            .nrows = ggml_nrows(src0),
            .ne00 = src0->ne[0],
            .src0_d = (const float*)src0->data,
            .src1_d = (const float*)src1->data,
            .dst_d = (float*)dst->data
        };

        cross_entropy_loss_cuda(&ctx, stream);
    }

    void cross_entropy_loss_back(ggml_cuda_pool& pool, cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* grad = dst->src[0];
        const ggml_tensor* src0f = dst->src[1];
        const ggml_tensor* src1f = dst->src[2];

        GGML_ASSERT(src0f->type == GGML_TYPE_F32);
        GGML_ASSERT(src1f->type == GGML_TYPE_F32);
        GGML_ASSERT(grad->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_is_scalar(grad));
        GGML_ASSERT(ggml_is_contiguous(src0f));
        GGML_ASSERT(ggml_is_contiguous(src1f));
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
        GGML_ASSERT(ggml_are_same_shape(src0f, dst));

        const int id = ggml_cuda_get_device();

        cross_entropy_back_context ctx{
            .id = id,
            .smpbo = ggml_cuda_info().devices[id].smpbo,
            .pool = pool,
            .nrows = ggml_nrows(src0f),
            .ne00 = src0f->ne[0],
            .grad_d = (const float*)grad->data,
            .src0f_d = (const float*)src0f->data,
            .src1f_d = (const float*)src1f->data,
            .dst_d = (float*)dst->data
        };

        cross_entropy_loss_back_cuda(&ctx, stream);
    }

    void opt_step_adamw(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src0_grad = dst->src[1];
        const ggml_tensor* src0_grad_m = dst->src[2];
        const ggml_tensor* src0_grad_v = dst->src[3];
        const ggml_tensor* adamw_params = dst->src[4];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
        GGML_ASSERT(src0_grad_m->type == GGML_TYPE_F32);
        GGML_ASSERT(src0_grad_v->type == GGML_TYPE_F32);
        GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ggml_is_contiguous(src0_grad));
        GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
        GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
        GGML_ASSERT(ggml_is_contiguous(adamw_params));
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
        GGML_ASSERT(adamw_params->nelements() == 7);

        float* src0_d = (float*)src0->data;
        const float* src0_grad_d = (const float*)src0_grad->data;
        float* src0_grad_m_d = (float*)src0_grad_m->data;
        float* src0_grad_v_d = (float*)src0_grad_v->data;
        const float* adamw_params_d = (const float*)adamw_params->data;

        const int64_t ne = src0->nelements();

        opt_step_adamw_f32_cuda(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, stream);
    }

    void repeat_back(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];

        GGML_ASSERT(src0->type == dst->type);
        GGML_ASSERT(ggml_is_contiguous(dst));
        GGML_ASSERT(ggml_can_repeat(dst, src0));

        GGML_ASSERT(dst->ne[2] * dst->ne[3] <= (1 << 15));

        repeat_back_context ctx{
            .dst_type = dst->type,
            .src0_d = src0->data,
            .dst_d = dst->data,
            .src0_ts = ggml_type_size(src0->type),
            .ne00 = src0->ne[0],
            .ne01 = src0->ne[1],
            .ne02 = src0->ne[2],
            .ne03 = src0->ne[3],
            .nb00 = src0->nb[0],
            .nb01 = src0->nb[1],
            .nb02 = src0->nb[2],
            .nb03 = src0->nb[3],
            .ne0 = dst->ne[0],
            .ne1 = dst->ne[1],
            .ne2 = dst->ne[2],
            .ne3 = dst->ne[3]
        };

        repeat_back_cuda(&ctx, stream);
    }

    void l2_norm(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        float eps = std::bit_cast<float>(dst->op_params[0]);
        GGML_ASSERT(eps >= 0.0f);

        const size_t ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        const int64_t s01 = src0->nb[1] / ts0;
        const int64_t s02 = src0->nb[2] / ts0;
        const int64_t s03 = src0->nb[3] / ts0;

        l2_norm_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1],
            src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }

    void ssm_conv(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];  // conv_x
        const ggml_tensor* src1 = dst->src[1];  // conv1d.weight

        const int64_t nc = src1->ne[0];                // d_conv
        const int64_t nr = src0->ne[1];                // d_inner
        const int64_t n_t = dst->ne[1];                 // tokens per sequence
        const int64_t n_s = dst->ne[2];                 // number of sequences in the batch

        GGML_ASSERT(dst->ne[0] == nr);
        GGML_ASSERT(src0->nb[0] == sizeof(float));
        GGML_ASSERT(src1->nb[0] == sizeof(float));
        GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        ssm_conv_f32_cuda(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, dst->nb[0], dst->nb[1],
            dst->nb[2], nc, nr, n_t, n_s, stream);
    }

    void ssm_scan(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];  // s
        const ggml_tensor* src1 = dst->src[1];  // x
        const ggml_tensor* src2 = dst->src[2];  // dt
        const ggml_tensor* src3 = dst->src[3];  // A
        const ggml_tensor* src4 = dst->src[4];  // B
        const ggml_tensor* src5 = dst->src[5];  // C
        const ggml_tensor* src6 = dst->src[6];  // ids

        const int64_t nc = src0->ne[0];  // d_state
        const int64_t nr = src0->ne[1];  // head_dim or 1
        const int64_t nh = src1->ne[1];  // n_head
        const int64_t ng = src4->ne[1];  // n_group
        const int64_t n_t = src1->ne[2];  // number of tokens per sequence
        const int64_t n_s = src1->ne[3];  // number of sequences in the batch

        const int64_t s_off = src1->nelements() * sizeof(float);

        GGML_ASSERT(src1->nelements() + nc * nr * nh * n_s == dst->nelements());
        GGML_ASSERT(src0->nb[0] == sizeof(float));
        GGML_ASSERT(src1->nb[0] == sizeof(float));
        GGML_ASSERT(src2->nb[0] == sizeof(float));
        GGML_ASSERT(src3->nb[0] == sizeof(float));
        GGML_ASSERT(src4->nb[0] == sizeof(float));
        GGML_ASSERT(src5->nb[0] == sizeof(float));
        GGML_ASSERT(src6->nb[0] == sizeof(int32_t));

        const float* src0_d = (const float*)src0->data;
        const float* src1_d = (const float*)src1->data;
        const float* src2_d = (const float*)src2->data;
        const float* src3_d = (const float*)src3->data;
        const float* src4_d = (const float*)src4->data;
        const float* src5_d = (const float*)src5->data;
        const int32_t* src6_d = (const int32_t*)src6->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src6->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
            src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
            s_off, nc, nr, nh, ng, n_t, n_s, stream);
    }

    void conv2d_dw(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* kernel = dst->src[0];
        const ggml_tensor* input = dst->src[1];

        GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

        conv2d_dw_context ctx{
			.input_is_contiguous = ggml_is_contiguous(input),
            .input_is_contiguous_channels = ggml_is_contiguous_channels(input),
            .in_w = static_cast<int>(input->ne[0]),
            .in_h = static_cast<int>(input->ne[1]),
            .kernel_w = static_cast<int>(kernel->ne[0]),
            .kernel_h = static_cast<int>(kernel->ne[1]),
            .out_w = static_cast<int>(dst->ne[0]),
			.out_h = static_cast<int>(dst->ne[1]),
            .channels = static_cast<int>(dst->ne[2]),
            .batches = static_cast<int>(dst->ne[3]),
            .x_d = (const float*)input->data,
            .y_d = (float*)dst->data,
            .w_d = (const float*)kernel->data,
            .stride_x = dst->op_params[0],
            .stride_y = dst->op_params[1],
            .padding_x = dst->op_params[2],
			.padding_y = dst->op_params[3],
            .dilation_x = dst->op_params[4],
			.dilation_y = dst->op_params[5]
        };
        conv2d_dw_cuda(&ctx, stream);
    }

    //input is (W, H, C_in, N), Kernel is (W, H, C_out, C_in)
    void conv_2d_transpose_p0(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* kernel = dst->src[0];
        const ggml_tensor* input = dst->src[1];

        GGML_ASSERT(kernel->type == GGML_TYPE_F16 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

        const int channels_in = input->ne[2];
        const int stride = dst->op_params[0];
        const int batches = input->ne[3];

        GGML_ASSERT(channels_in == kernel->ne[3]);
        GGML_ASSERT(stride > 0);

        GGML_ASSERT(ggml_is_contiguous(input));
        GGML_ASSERT(ggml_is_contiguous(kernel));
        GGML_ASSERT(ggml_is_contiguous(dst));

        conv2d_transpose_context ctx{
            .input_w = static_cast<int>(input->ne[0]),
            .input_h = static_cast<int>(input->ne[1]),
            .output_w = static_cast<int>(dst->ne[0]),
            .output_h = static_cast<int>(dst->ne[1]),
            .channels_in = channels_in,
            .channels_out = static_cast<int>(kernel->ne[2]),
            .kernel_w = static_cast<int>(kernel->ne[0]),
            .kernel_h = static_cast<int>(kernel->ne[1]),
            .stride = stride,
            .batches = batches,
            .input_data = (const float*)input->data,
            .output_data = (float*)dst->data,
            .kernel_data = (const half*)kernel->data,
        };

        conv_2d_transpose_p0_cuda(&ctx, stream);
    }

    void glu(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];
        const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
        GGML_ASSERT(ggml_is_contiguous_1(src0));
        GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
        GGML_ASSERT(ggml_is_contiguous(dst));

        GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        GGML_ASSERT(src0->type == dst->type);
        GGML_ASSERT(dst->ne[0] == nc);
        GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

        if (src1) {
            GGML_ASSERT(ggml_is_contiguous_1(src1));
            GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
            GGML_ASSERT(src1->ne[0] == nc);
            GGML_ASSERT(src0->type == src1->type);
        }

        gated_context ctx{
            .stream = stream,
            .src0_type = src0->type,
            .swapped = dst->op_params[1],
			.src0_d = src0->data,
            .src1_d = src1 ? src1->data : src0->data,
			.dst_d = dst->data,
            .src0_o = src0->nb[1],
            .src1_o = src1 ? src1->nb[1] : src0->nb[1],
            .nc = nc,
			.dst_nelements = dst->nelements(),
            .src1_exist = src1 != nullptr
        };

        switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
            reglu_cuda(&ctx);
            break;
        case GGML_GLU_OP_GEGLU:
            geglu_cuda(&ctx);
            break;
        case GGML_GLU_OP_SWIGLU:
            swiglu_cuda(&ctx);
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            geglu_erf_cuda(&ctx);
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            geglu_quick_cuda(&ctx);
            break;
        default:
            std::unreachable();
        }
    }

    void mean(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src0));
        mean_cuda((const float*)src0->data, (float*)dst->data, src0->ne[0], ggml_nrows(src0), stream);
    }

    void set_rows(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_I64);

        set_rows_context ctx{
            .dst_type = dst->type,
            .src0_d = (const float*)src0->data,
            .src1_d = (const int64_t*)src1->data,
            .dst_d = dst->data,
            .ne00 = src0->ne[0],
            .ne01 = src0->ne[1],
            .ne02 = src0->ne[2],
            .ne03 = src0->ne[3],
            .nb00 = src0->nb[0],
            .nb01 = src0->nb[1],
            .nb02 = src0->nb[2],
            .nb03 = src0->nb[3],
            .ne10 = src1->ne[0],
            .ne11 = src1->ne[1],
            .ne12 = src1->ne[2],
            .ne13 = src1->ne[3],
            .nb10 = src1->nb[0],
            .nb11 = src1->nb[1],
            .nb12 = src1->nb[2],
            .nb13 = src1->nb[3],
            .nb1 = dst->nb[1],
            .nb2 = dst->nb[2],
            .nb3 = dst->nb[3]
        };
        set_rows_cuda(&ctx, stream);
    }
}
