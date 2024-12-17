module;
#include <assert.h>
#include <bit>
#include "common.h"
#include "cu/cuda_func.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml:cuda.op;
import :ds;
import :tensor;

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

    void mul_mat_vec(cudaStream_t stream, ggml_tensor* dst)
    {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F16);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int64_t ne00 = src0->ne[0];
        const int64_t ne01 = src0->ne[1];

        GGML_ASSERT(src1->ne[1] == 1);

        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

        const half* src0_d = (const half*)src0->data;
        const float* src1_d = (const float*)src1->data;
        float* dst_d = (float*)dst->data;

        const int64_t ne02 = src0->ne[2];
        const int64_t ne12 = src1->ne[2];
        GGML_ASSERT(dst->ne[2] == ne12);

        GGML_ASSERT(src0->ne[3] == 1);
        GGML_ASSERT(src1->ne[3] == 1);
        GGML_ASSERT(dst->ne[3] == 1);

        const int64_t stride_row = src0->nb[1] / ggml_type_size(src0->type);
        const int64_t channel_stride_x = src0->nb[2] / ggml_type_size(src0->type);
        const int64_t channel_stride_y = src1->nb[2] / ggml_type_size(src1->type);
        const int64_t channel_stride_dst = dst->nb[2] / ggml_type_size(dst->type);

        mul_mat_vec_cuda(src0_d, src1_d, dst_d, ne00, ne01, stride_row, ne02, ne12, channel_stride_x, channel_stride_y, channel_stride_dst, prec, stream);
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
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(ggml_is_contiguous(src0));

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        const int64_t nelements = src0->nelements();
        switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_ABS:
            return abs_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_SGN:
            return sgn_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_ELU:
            return elu_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_NEG:
            return neg_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_STEP:
            return step_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_GELU:
            return gelu_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_SILU:
            return silu_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_GELU_QUICK:
            return gelu_quick_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_TANH:
            return tanh_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_RELU:
            return relu_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_SIGMOID:
            return sigmoid_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_HARDSIGMOID:
            return hardsigmoid_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_HARDSWISH:
            return hardswish_f32_cuda(src0_d, dst_d, nelements, stream);
        case GGML_UNARY_OP_EXP:
            return exp_f32_cuda(src0_d, dst_d, nelements, stream);
        default:
            assert(false);
        }
    }

    void get_rows(cudaStream_t stream, ggml_tensor* dst) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
        GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
        GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

        get_row_context context{
            .type = src0->type,
            .src0_d = src0->data,
            .src1_d = (const int32_t*)src1->data,
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
            .s1 = dst->nb[1] / ggml_element_size(dst),
            .s2 = dst->nb[2] / ggml_element_size(dst),
            .s3 = dst->nb[3] / ggml_element_size(dst),
            .s10 = src1->nb[0] / ggml_element_size(src1),
            .s11 = src1->nb[1] / ggml_element_size(src1),
            .s12 = src1->nb[2] / ggml_element_size(src1)
        };

        get_rows_cuda(&context, stream);
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

        float scale = std::bit_cast<float>(dst->op_params[0]);

        scale_f32_cuda(src0_d, dst_d, scale, src0->nelements(), stream);
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

        rwkv_wkv6_cuda(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d, stream);
    }

    void gated_linear_attn(cudaStream_t stream, ggml_tensor* dst) {
        const float* k_d = (const float*)dst->src[0]->data;
        const float* v_d = (const float*)dst->src[1]->data;
        const float* r_d = (const float*)dst->src[2]->data;
        const float* td_d = (const float*)dst->src[3]->data;
        const float* s_d = (const float*)dst->src[4]->data;

        const int64_t B = dst->src[4]->ne[1];
        const int64_t T = dst->src[0]->ne[2];
        const int64_t C = dst->ne[0];
        const int64_t H = dst->src[0]->ne[1];

        float scale = std::bit_cast<float>(dst->op_params[0]);

        float* dst_d = (float*)dst->data;

        GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
        GGML_ASSERT(C % H == 0);
        GGML_ASSERT(C / H == 64 || C / H == 128);

        rwkv_wkv6_cuda(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d, stream);
    }
}