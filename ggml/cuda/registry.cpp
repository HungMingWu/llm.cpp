module;
#include <bit>
#define WARP_SIZE 32

module ggml;
import :cuda.op;

static int register_ok = []() {
    get_reg().register_backend(ggml_backend_cuda_reg());
    return 0;
}();

bool ggml_backend_cuda_device::supports_op(const ggml_tensor* op)
{
    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (auto& src : op->src) {
            if (!src) continue;
            if (!src->buffer) continue;
            if (to_split_buffer_type(src->buffer->get_type())) {
                return false;
            }
        }
    }

    // check if all the sources are allocated on this device
    for (auto& src : op->src) {
        if (!src) continue;
        if (!src->buffer) continue;
        if (auto cuda_buffer_type = to_cuda_buffer_type(src->buffer->get_type())) {
            if (cuda_buffer_type->device != device) {
                return false;
            }
        }
    }

    switch (op->op) {
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SGN:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_ELU:
            return ggml_is_contiguous(op->src[0]);
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(op)) {
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            return ggml_is_contiguous_1(op->src[0]);
        default:
            return false;
        }
        break;
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
    {
        const ggml_tensor* a = op->src[0];
        const ggml_tensor* b = op->src[1];
        // for small weight matrices the active device can end up without any rows, don't use row split in those cases
        // this avoids some edge cases (and the performance would not be good anyways)
        cuda_split_backend_buffer_type* split_bufer_type = (a->buffer) ? to_split_buffer_type(a->buffer->get_type()) : nullptr;
        if (split_bufer_type) {
            if (a->ne[2] > 1 || a->ne[3] > 1) {
                return false;
            }
            // for small weight matrices the active device can end up without any rows, don't use row split in those cases
            // this avoids some edge cases (and the performance would not be good anyways)
            auto [row_low, row_high] = get_row_split(ggml_nrows(a), split_bufer_type->tensor_split, device);
            if (row_low == row_high) {
                return false;
            }
        }
        if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
            return false;
        }
#ifdef GGML_USE_MUSA
        const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
        if (b->ne[2] * b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
            if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                return false;
            }
            if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                return false;
            }
        }
#endif // GGML_USE_MUSA
        switch (a->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
        }
    } break;
    case GGML_OP_OUT_PROD:
        return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
    case GGML_OP_GET_ROWS:
    {
        switch (op->src[0]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
        }
    } break;
    case GGML_OP_GET_ROWS_BACK:
    {
        return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
    } break;
    case GGML_OP_SET_ROWS:
    {
        return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
            op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
            op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
            op->src[0]->type == GGML_TYPE_F32 &&
            (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
    } break;
    case GGML_OP_CPY:
    {
        ggml_type src0_type = op->src[0]->type;
        ggml_type src1_type = op->src[1]->type;
        if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
            (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
            ) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
            return true;
        }
        if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
            return true;
        }
        if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
            return true;
        }
        if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
            return true;
        }
        if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
            return true;
        }
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
            return true;
        }
        if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
            return true;
        }
        return false;
    } break;
    case GGML_OP_DUP:
    {
        ggml_type src0_type = op->src[0]->type;
        return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
    } break;
    case GGML_OP_ARGMAX:
    case GGML_OP_COUNT_EQUAL:
    {
        return true;
    } break;
    case GGML_OP_REPEAT:
    {
        ggml_type src0_type = op->src[0]->type;
        return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
    } break;
    case GGML_OP_REPEAT_BACK:
        return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2] * op->src[0]->ne[3]) <= (1 << 15);
    case GGML_OP_CONCAT:
    {
        ggml_type src0_type = op->src[0]->type;
        return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
    } break;
    case GGML_OP_CONV_TRANSPOSE_1D:
    {
        ggml_type src0_type = op->src[0]->type;
        ggml_type src1_type = op->src[1]->type;
        if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16) && src1_type == GGML_TYPE_F32) {
            return true;
        }
        return false;
    } break;
    case GGML_OP_SILU_BACK:
        return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_L2_NORM:
        return true;
    case GGML_OP_RMS_NORM_BACK:
        return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_ADD:
    case GGML_OP_ADD_ID:
    case GGML_OP_ADD1:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_CLAMP:
    case GGML_OP_LOG:
        return true;
    case GGML_OP_SSM_SCAN: {
        if (op->src[3]->ne[0] == 1) {
            // Mamba2
            // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
            return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
        }
        else {
            // Mamba
            // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
            return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
        }
    }
    case GGML_OP_SSM_CONV: {
        // assumes d_inner % threads == 0
        return op->src[0]->ne[1] % 128 == 0;
    }
    case GGML_OP_CONT:
        return true;
    case GGML_OP_DIAG_MASK_INF:
        return true;
    case GGML_OP_SOFT_MAX:
        return true;
    case GGML_OP_SOFT_MAX_BACK:
        return std::bit_cast<float>(op->op_params[1]) == 0.0f;
    case GGML_OP_ROLL:
        if (op->src[0]->type == GGML_TYPE_F32) {
            return true;
        }
        return false;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK: {
        return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
    }
    case GGML_OP_IM2COL:
    case GGML_OP_IM2COL_3D:
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_CONV_TRANSPOSE_2D:
    case GGML_OP_POOL_2D:
    case GGML_OP_ACC:
        return true;
    case GGML_OP_SUM:
        return ggml_is_contiguous_rows(op->src[0]);
    case GGML_OP_ARGSORT:
        // TODO: Support arbitrary column width
        return op->src[0]->ne[0] <= 1024;
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_PAD:
        return ggml_is_contiguous(op->src[0]);
    case GGML_OP_UPSCALE:
    case GGML_OP_PAD_REFLECT_1D:
    case GGML_OP_ARANGE:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_LEAKY_RELU:
    case GGML_OP_RWKV_WKV6:
    case GGML_OP_GATED_LINEAR_ATTN:
    case GGML_OP_RWKV_WKV7:
        return true;
    case GGML_OP_FLASH_ATTN_EXT:
        return op::ggml_cuda_flash_attn_ext_supported(device, op);
    case GGML_OP_CROSS_ENTROPY_LOSS:
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
    case GGML_OP_OPT_STEP_ADAMW:
    case GGML_OP_OPT_STEP_SGD:
        return true;
    default:
        return false;
    }
}