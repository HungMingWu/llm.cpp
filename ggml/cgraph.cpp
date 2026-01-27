module;
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <format>
#include <fstream>
#include <ranges>
#include <unordered_map>

module ggml;
import :ds;
import :log;
import :tensor;
import :traits;

#define GGML_ABORT(...)
#define GGML_ASSERT(...)
#define GGML_PRINT_DEBUG(...)

static ggml_tensor* ggml_acc_impl(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    size_t                nb1,
    size_t                nb2,
    size_t                nb3,
    size_t                offset,
    bool                  inplace) {
    GGML_ASSERT(b->nelements() <= a->nelements());
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { (int32_t)nb1, (int32_t)nb2, (int32_t)nb3, (int32_t)offset, inplace ? 1 : 0 };
    ggml_set_op_params(*result, params, sizeof(params));

    result->op = GGML_OP_ACC;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

static void ggml_acc_or_set(
    struct ggml_context* ctx,
    struct ggml_cgraph* cgraph,
    ggml_tensor* src,
    struct ggml_tensor* tensor,
    const  size_t         nb1,
    const  size_t         nb2,
    const  size_t         nb3,
    const  size_t         offset) {
    auto& grads = cgraph->grads[src];
    GGML_ASSERT(src);
    if (grads) {
        grads = ggml_acc_impl(ctx, grads, tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[src]);
    }
    else {
        struct ggml_tensor* a_zero = ggml_scale(ctx, src, 0.0f, false); // FIXME this is going to produce NaN if a contains inf/NaN
        grads = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    grads->set_name("grad for {}", src->name);
    cgraph->build_forward_expand(grads);
}

static ggml_tensor* ggml_add_impl(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_ADD;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

static void ggml_add_or_set(
    ggml_context* ctx,
    ggml_cgraph* cgraph,
    ggml_tensor* src,
    ggml_tensor* tensor) {
    GGML_ASSERT(src);
    auto& grads = cgraph->grads[src];
    if (grads) {
        grads = ggml_add_impl(ctx, grads, tensor, /*inplace =*/ cgraph->grad_accs[src]);
    }
    else {
        grads = tensor;
    }
    grads->set_name("grad for {}", src->name);
    cgraph->build_forward_expand(grads);
}

static ggml_tensor* ggml_diag_mask_zero_impl(
    ggml_context* ctx,
    ggml_tensor* a,
    int                   n_past,
    bool                  inplace) {
    ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(*result, params, sizeof(params));

    result->op = GGML_OP_DIAG_MASK_ZERO;
    result->src.push_back(a);

    return result;
}

ggml_tensor* ggml_pool_2d_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* af,
    enum ggml_op_pool     op,
    int                   k0,
    int                   k1,
    int                   s0,
    int                   s1,
    float                 p0,
    float                 p1) {
    ggml_tensor* result = ctx->create(GGML_TYPE_F32, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, (int32_t)p0, (int32_t)p1 };
    ggml_set_op_params(*result, params, sizeof(params));

    result->op = GGML_OP_POOL_2D_BACK;
    result->src.push_back(a);
    result->src.push_back(af);

    return result;
}

static ggml_tensor* ggml_sub_impl(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_SUB;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

static void ggml_sub_or_set(
    ggml_context* ctx,
    ggml_cgraph* cgraph,
    ggml_tensor* src,
    ggml_tensor* tensor) {
    GGML_ASSERT(src);
    auto& grads = cgraph->grads[src];
    if (grads) {
        grads = ggml_sub_impl(ctx, grads, tensor, cgraph->grad_accs[src]);
    }
    else {
        grads = ggml_neg(ctx, tensor);
    }
    grads->set_name("grad for {}", src->name);
    cgraph->build_forward_expand(grads);
}

static void ggml_add1_or_set(
    ggml_context* ctx,
    ggml_cgraph* cgraph,
    ggml_tensor* src,
    ggml_tensor* tensor) {
    GGML_ASSERT(src);
    auto& grads = cgraph->grads[src];
    if (grads) {
        grads = ggml_add1(ctx, grads, tensor, cgraph->grad_accs[src]);
    }
    else {
        grads = ggml_repeat(ctx, tensor, src);
    }
    grads->set_name("grad for {}", src->name);
    cgraph->build_forward_expand(cgraph->grads[src]);
}

static ggml_tensor* ggml_scale_impl(
    ggml_context* ctx,
    ggml_tensor* a,
    float s,
    float b,
    bool inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    float params[2] = { s, b };
    ggml_set_op_params(*result, &params, sizeof(params));

    result->op = GGML_OP_SCALE;
    result->src.push_back(a);

    return result;
}

ggml_tensor* ggml_im2col_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    std::array<int64_t, 4> ne,
    int                   s0,
    int                   s1,
    int                   p0,
    int                   p1,
    int                   d0,
    int                   d1,
    bool                  is_2D) {
    ggml_tensor* result = ctx->create(GGML_TYPE_F32, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(*result, params, sizeof(params));

    result->op = GGML_OP_IM2COL_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

static void ggml_compute_backward(
    ggml_context* ctx, ggml_cgraph* cgraph, ggml_tensor* tensor, std::unordered_map<ggml_tensor*, bool> &grads_needed) {
    ggml_tensor* grad = ggml_graph_get_grad(cgraph, tensor);

    if (!grad) {
        return;
    }

    ggml_tensor* src0 = tensor->src[0];
    ggml_tensor* src1 = tensor->src[1];
    ggml_tensor* src2 = tensor->src[2];
    const bool src0_needs_grads = src0 && grads_needed[src0];
    const bool src1_needs_grads = src1 && grads_needed[src1];
    const bool src2_needs_grads = src2 && grads_needed[src2];

    switch (tensor->op) {
    case GGML_OP_DUP: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, grad);
        }
    } break;
    case GGML_OP_ADD: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, grad);
        }
        if (src1_needs_grads) {
            struct ggml_tensor* tmp = grad;
            if (!ggml_are_same_shape(src0, src1)) {
                tmp = ggml_repeat_back(ctx, tmp, src1);
            }
            ggml_add_or_set(ctx, cgraph, src1, tmp);
        }
    } break;
    case GGML_OP_ADD1: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, grad);
        }
        if (src1_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
        }
    } break;
    case GGML_OP_ACC: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, grad);
        }
        if (src1_needs_grads) {
            const size_t nb1 = ((int32_t*)tensor->op_params)[0];
            const size_t nb2 = ((int32_t*)tensor->op_params)[1];
            const size_t nb3 = ((int32_t*)tensor->op_params)[2];
            const size_t offset = ((int32_t*)tensor->op_params)[3];

            ggml_tensor* tensor_grad_view = ggml_view(ctx,
                grad, { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
                { nb1, nb2, nb3 }, offset);

            ggml_add_or_set(ctx, cgraph, src1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
        }
    } break;
    case GGML_OP_SUB: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, grad);
        }
        if (src1_needs_grads) {
            ggml_sub_or_set(ctx, cgraph, src1, grad);
        }
    } break;
    case GGML_OP_MUL: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, grad, src1, false));
        }
        if (src1_needs_grads) {
            struct ggml_tensor* tmp = ggml_mul(ctx, src0, grad, false);
            if (!ggml_are_same_shape(src0, src1)) {
                tmp = ggml_repeat_back(ctx, tmp, src1);
            }
            ggml_add_or_set(ctx, cgraph, src1, tmp);
        }
    } break;
    case GGML_OP_DIV: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_div(ctx, grad, src1, false));
        }
        if (src1_needs_grads) {
            ggml_sub_or_set(ctx, cgraph, src1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1, false), false));
        }
    } break;
    case GGML_OP_SQR: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_scale(ctx, ggml_mul(ctx, src0, grad, false), 2.0f, false));
        }
    } break;
    case GGML_OP_SQRT: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_scale(ctx, ggml_div(ctx, grad, tensor, false), 0.5f, false));
        }
    } break;
    case GGML_OP_LOG: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_div(ctx, grad, src0, false));
        }
    } break;
    case GGML_OP_SIN: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, grad, ggml_cos(ctx, src0), false));
        }
    } break;
    case GGML_OP_COS: {
        if (src0_needs_grads) {
            ggml_sub_or_set(ctx, cgraph, src0, ggml_mul(ctx, grad, ggml_sin(ctx, src0), false));
        }
    } break;
    case GGML_OP_SUM: {
        if (src0_needs_grads) {
            ggml_add1_or_set(ctx, cgraph, src0, grad);
        }
    } break;
    case GGML_OP_SUM_ROWS: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_repeat(ctx, grad, src0));
        }
    } break;
    case GGML_OP_MEAN: {
        if (src0_needs_grads) {
            ggml_add1_or_set(ctx, cgraph, src0, ggml_scale_impl(ctx, grad, 1.0f / src0->ne[0], 0.0, false));
        }
    } break;
    case GGML_OP_REPEAT: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_repeat_back(ctx, grad, src0));
        }
    } break;
    case GGML_OP_REPEAT_BACK: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_repeat(ctx, grad, src0));
        }
    } break;
    case GGML_OP_RMS_NORM: {
        if (src0_needs_grads) {
            float eps;
            memcpy(&eps, tensor->op_params, sizeof(float));
            ggml_add_or_set(ctx, cgraph, src0, ggml_rms_norm_back(ctx, grad, src0, eps));
        }
    } break;
    case GGML_OP_MUL_MAT: {
        // https://cs231n.github.io/optimization-2/#staged
        // # forward pass
        // s0 = np.random.randn(5, 10)
        // s1 = np.random.randn(10, 3)
        // t = s0.dot(s1)

        // # now suppose we had the gradient on t from above in the circuit
        // dt = np.random.randn(*t.shape) # same shape as t
        // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
        // ds1 = t.T.dot(dt)

        // tensor.shape [m,p,qq,rr]
        // src0.shape   [n,m,q1,r1]
        // src1.shape   [n,p,qq,rr]

        if (src0_needs_grads) {
            GGML_ASSERT(grad->ne[2] == src1->ne[2]);
            GGML_ASSERT(grad->ne[3] == src1->ne[3]);
            struct ggml_tensor* tmp =
                ggml_out_prod(ctx, // [n,m,qq,rr]
                    src1,          // [n,p,qq,rr]
                    grad);         // [m,p,qq,rr]
            if (!ggml_are_same_shape(tmp, src0)) {
                GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                GGML_ASSERT(tmp->ne[3] == 1);

                const int64_t nr2 = tmp->ne[2] / src0->ne[2];
                const size_t nb2 = tmp->nb[2] * nr2;
                const size_t nb3 = tmp->nb[2];

                tmp = ggml_view(ctx, tmp, { src0->ne[0], src0->ne[1], src0->ne[2], nr2 }, { tmp->nb[1], nb2, nb3 }, 0);
                tmp = ggml_repeat_back(ctx, tmp, src0);
            }
            ggml_add_or_set(ctx, cgraph, src0, tmp);
        }
        if (src1_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src1,
                // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                //     ggml_cont(ctx,                  // [m,n,q1,r1]
                //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                //     grad, false),                          // [m,p,qq,rr]

                // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                // avoid transpose of src0, rather transpose smaller tensor->grad
                // and then use ggml_out_prod
                ggml_out_prod(ctx,      // [n,p,qq,rr]
                    src0,               // [n,m,q1,r1]
                    ggml_transpose(ctx, // [p,m,qq,rr]
                        grad)));        // [m,p,qq,rr]
        }
    } break;
    case GGML_OP_SCALE: {
        if (src0_needs_grads) {
            float s;
            memcpy(&s, tensor->op_params, sizeof(float));
            ggml_add_or_set(ctx, cgraph, src0, ggml_scale_impl(ctx, grad, s, 0.0, false));
        }
    } break;
    case GGML_OP_SET: {
        const size_t nb1 = ((const int32_t*)tensor->op_params)[0];
        const size_t nb2 = ((const int32_t*)tensor->op_params)[1];
        const size_t nb3 = ((const int32_t*)tensor->op_params)[2];
        const size_t offset = ((const int32_t*)tensor->op_params)[3];

        struct ggml_tensor* tensor_grad_view = NULL;

        if (src0_needs_grads || src1_needs_grads) {
            GGML_ASSERT(src0->type == tensor->type);
            GGML_ASSERT(!cgraph->grads[isrc0] || cgraph->grads[isrc0]->type == grad->type);
            GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

            tensor_grad_view = ggml_view(ctx,
                grad, { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] },
                { nb1, nb2, nb3 }, offset);
        }

        if (src0_needs_grads) {
            struct ggml_tensor* tmp = ggml_neg(ctx, tensor_grad_view);
            ggml_add_or_set(ctx, cgraph, src0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
        }

        if (src1_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
        }
    } break;
    case GGML_OP_CPY: {
        // cpy overwrites value of src1 by src0 and returns view(src1)
        // the overwriting is mathematically equivalent to:
        // tensor = src0 * 1 + src1 * 0
        if (src0_needs_grads) {
            // dsrc0 = dtensor * 1
            ggml_add_or_set(ctx, cgraph, src0, grad);
        }
        if (src1_needs_grads) {
            // dsrc1 = dtensor * 0 -> noop
        }
    } break;
    case GGML_OP_CONT: {
        // same as cpy
        if (src0_needs_grads) {
            GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
            GGML_ASSERT(ggml_is_contiguous(grad));
            GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
            ggml_add_or_set(ctx, cgraph, src0,
                ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
        }
    } break;
    case GGML_OP_RESHAPE: {
        if (src0_needs_grads) {
            struct ggml_tensor* grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
            ggml_add_or_set(ctx, cgraph, src0, ggml_reshape(ctx, grad_cont, src0));
        }
    } break;
    case GGML_OP_VIEW: {
        if (src0_needs_grads) {
            size_t offset;

            memcpy(&offset, tensor->op_params, sizeof(offset));

            size_t nb1 = tensor->nb[1];
            size_t nb2 = tensor->nb[2];
            size_t nb3 = tensor->nb[3];

            if (cgraph->grads[src0] && src0->type != cgraph->grads[src0]->type) {
                // gradient is typically F32, but src0 could be other type
                size_t ng = ggml_element_size(cgraph->grads[src0]);
                size_t n0 = ggml_element_size(src0);
                GGML_ASSERT(offset % n0 == 0);
                GGML_ASSERT(nb1 % n0 == 0);
                GGML_ASSERT(nb2 % n0 == 0);
                GGML_ASSERT(nb3 % n0 == 0);
                offset = (offset / n0) * ng;
                nb1 = (nb1 / n0) * ng;
                nb2 = (nb2 / n0) * ng;
                nb3 = (nb3 / n0) * ng;
            }

            ggml_acc_or_set(ctx, cgraph, src0, grad, nb1, nb2, nb3, offset);
        }
    } break;
    case GGML_OP_PERMUTE: {
        if (src0_needs_grads) {
            const int32_t* axes = (const int32_t*)tensor->op_params;
            const int axis0 = axes[0] & 0x3;
            const int axis1 = axes[1] & 0x3;
            const int axis2 = axes[2] & 0x3;
            const int axis3 = axes[3] & 0x3;
            int axb[4] = { 0,0,0,0 }; // axes backward
            axb[axis0] = 0;
            axb[axis1] = 1;
            axb[axis2] = 2;
            axb[axis3] = 3;
            ggml_add_or_set(ctx, cgraph, src0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
        }
    } break;
    case GGML_OP_TRANSPOSE: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_transpose(ctx, grad));
        }
    } break;
    case GGML_OP_GET_ROWS: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_get_rows_back(ctx, grad, src1, src0));
        }
        if (src1_needs_grads) {
            // noop
        }
    } break;
    case GGML_OP_DIAG_MASK_INF: {
        if (src0_needs_grads) {
            /* ggml_diag_mask_inf_impl() shouldn't be here */
            /* ref:  https://github.com/ggerganov/llama.cpp/pull/4203#discussion_r1412377992 */
            const int n_past = ((const int32_t*)tensor->op_params)[0];
            ggml_add_or_set(ctx, cgraph, src0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
        }
    } break;
    case GGML_OP_DIAG_MASK_ZERO: {
        if (src0_needs_grads) {
            const int n_past = ((const int32_t*)tensor->op_params)[0];
            ggml_add_or_set(ctx, cgraph, src0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
        }
    } break;
    case GGML_OP_SOFT_MAX: {
        if (src0_needs_grads) {
            float scale = 1.0f;
            float max_bias = 0.0f;

            memcpy(&scale, (const float*)tensor->op_params + 0, sizeof(float));
            memcpy(&max_bias, (const float*)tensor->op_params + 1, sizeof(float));

            ggml_add_or_set(ctx, cgraph, src0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
        }
        GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
    } break;
    case GGML_OP_ROPE: {
        if (src0_needs_grads) {
            //const int n_past = ((int32_t *) tensor->op_params)[0];
            const int n_dims = ((const int32_t*)tensor->op_params)[1];
            const int mode = ((const int32_t*)tensor->op_params)[2];
            //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
            const int n_ctx_orig = ((const int32_t*)tensor->op_params)[4];
            float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
            int sections[4] = { 0, 0, 0, 0 };

            memcpy(&freq_base, (const float*)tensor->op_params + 5, sizeof(float));
            memcpy(&freq_scale, (const float*)tensor->op_params + 6, sizeof(float));
            memcpy(&ext_factor, (const float*)tensor->op_params + 7, sizeof(float));
            memcpy(&attn_factor, (const float*)tensor->op_params + 8, sizeof(float));
            memcpy(&beta_fast, (const float*)tensor->op_params + 9, sizeof(float));
            memcpy(&beta_slow, (const float*)tensor->op_params + 10, sizeof(float));
            memcpy(&sections, tensor->op_params + 11, sizeof(sections));

            struct ggml_tensor* rope_back = grad->ne[2] == src1->ne[0] ?
                ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                    mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                    mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            ggml_add_or_set(ctx, cgraph, src0, rope_back);
        }
        GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
    } break;
    case GGML_OP_IM2COL: {
        if (src1_needs_grads) {
            const int32_t s0 = ggml_get_op_params_i32(tensor, 0);
            const int32_t s1 = ggml_get_op_params_i32(tensor, 1);
            const int32_t p0 = ggml_get_op_params_i32(tensor, 2);
            const int32_t p1 = ggml_get_op_params_i32(tensor, 3);
            const int32_t d0 = ggml_get_op_params_i32(tensor, 4);
            const int32_t d1 = ggml_get_op_params_i32(tensor, 5);
            const bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

            ggml_add_or_set(ctx, cgraph, src1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
        }
    } break;
    case GGML_OP_POOL_2D: {
        if (src0_needs_grads) {
            const auto op = std::bit_cast<ggml_op_pool>(tensor->op_params[0]);
            const      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
            const      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
            const      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
            const      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
            const      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
            const      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

            ggml_add_or_set(ctx, cgraph, src0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
        }
    } break;
    case GGML_OP_WIN_PART:
    case GGML_OP_WIN_UNPART:
    case GGML_OP_UNARY: {
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_ABS: {
            if (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad, false));
            }
        } break;
        case GGML_UNARY_OP_SGN: {
            // noop
        } break;
        case GGML_UNARY_OP_NEG: {
            if (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, src0, grad);
            }
        } break;
        case GGML_UNARY_OP_STEP: {
            // noop
        } break;
        case GGML_UNARY_OP_RELU: {
            if (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, ggml_step(ctx, src0), grad, false));
            }
        } break;
        case GGML_UNARY_OP_SILU: {
            if (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src0, ggml_silu_back(ctx, grad, src0));
            }
        } break;
        case GGML_UNARY_OP_EXP: {
            if (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, tensor, grad, false));
            }
        } break;
        case GGML_UNARY_OP_EXPM1: {
            if (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, grad, ggml_exp(ctx, src0), false));
            }
        } break;
        case GGML_UNARY_OP_SOFTPLUS: {
            if (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0), false));
            }
        } break;
        default: {
            fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
            GGML_ABORT("fatal error");
        } //break;
        }
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS: {
        if (src0_needs_grads) {
            ggml_add_or_set(ctx, cgraph, src0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
        }
        GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
    } break;
    case GGML_OP_GLU: {
        switch (ggml_get_glu_op(tensor)) {
        case GGML_GLU_OP_SWIGLU: {
            if (src0_needs_grads) {
                GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                ggml_add_or_set(ctx, cgraph, src0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1, false), src0));
            }
            if (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, src1, ggml_mul(ctx, ggml_silu(ctx, src0, false), grad, false));
            }
        } break;
        default: {
            GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
        } //break;
        }
    } break;
    case GGML_OP_NONE: {
        // noop
    } break;
    case GGML_OP_COUNT:
    default: {
        GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op))
    } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[src0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[src1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[src2]));
}

void ggml_cgraph::visit_parents(ggml_tensor* node)
{
    // check if already visited
    if (use_counts.contains(node)) return;

    use_counts[node] = 0;
    if (order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) {
        std::for_each(node->src.begin(), node->src.end(), [this](ggml_tensor* src) {
            if (src) {
                visit_parents(src);

                // Update the use count for this operand.
                use_counts[src]++;
            }
        });
    }
    else {
        std::for_each(node->src.rbegin(), node->src.rend(), [this](ggml_tensor* src) {
            if (src) {
                visit_parents(src);

                // Update the use count for this operand.
                use_counts[src]++;
            }
        });
    }

    if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
        if (node->get_name().empty()) {
            node->set_name("leaf_{}", leafs.size());
        }
        leafs.push_back(node);
    }
    else {
        if (node->get_name().empty()) {
            node->set_name("node_{}", nodes.size());
        }
        nodes.push_back(node);
    }
}

void ggml_cgraph::build_forward_expand(ggml_tensor* tensor)
{
    const size_t ori_nodes = nodes.size();
    visit_parents(tensor);
    const size_t append_nodes = nodes.size() - ori_nodes;

    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, append_nodes);

    if (append_nodes > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(nodes.back() == tensor);
    }
}

void ggml_cgraph::clear()
{
    nodes.clear();
    leafs.clear();
    grads.clear();
    grad_accs.clear();
    use_counts.clear();
}

void ggml_cgraph::reset()
{
    for (auto &node : nodes) {
        ggml_tensor* grad_acc = grad_accs[node];

        if (node->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(node->src[2]);
            ggml_set_zero(node->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        if (grad_acc) {
            if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                const float onef = 1.0f;
                if (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, sizeof(float));
                }
                else {
                    GGML_ASSERT(grad_acc->data);
                    *((float*)grad_acc->data) = onef;
                }
            }
            else {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

void ggml_cgraph::build_backward_expand(ggml_context* ctx, std::span<ggml_tensor*> grad_accs)
{
    GGML_ASSERT(nodes.size() > 0);

    const int n_nodes_f = nodes.size();
	std::unordered_map<ggml_tensor*, bool> grads_needed;

    {
        bool any_params = false;
        bool any_loss = false;
        for (int i = 0; i < n_nodes_f; ++i) {
            ggml_tensor* node = nodes[i];
            any_params = any_params || (node->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss = any_loss || (node->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    for (int i = 0; i < n_nodes_f; ++i) {
        ggml_tensor* node = nodes[i];

        if (node->type == GGML_TYPE_I32) {
            continue;
        }

        bool node_needs_grad = (node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = { false };
        switch (node->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
        case GGML_OP_IM2COL:      // only used for its shape
        case GGML_OP_IM2COL_BACK: // same as IM2COL
            ignore_src[0] = true;
            break;
        case GGML_OP_UNARY: {
            const enum ggml_unary_op uop = ggml_get_unary_op(node);
            // SGN and STEP unary ops are piecewise constant
            if (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                ignore_src[0] = true;
            }
        } break;

                          // gradients in node->src[1] for one reason or another have no effect on output gradients
        case GGML_OP_CPY:           // gradients in CPY target are irrelevant
        case GGML_OP_GET_ROWS:      // row indices not differentiable
        case GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
        case GGML_OP_ROPE:          // positions not differentiable
            ignore_src[1] = true;
            break;

        default:
            break;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || ignore_src[j] || !grads_needed[node->src[j]]) {
                continue;
            }
            GGML_ASSERT(node->src[j]->type == GGML_TYPE_F32 || node->src[j]->type == GGML_TYPE_F16);
            node_needs_grad = true;
            break;
        }
        if (!node_needs_grad) {
            continue;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE);

        if (i < grad_accs.size()) {
            this->grad_accs[node] = grad_accs[i];
            grads[node] = this->grad_accs[node];
        }
        else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            const auto& ne = node->ne;
            this->grad_accs[node] = ctx->create(GGML_TYPE_F32, ne);
            grads[node] = this->grad_accs[node];
        }
        grads_needed[node] = true;
    }

    for (int i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, this, nodes[i], grads_needed);
    }
}

void graph_print(const ggml_cgraph* cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = {}", cgraph->nodes.size());
    for (size_t i = 0; i < cgraph->nodes.size(); i++) {
        auto &node = cgraph->nodes[i];

        GGML_LOG_INFO(" - {:3}: [ {:5}, {:5}, {:5}] {:16} {}",
            i,
            node->ne[0], node->ne[1], node->ne[2],
            ggml_op_name(node->op), (node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
            ggml_graph_get_grad(cgraph, node) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = {}", cgraph->leafs.size());
    for (size_t i = 0; i < cgraph->leafs.size(); i++) {
        auto &node = cgraph->leafs[i];

        GGML_LOG_INFO(" - {:3}: [ {:5}, {:5}] {:8} {:16}",
            i,
            node->ne[0], node->ne[1],
            ggml_op_name(node->op),
            node->get_name());
    }

    GGML_LOG_INFO("========================================");
}

static ggml_tensor* ggml_graph_get_parent(const ggml_cgraph* cgraph, const ggml_tensor* node)
{
    for (auto &parent : cgraph->nodes) {
        ggml_tensor* grad = ggml_graph_get_grad(cgraph, parent);

        if (grad == node) {
            return parent;
        }
    }

    return NULL;
}

static bool ggml_graph_find(const ggml_cgraph* cgraph, const ggml_tensor* node) {
    if (cgraph == NULL) {
        return true;
    }

    auto it = std::ranges::find(cgraph->nodes, node);
    return it != cgraph->nodes.end();
}

static void ggml_graph_dump_dot_node_edge(
    std::ofstream& os,
    const ggml_cgraph* gb, 
    ggml_tensor* node, 
    ggml_tensor* parent, std::string_view label) 
{
    ggml_tensor* gparent = ggml_graph_get_parent(gb, node);
    ggml_tensor* gparent0 = ggml_graph_get_parent(gb, parent);
    os << std::format(R"(  "{}" -> "{}" [ arrowhead = {}; style = {}; label = "{}"; ])",
        gparent0 ? (void*)gparent0 : (void*)parent,
        gparent ? (void*)gparent : (void*)node,
        gparent ? "empty" : "vee",
        gparent ? "dashed" : "solid",
        label) << "\n";
}

static void ggml_graph_dump_dot_leaf_edge(
    std::ofstream& os,
    ggml_tensor* node,
    ggml_tensor* parent, 
    std::string_view label) {
    os << std::format(R"(  "{}" -> "{}" [ label = "{}"; ]\n)",
        (void*)parent,
        (void*)node,
        label);
}

void ggml_graph_dump_dot(const ggml_cgraph* gb, const ggml_cgraph* gf, const char* filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        return;
    }
    outFile << "digraph G {\n";
    outFile << "  newrank = true;\n";
    outFile << "  rankdir = TB;\n";

    for (size_t i = 0; i < gb->nodes.size(); i++) {
        auto& node = gb->nodes[i];
        ggml_tensor* grad = ggml_graph_get_grad(gb, node);

        if (ggml_graph_get_parent(gb, node) != nullptr) {
            continue;
        }

        const std::string_view color = [&] {
            if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                return "yellow";
            }
            else if (grad) {
                if (ggml_graph_find(gf, node)) {
                    return "green";
                }
                else {
                    return "lightblue";
                }
            }
            else {
                return "white";
            }
        }();

        outFile << std::format(
            R"(  "{}" [ style = filled; fillcolor = {}; shape = record; label=")", (void*)node, color);

        if (!node->get_name().empty()) {
            outFile << std::format("{} ({})|", node->get_name(), ggml_type_name(node->type));
        }
        else {
            outFile << std::format("({})|", ggml_type_name(node->type));
        }

        if (ggml_is_matrix(node)) {
            outFile << std::format("{} [{}, {}] | <x>{}", i, node->ne[0], node->ne[1], ggml_op_symbol(node->op));
        }
        else {
            outFile << std::format("{} [{}, {}, {}] | <x>{}", i, node->ne[0], node->ne[1], node->ne[2], ggml_op_symbol(node->op));
        }

        if (grad) {
            outFile << std::format(R"( | <g>{}"; ])", ggml_op_symbol(grad->op)) << "\n";
        }
        else {
            outFile << std::format(R"("; ])") << "\n";
        }
    }

    for (size_t i = 0; i < gb->leafs.size(); i++) {
        auto& node = gb->leafs[i];
        const std::string_view color = "pink";

        outFile << std::format(
            R"(  "{}" [ style = filled; fillcolor = {}; shape = record; label="<x>)",
            (void*)node, color);

        if (!node->get_name().empty()) {
            outFile << std::format("{} ({})|", node->get_name(), ggml_type_name(node->type));
        }
        else {
            outFile << std::format("({})|", ggml_type_name(node->type));
        }

        outFile << std::format("CONST {} [{}, {}]", i, node->ne[0], node->ne[1]);
        if (node->nelements() < 5 && node->data != NULL) {
            outFile << std::format(" | (");
            for (int j = 0; j < node->nelements(); j++) {
                // FIXME: use ggml-backend to obtain the tensor data
                //if (node->type == GGML_TYPE_I8 || node->type == GGML_TYPE_I16 || node->type == GGML_TYPE_I32) {
                //    fprintf(fp, "%d", ggml_get_i32_1d(node, j));
                //}
                //else if (node->type == GGML_TYPE_F32 ||
                //         node->type == GGML_TYPE_F16 ||
                //         node->type == GGML_TYPE_BF16) {
                //    fprintf(fp, "%.1e", (double)ggml_get_f32_1d(node, j));
                //}
                //else
                {
                    outFile << std::format("#");
                }
                if (j < node->nelements() - 1) {
                    outFile << std::format(", ");
                }
            }
            outFile << std::format(")");
        }
        outFile << std::format(R"("; ])") << "\n";
    }

    for (auto &node : gb->nodes)
        for (size_t i = 0; i < node->src.size(); i++) {
            if (!node->src[i]) continue;
            std::string label = std::format("src {}", i);
            ggml_graph_dump_dot_node_edge(outFile, gb, node, node->src[i], label);
        }

    for (auto &node : gb->leafs)
        for (size_t i = 0; i < node->src.size(); i++) {
            if (!node->src[i]) continue;
            std::string label = std::format("src {}", i);
            ggml_graph_dump_dot_leaf_edge(outFile, node, node->src[i], label);
        }

    outFile << "}\n";
}

void ggml_cgraph::add_node(ggml_tensor* tensor) {
    nodes.push_back(tensor);
}

ggml_tensor* ggml_cgraph::get_tensor(std::string_view name)
{
    for (auto leaf : leafs) {
        if (leaf->name == name)
            return leaf;
    }

    for (auto node : nodes) {
        if (node->name == name)
            return node;
    }

    return nullptr;
}

void ggml_dump_graph_nodes(ggml_cgraph* cgraph, const char* binary_name)
{
    FILE* fp = fopen(binary_name, "wb");
    for (int32_t i = 0; i < cgraph->nodes.size(); i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (ggml_is_view_op(node->op)) continue;
        fwrite(&i, sizeof(i), 1, fp);
        int64_t length = node->nbytes();
        fwrite(&length, sizeof(length), 1, fp);
        std::vector<uint8_t> data(length);
        ggml_backend_tensor_get(node, data.data(), 0, sizeof(uint8_t) * length);
        fwrite(data.data(), sizeof(uint8_t), length, fp);
    }
    fclose(fp);
}

int32_t ggml_cgraph::get_use_count(int node_idx) const {
    const ggml_tensor* node = nodes[node_idx];

    if (!use_counts.contains(node)) return 0;
    return use_counts.at(node);
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
static inline bool ggml_node_has_n_uses(const ggml_cgraph* cgraph, int node_idx, int32_t n_uses) {
    ggml_tensor* node = cgraph->nodes[node_idx];

    // check the use count against how many we're replacing
    if (cgraph->get_use_count(node_idx) != n_uses) {
        return false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    if (node->view_src) {
        return false;
    }

    // If the user requested output for the node, can't fuse
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }

    return true;
}

// Returns true if nodes [i, i+ops.size()) are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
bool ggml_can_fuse(const ggml_cgraph* cgraph, int node_idx, const enum ggml_op* ops, int num_ops) {
    if (node_idx + num_ops > cgraph->nodes.size()) {
        return false;
    }

    for (int i = 0; i < num_ops; ++i) {
        ggml_tensor* node = cgraph->nodes[node_idx + i];
        if (node->op != ops[i]) {
            return false;
        }
        if (i < num_ops - 1 && !ggml_node_has_n_uses(cgraph, node_idx + i, 1)) {
            return false;
        }
        if (i > 0) {
            ggml_tensor* prev = cgraph->nodes[node_idx + i - 1];
            if (node->src[0] != prev && node->src[1] != prev) {
                return false;
            }
            if (!ggml_are_same_shape(node, prev)) {
                return false;
            }
        }
    }
    return true;
}