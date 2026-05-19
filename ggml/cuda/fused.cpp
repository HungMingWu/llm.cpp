module;
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <bit>
#include <vector>
#include "fused/fused.h"
#include "op/cuda_func.h"
#define GGML_ASSERT(x) assert(x)

module ggml;
import :cuda.fused;
import :cuda.buffer_type;
import :cuda.utils;

namespace fused
{
    bool should_mul_mat_vec_f(const ggml_tensor* tensor) {
        ggml_tensor* src0 = tensor->src[0];
        ggml_tensor* src1 = tensor->src[1];
        const ggml_tensor* dst = tensor;

        const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

        bool use_mul_mat_vec_f =
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        use_mul_mat_vec_f = use_mul_mat_vec_f && utils::should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

        const bool split = (dynamic_cast<cuda_split_backend_buffer_type*>(src0->buffer->get_type()) != nullptr) ||
            (dynamic_cast<cuda_split_backend_buffer_type*>(src1->buffer->get_type()) != nullptr);

        //TODO: add support for fusion for split buffers
        if (split) {
            return false;
        }

        //we only support fusion for ncols_dst = 1
        if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
            return false;
        }

        if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
            return false;
        }


        return use_mul_mat_vec_f;
    }

    bool should_mul_mat_vec_q(const ggml_tensor* tensor) {
        ggml_tensor* src0 = tensor->src[0];
        ggml_tensor* src1 = tensor->src[1];
        const ggml_tensor* dst = tensor;

        const bool bad_padding_clear = src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            src0->nbytes() != src0->buffer->get_alloc_size(src0) &&
            src0->view_src;

        bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

        // fusion is not universally faster on Pascal
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (cc <= GGML_CUDA_CC_PASCAL) {
            return false;
        }
        //we only support fusion for ncols_dst = 1
        if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
            return false;
        }

        if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
            return false;
        }

        const bool split = (dynamic_cast<cuda_split_backend_buffer_type*>(src0->buffer->get_type()) != nullptr) ||
            (dynamic_cast<cuda_split_backend_buffer_type*>(src1->buffer->get_type()) != nullptr);

        //TODO: add support for fusion for split buffers
        if (split) {
            return false;
        }

        return use_mul_mat_vec_q;
    }

    bool ggml_cuda_topk_moe_fusion(const ggml_cgraph* cgraph, int node_idx, ggml_cuda_topk_moe_args& args) {
        args.sigmoid = false;
        args.softmax = false;
        args.delayed_softmax = false;
        args.prob_bias = false;
        args.norm = false;

        const int      n_nodes = cgraph->nodes.size();
        const auto nodes = cgraph->nodes.data();

        if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
            args.softmax = true;
        }

        if (nodes[node_idx]->op == GGML_OP_UNARY) {
            if (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
                return false;
            }
            args.sigmoid = true;
        }

        if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
            args.delayed_softmax = true;
        }

        node_idx++;

        if (args.sigmoid || args.softmax) {
            // SOFTMAX -> RESHAPE
            if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            ggml_tensor* probs_reshaped = nodes[node_idx];
            node_idx++;

            if (node_idx >= n_nodes) {
                return false;
            }

            // src of bias add is the unreshaped probs (-2 instead of -1)
            if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
                args.prob_bias = true;
                node_idx++;
            }
            // RESHAPE/ADD -> ARGSORT
            if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
                return false;
            }

            if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
                return false;
            }

            node_idx++;

            // ARGSORT-> VIEW
            if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;

            if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
                return false;
            }

            // GET_ROWS
            if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
        else if (args.delayed_softmax) {
            if (node_idx - 2 < 0) {
                return false;
            }
            ggml_tensor* probs_reshaped = nodes[node_idx - 2];

            // VIEW->ARGSORT
            if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;

            // GET_ROWS
            if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
                return false;
            }
            node_idx++;

            static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

            for (const ggml_op op : remaining_ops) {
                if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                    return false;
                }
                node_idx++;
            }
        }

        // At this point we can check for norm + scale. Everything is now at least valid till the norm
        if (node_idx >= n_nodes) {
            return true;
        }

        if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
            //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
            static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

            args.norm = true;
            for (const ggml_op op : norm_ops) {
                if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                    node_idx++;
                }
                else {
                    args.norm = false;
                    return true;
                }
            }

            // DIV <- CLAMP, RESHAPE
            if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
                args.norm = false;
                return true;
            }
            node_idx++;

            if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                args.norm = false;
                return true;
            }

            node_idx++;
        }

        if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
            args.scale = true;
        }

        return true;
    }

    void add(cudaStream_t stream, ggml_tensor* dst, int n_fuse)
    {
        bin_bcast_context ctx = utils::create_bcast_context(dst->src[0], dst->src[1], dst);
        fused_add_cuda(ctx, n_fuse, stream);
    }

    void mul(cudaStream_t stream, ggml_tensor* dst, int n_fuse)
    {
        bin_bcast_context ctx = utils::create_bcast_context(dst->src[0], dst->src[1], dst);
        fused_mul_cuda(ctx, n_fuse, stream);
    }

    // fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
    void softcap(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* src) {
        const ggml_tensor* src0 = src->src[0];
        const float* src0_d = (const float*)src0->data;
        float* dst_d = (float*)dst->data;

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);

        float scale;
        float softcap;
        memcpy(&scale, (float*)src->op_params + 0, sizeof(float));
        memcpy(&softcap, (float*)dst->op_params + 0, sizeof(float));

        softcap_f32_cuda(src0_d, dst_d, scale, softcap, src0->nelements(), stream);
    }

    void rms_norm(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* mul_tensor) {
        const ggml_tensor* rms_norm_src = dst->src[0];
        float eps = 0.0f;

        memcpy(&eps, dst->op_params, sizeof(float));

        const float* src0_d = (const float*)rms_norm_src->data;
        const float* mul_d = nullptr;
        const ggml_tensor* mul_src = nullptr;

        if (mul_tensor->src[0] == dst) {
            mul_d = (float*)mul_tensor->src[1]->data;
            mul_src = mul_tensor->src[1];
        }
        else if (mul_tensor->src[1] == dst) {
            mul_d = (float*)mul_tensor->src[0]->data;
            mul_src = mul_tensor->src[0];
        }
        else {
            GGML_ASSERT(false);
        }

        float* dst_d = (float*)mul_tensor->data;

        GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(eps >= 0.0f);

        const int mul_ncols = mul_src->ne[0];
        const int mul_nrows = mul_src->ne[1];
        const int mul_nchannels = mul_src->ne[2];
        const int mul_nsamples = mul_src->ne[3];

        rms_norm_mul_f32_cuda(stream, eps, dst_d, mul_tensor->ne, mul_tensor->nb,
            src0_d, rms_norm_src->ne, rms_norm_src->nb, mul_d, mul_src->ne, mul_src->nb);
    }

    void rms_norm_add(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor) {
        const ggml_tensor* rms_norm_src = dst->src[0];
        float               eps = 0.0f;

        memcpy(&eps, dst->op_params, sizeof(float));

        const float* src0_d = (const float*)rms_norm_src->data;
        const float* mul_d = nullptr;
        const ggml_tensor* mul_src = nullptr;

        if (mul_tensor->src[0] == dst) {
            mul_d = (float*)mul_tensor->src[1]->data;
            mul_src = mul_tensor->src[1];
        }
        else if (mul_tensor->src[1] == dst) {
            mul_d = (float*)mul_tensor->src[0]->data;
            mul_src = mul_tensor->src[0];
        }
        else {
            GGML_ASSERT(false);
        }

        const float* add_d = nullptr;
        const ggml_tensor* add_src = nullptr;

        if (add_tensor->src[0] == mul_tensor) {
            add_d = (float*)add_tensor->src[1]->data;
            add_src = add_tensor->src[1];
        }
        else if (add_tensor->src[1] == mul_tensor) {
            add_d = (float*)add_tensor->src[0]->data;
            add_src = add_tensor->src[0];
        }
        else {
            GGML_ASSERT(false);
        }

        float* dst_d = (float*)add_tensor->data;

        GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(eps >= 0.0f);

        rms_norm_mul_f32_cuda(stream, eps, dst_d, add_tensor->ne, add_tensor->nb,
            src0_d, rms_norm_src->ne, rms_norm_src->nb, mul_d,
            mul_src->ne, mul_src->nb, add_d, add_src->ne, add_src->nb);
    }

    void topk_moe(cudaStream_t stream,
        const ggml_tensor* logits,
        ggml_tensor* weights,
        ggml_tensor* ids,
        const ggml_tensor* clamp,
        const ggml_tensor* scale,
        const ggml_tensor* bias,
        const ggml_cuda_topk_moe_args& args)
    {
        GGML_ASSERT(logits->type == GGML_TYPE_F32);
        GGML_ASSERT(weights->type == GGML_TYPE_F32);
        GGML_ASSERT(ids->type == GGML_TYPE_I32);

        const int n_experts = logits->ne[0];

        GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t)n_experts);

        topk_moe_context ctx{
            .has_bias = bias != nullptr,
            .logits = (const float*)logits->data,
            .weights = (float*)weights->data,
            .bias = bias ? (float*)bias->data : nullptr,
            .ids = (int32_t*)ids->data,
            .n_rows = logits->ne[1],
            .n_experts = n_experts,
            .n_expert_used = weights->ne[1],
            .clamp_val = clamp ? std::bit_cast<float>(clamp->op_params[0]) : -INFINITY,
            .scale_val = scale ? std::bit_cast<float>(scale->op_params[0]) : 1.0f,
            .config = {
                .use_sigmoid = args.sigmoid,
                .with_norm = clamp != nullptr,
                .delayed_softmax = args.delayed_softmax
            }
        };

        topk_moe_cuda(ctx, stream);
    }
}
