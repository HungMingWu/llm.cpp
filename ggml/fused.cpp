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
import :fused;

static int ggml_node_list_find_tensor(const ggml_cgraph* cgraph,
    const int* idxs,
    int                        count,
    const ggml_tensor* tensor) {
    GGML_ASSERT(cgraph && idxs);
    for (int i = 0; i < count; ++i) {
        const int node_idx = idxs[i];

        if (node_idx >= cgraph->nodes.size()) {
            return -1;
        }
        if (cgraph->nodes[node_idx] == tensor) {
            return i;
        }
    }
    return -1;
}

static bool ggml_can_fuse_subgraph_ext(const ggml_cgraph* cgraph,
    const int* node_idxs,
    int                        count,
    const enum ggml_op* ops,
    const int* outputs,
    int                        num_outputs) {
    GGML_ASSERT(outputs && num_outputs > 0);

    for (int i = 0; i < count; ++i) {
        if (node_idxs[i] >= cgraph->nodes.size()) {
            return false;
        }

        const struct ggml_tensor* node = cgraph->nodes[node_idxs[i]];

        if (node->op != ops[i]) {
            return false;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }

        if (ggml_node_list_find_tensor(cgraph, outputs, num_outputs, node) != -1) {
            continue;
        }

        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }

        int subgraph_uses = 0;
        for (int j = i + 1; j < count; ++j) {
            const ggml_tensor* other_node = cgraph->nodes[node_idxs[j]];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                if (other_node->src[src_idx] == node) {
                    subgraph_uses++;
                }
            }
        }

        if (subgraph_uses != cgraph->get_use_count(node_idxs[i])) {
            return false;
        }

        // if node is a view, check if the view_src and all it's parent view_srcs are within the subgraph
        struct ggml_tensor* view_src = node->view_src;
        while (view_src) {
            if (ggml_node_list_find_tensor(cgraph, node_idxs, count, view_src) == -1) {
                return false;
            }
            view_src = view_src->view_src;
        }
    }

    return true;
}

static bool ggml_cuda_should_use_topk_moe(const ggml_tensor* gating_op,
    const ggml_tensor* weights,
    const ggml_tensor* logits,
    const ggml_tensor* ids) {
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor* softmax = gating_op;
        float               scale = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float*)softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float*)softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    }
    else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor* ffn_up,
    const ggml_tensor* ffn_gate,
    const ggml_tensor* glu,
    const ggml_tensor* ffn_up_bias = nullptr,
    const ggml_tensor* ffn_gate_bias = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }

    const bool is_mul_mat = ffn_up->op == GGML_OP_MUL_MAT && ffn_gate->op == GGML_OP_MUL_MAT && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id = ffn_up->op == GGML_OP_MUL_MAT_ID && ffn_gate->op == GGML_OP_MUL_MAT_ID && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (glu->src[0] != ffn_gate_bias || glu->src[1] != ffn_up_bias) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul = ffn_up_bias->src[0] == ffn_up || ffn_up_bias->src[1] == ffn_up;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate || ffn_gate_bias->src[1] == ffn_gate;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        }
        else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up || ffn_gate_bias->src[0] != ffn_gate) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }
    else {
        if (glu->src[0] != ffn_gate && glu->src[1] != ffn_up) {
            return false;
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (ffn_up->src[2] && (ffn_up->src[2] != ffn_gate->src[2])) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 3> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

#if 0
    const bool split = dynamic_cast<cuda_split_backend_buffer_type*>(ffn_up->src[0]->buffer->get_type()) ||
        dynamic_cast<cuda_split_backend_buffer_type*>(ffn_gate->src[0]->buffer->get_type());

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }
#endif
    return true;
}

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor* rope,
    const ggml_tensor* view,
    const ggml_tensor* set_rows) {
    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t*)rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
static bool ggml_node_has_n_uses(const ggml_cgraph* cgraph, int node_idx, int32_t n_uses) {
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

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
static inline bool ggml_can_fuse_ext(const ggml_cgraph* cgraph, const int* node_idxs, const enum ggml_op* ops, int num_ops) {
    for (int i = 0; i < num_ops; ++i) {
        if (node_idxs[i] + num_ops > cgraph->nodes.size()) {
            return false;
        }

        struct ggml_tensor* node = cgraph->nodes[node_idxs[i]];
        if (node->op != ops[i]) {
            return false;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (i < num_ops - 1 && !ggml_node_has_n_uses(cgraph, node_idxs[i], 1)) {
            return false;
        }
        if (i > 0) {
            struct ggml_tensor* prev = cgraph->nodes[node_idxs[i - 1]];
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

namespace fused
{
    bool ggml_can_fuse_subgraph(const ggml_cgraph* cgraph,
        int                                 start_idx,
        std::initializer_list<enum ggml_op> ops,
        std::initializer_list<int>          outputs) {
        return fused::ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
    }

    // same as above, for sequential indices starting at node_idx
    bool ggml_can_fuse(const ggml_cgraph* cgraph, int node_idx, const enum ggml_op* ops, int num_ops) {
        assert(num_ops < 32);

        if (node_idx + num_ops > cgraph->nodes.size()) {
            return false;
        }

        int idxs[32];
        for (int i = 0; i < num_ops; ++i) {
            idxs[i] = node_idx + i;
        }

        return ggml_can_fuse_ext(cgraph, idxs, ops, num_ops);
    }

    // nicer C++ syntax for ggml_can_fuse
    bool ggml_can_fuse(const struct ggml_cgraph* cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
        return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());
    }

    // returns whether the write (out) nodes overwrite the read nodes in operation
    bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph* cgraph,
        const int           node_idx,
        const int           node_count,
        const int* out_nodes,
        const int           out_count,
        const bool          is_topk_moe) {
        auto nodes_overlap = [&](const ggml_tensor* a, const ggml_tensor* b) {
            const int64_t a_start = (int64_t)a->data;
            const int64_t a_end = a_start + a->buffer->get_alloc_size(a);

            const int64_t b_start = (int64_t)b->data;
            const int64_t b_end = b_start + b->buffer->get_alloc_size(b);

            if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
                return true;
            }

            return false;
        };

        bool is_ok = true;
        // exception for topk-moe, as each row is read entirely before writing
        if (ggml_nrows(cgraph->nodes[node_idx]) == 1 && is_topk_moe) {
            return true;
        }

        for (int i = 0; i < out_count; ++i) {
            const ggml_tensor* dst = cgraph->nodes[out_nodes[i]];

            for (int j = node_idx; j < node_idx + node_count; ++j) {
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.

                for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                    const ggml_tensor* src = cgraph->nodes[j]->src[src_idx];

                    if (!src || src->op == GGML_OP_NONE) {
                        continue;
                    }

                    if (nodes_overlap(dst, src)) {
                        bool found = false;

                        for (int k = node_idx; k < j; ++k) {
                            if (cgraph->nodes[k] == src) {
                                found = true;
                                break;
                            }
                        }

                        if (!found) {
                            is_ok = false;
                            break;
                        }
                    }
                }
            }
        }

        return is_ok;
    }

    bool ggml_cuda_can_fuse(const ggml_cgraph* cgraph,
        int                                       node_idx,
        std::initializer_list<enum ggml_op>       ops,
        std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
        const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
        GGML_ASSERT(unary_ops.size() == num_unary);
#endif

        const auto is_equal = [](const std::initializer_list<enum ggml_op>& list1,
            const std::initializer_list<enum ggml_op>& list2) {
                return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
        };

        std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
        std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

        std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
        std::initializer_list<enum ggml_op> mul_mat_glu_ops = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

        if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
            ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
            const ggml_tensor* ffn_gate = cgraph->nodes[node_idx];
            const ggml_tensor* ffn_gate_bias = cgraph->nodes[node_idx + 1];
            const ggml_tensor* ffn_up = cgraph->nodes[node_idx + 2];
            const ggml_tensor* ffn_up_bias = cgraph->nodes[node_idx + 3];
            const ggml_tensor* glu = cgraph->nodes[node_idx + 4];

            if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
                int out_nodes[] = { node_idx + 4 };
                return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
            }
        }

        if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
            ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            const ggml_tensor* ffn_gate = cgraph->nodes[node_idx];
            const ggml_tensor* ffn_up = cgraph->nodes[node_idx + 1];
            const ggml_tensor* glu = cgraph->nodes[node_idx + 2];

            if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
                int out_nodes[] = { node_idx + 2 };
                return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
            }
        }

        std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

        if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            const ggml_tensor* rope = cgraph->nodes[node_idx];
            const ggml_tensor* view = cgraph->nodes[node_idx + 1];
            const ggml_tensor* set_rows = cgraph->nodes[node_idx + 2];

            if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
                return true;
            }
        }

        if (!ggml_can_fuse(cgraph, node_idx, ops)) {
            return false;
        }

        if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
            const ggml_tensor* rms_norm = cgraph->nodes[node_idx];
            const ggml_tensor* mul = cgraph->nodes[node_idx + 1];
            const ggml_tensor* add = nullptr;

            if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
                add = cgraph->nodes[node_idx + 2];
            }

            GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
            GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

            //rms norm only supports F32
            if (mul->src[0]->type != GGML_TYPE_F32 ||
                mul->src[1]->type != GGML_TYPE_F32 ||
                mul->type != GGML_TYPE_F32) {
                return false;
            }

            if (add && (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32)) {
                return false;
            }

            //if rms norm is the B operand, then we don't handle broadcast
            if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
                return false;
            }

            //rms_norm kernel assumes contiguous rows
            if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
                return false;
            }

            if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
                return false;
            }

            return true;
        }

        if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
            && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
            const ggml_tensor* ssm_conv = cgraph->nodes[node_idx];
            const ggml_tensor* silu = cgraph->nodes[node_idx + 1];
            if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
                return false;
            }

            if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
                return false;
            }

            return true;
        }

        if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
            && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
            const ggml_tensor* ssm_conv = cgraph->nodes[node_idx];
            const ggml_tensor* add = cgraph->nodes[node_idx + 1];
            const ggml_tensor* silu = cgraph->nodes[node_idx + 2];
            if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
                return false;
            }

            if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
                return false;
            }

            // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
            const ggml_tensor* bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
            if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
                return false;
            }
            if (bias->nelements() != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
                return false;
            }

            return true;
        }

        if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
            && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
            const ggml_tensor* unary = cgraph->nodes[node_idx];
            const ggml_tensor* mul = cgraph->nodes[node_idx + 1];

            if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
                return false;
            }

            if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
                return false;
            }

            if (unary->type != mul->type) {
                return false;
            }

            const ggml_tensor* other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
            if (other->type != unary->type) {
                return false;
            }
            if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
                return false;
            }

            return true;
        }

        if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
            && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
            const ggml_tensor* unary = cgraph->nodes[node_idx];
            const ggml_tensor* sqr = cgraph->nodes[node_idx + 1];

            if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
                return false;
            }

            if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
                return false;
            }

            if (unary->type != sqr->type) {
                return false;
            }

            if (!ggml_is_contiguous(unary->src[0])) {
                return false;
            }

            return true;
        }

        if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
            && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
            const ggml_tensor* scale = cgraph->nodes[node_idx];
            const ggml_tensor* tanh = cgraph->nodes[node_idx + 1];
            const ggml_tensor* scale2 = cgraph->nodes[node_idx + 2];

            GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
            GGML_ASSERT(scale->type == GGML_TYPE_F32);

            if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
                return false;
            }

            // Check for bias
            if (std::bit_cast<float>(scale->op_params[1]) != 0.0f || std::bit_cast<float>(scale2->op_params[1]) != 0.0f) {
                return false;
            }

            return true;
        }

        return false;
    }

    bool should_use_topk_moe(const ggml_tensor* gating_op,
        const ggml_tensor* weights,
        const ggml_tensor* logits,
        const ggml_tensor* ids) {
        const int n_expert = ids->nb[1] / ids->nb[0];
        if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 576) {
            return false;
        }

        if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
            return false;
        }

        if (gating_op->op == GGML_OP_SOFT_MAX) {
            const ggml_tensor* softmax = gating_op;
            float               scale = 1.0f;
            float               max_bias = 0.0f;

            memcpy(&scale, (const float*)softmax->op_params + 0, sizeof(float));
            memcpy(&max_bias, (const float*)softmax->op_params + 1, sizeof(float));

            if (!ggml_is_contiguous(softmax->src[0])) {
                return false;
            }

            if (scale != 1.0f || max_bias != 0.0f) {
                return false;
            }

            // don't fuse when masks or sinks are present
            if (softmax->src[1] || softmax->src[2]) {
                return false;
            }
        }
        else if (gating_op->op == GGML_OP_UNARY) {
            ggml_unary_op op = ggml_get_unary_op(gating_op);

            if (op != GGML_UNARY_OP_SIGMOID) {
                return false;
            }
        }

        return true;
    }

    // Returns true if the subgraph formed by {node_idxs} can be fused
    // checks whethers all nodes which are not part of outputs can be elided
    // by checking if their num_uses are confined to the subgraph
    bool ggml_can_fuse_subgraph(const ggml_cgraph* cgraph,
        int                        node_idx,
        int                        count,
        const ggml_op* ops,
        const int* outputs,
        int                        num_outputs) {
        GGML_ASSERT(count < 32);
        if (node_idx + count > cgraph->nodes.size()) {
            return false;
        }

        int idxs[32];

        for (int i = 0; i < count; ++i) {
            idxs[i] = node_idx + i;
        }

        return ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, outputs, num_outputs);
    }
}
