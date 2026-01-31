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
import :cuda.buffer_type;
import :cuda.fused;
import :cuda.utils;

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

inline bool ggml_can_fuse_subgraph(const ggml_cgraph* cgraph,
    int                                 start_idx,
    std::initializer_list<enum ggml_op> ops,
    std::initializer_list<int>          outputs = {}) {
    return fused::ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
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

    const bool split = dynamic_cast<cuda_split_backend_buffer_type*>(ffn_up->src[0]->buffer->get_type()) ||
        dynamic_cast<cuda_split_backend_buffer_type*>(ffn_gate->src[0]->buffer->get_type());

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

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

    static bool ggml_cuda_topk_moe_fusion(const ggml_cgraph* cgraph, int node_idx, ggml_cuda_topk_moe_args& args) {
        args.sigmoid = false;
        args.softmax = false;
        args.delayed_softmax = false;
        args.prob_bias = false;
        args.norm = false;

        const int      n_nodes = cgraph->nodes.size();
        auto  nodes = cgraph->nodes.data();

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
                return true;
            }
        }

        if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
            ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            const ggml_tensor* ffn_gate = cgraph->nodes[node_idx];
            const ggml_tensor* ffn_up = cgraph->nodes[node_idx + 1];
            const ggml_tensor* glu = cgraph->nodes[node_idx + 2];

            if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
                return true;
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

            //rms_norm kernel assumes contigous rows
            if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
                return false;
            }

            if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
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

    void add(cudaStream_t stream, ggml_tensor* dst, int n_fuse)
    {
        bin_bcast_context ctx = utils::create_bcast_context(dst->src[0], dst->src[1], dst);
        fused_add_cuda(ctx, n_fuse, stream);
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
            src0_d, rms_norm_src->ne, rms_norm_src->nb,  mul_d,
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

        topk_moe_context ctx {
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
