module;
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :traits;
import :quants;
import :cpu.from_float;

bool ggml_quantize_requires_imatrix(enum ggml_type type) {
    return
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ1_S;//   ||
    //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
	enum ggml_type   type,
	const float* src,
	void* dst,
	int64_t   start,
	int64_t   nrows,
	int64_t   n_per_row,
	const float* imatrix) {
    const int64_t n = (int64_t)nrows * n_per_row;

    if (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != nullptr);
    }

    //GGML_ASSERT(start % type_traits[type].blck_size == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    const size_t start_row = start / n_per_row;
    const size_t row_size = ggml_row_size(type, n_per_row);

    size_t result = 0;
    switch (type) {
    case GGML_TYPE_Q4_0:    result = quantize_q4_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q4_1:    result = quantize_q4_1(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_MXFP4:   result = quantize_mxfp4(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q5_0:    result = quantize_q5_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q5_1:    result = quantize_q5_1(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q8_0:    result = quantize_q8_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q2_K:    result = quantize_q2_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q3_K:    result = quantize_q3_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q4_K:    result = quantize_q4_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q5_K:    result = quantize_q5_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_Q6_K:    result = quantize_q6_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_TQ1_0:   result = quantize_tq1_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_TQ2_0:   result = quantize_tq2_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ2_S:   result = quantize_iq2_s(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ3_S:   result = quantize_iq3_s(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ1_S:   result = quantize_iq1_s(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ1_M:   result = quantize_iq1_m(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
    case GGML_TYPE_F16:
    {
        size_t elemsize = sizeof(ggml_fp16_t);
        from_float(src + start, (ggml_fp16_t*)dst + start, n);
        result = n * elemsize;
    } break;
    case GGML_TYPE_BF16:
    {
        size_t elemsize = sizeof(ggml_bf16_t);
        from_float(src + start, (ggml_bf16_t*)dst + start, n);
        result = n * elemsize;
    } break;
    case GGML_TYPE_F32:
    {
        size_t elemsize = sizeof(float);
        result = n * elemsize;
        memcpy((uint8_t*)dst + start * elemsize, src + start, result);
    } break;
    default:
        assert(false);
    }

    GGML_ASSERT(result == nrows * row_size);

    return result;
}

enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype) {
    enum ggml_type wtype = GGML_TYPE_COUNT;

    switch (ftype) {
    case GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   break;
    case GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   break;
    case GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  break;
    case GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  break;
    case GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  break;
    case GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  break;
    case GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  break;
    case GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  break;
    case GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; break;
    case GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  break;
    case GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  break;
    case GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  break;
    case GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  break;
    case GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  break;
    case GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  break;
    case GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   break;
    case GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  break;
    case GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    break;
    case GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    break;
    case GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   break;
    case GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   break;
    case GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    break;
    case GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    break;
    case GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_COUNT; break;
    case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_COUNT; break;
    }

    GGML_ASSERT(wtype != GGML_TYPE_COUNT);

    return wtype;
}

void ggml_backend_tensor_copy(ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (src->buffer->is_host()) {
        ggml_backend_tensor_set(dst, src->data, 0, src->nbytes());
    }
    else if (dst->buffer->is_host()) {
        ggml_backend_tensor_get(src, dst->data, 0, src->nbytes());
    }
    else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#if 0 //ndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif
        size_t nbytes = src->nbytes();
        std::vector<std::byte> data(nbytes);
        ggml_backend_tensor_get(src, data.data(), 0, nbytes);
        ggml_backend_tensor_set(dst, data.data(), 0, nbytes);
    }
}

static void graph_copy_init_tensor(std::unordered_map<ggml_tensor*, ggml_tensor*>& node_copies,
    std::unordered_map<ggml_tensor*, bool>& node_init, ggml_tensor* src) {
    if (node_init[src]) {
        return;
    }
    node_init[src] = true;

    ggml_tensor* dst = node_copies[src];
    if (dst->view_src != nullptr) {
        graph_copy_init_tensor(node_copies, node_init, src->view_src);
        ggml_backend_view_init(dst);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (auto s : src->src) {
        if (!s) continue;
        graph_copy_init_tensor(node_copies, node_init, s);
    }
}

struct graph_copy {
    std::unique_ptr<ggml_backend_buffer> buffer;
    ggml_context ctx_allocated;
    ggml_context ctx_unallocated;
    ggml_cgraph graph;
};

static ggml_tensor* graph_copy_dup_tensor(std::unordered_map<ggml_tensor*, ggml_tensor*>& node_copies,
    ggml_context* ctx_allocated, ggml_context* ctx_unallocated, ggml_tensor* src) {

    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(src->data && "graph must be allocated");

    auto it = node_copies.find(src);
    if (it != node_copies.end()) return it->second;

    ggml_tensor* dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != nullptr) {
        dst->view_src = graph_copy_dup_tensor(node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    dst->name = src->name;

    // copy src
    for (auto s : src->src) {
        if (!s) dst->src.push_back(nullptr);
        else dst->src.push_back(graph_copy_dup_tensor(node_copies, ctx_allocated, ctx_unallocated, s));
    }

    node_copies[src] = dst;
    return dst;
}

static graph_copy ggml_backend_graph_copy(ggml_backend* backend, ggml_cgraph* graph) {
    std::unordered_map<ggml_tensor*, ggml_tensor*> node_copies;
    std::unordered_map<ggml_tensor*, bool> node_init;

    ggml_context ctx_allocated, ctx_unallocated;

    // dup nodes
    for (auto node : graph->nodes) {
        graph_copy_dup_tensor(node_copies, &ctx_allocated, &ctx_unallocated, node);
    }

    // allocate nodes
    std::unique_ptr<ggml_backend_buffer> buffer = backend->alloc_tensors(&ctx_allocated);

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (auto node : graph->nodes) {
        graph_copy_init_tensor(node_copies, node_init, node);
    }

    // build graph copy
    ggml_cgraph graph_copy;
    for (auto node : graph->nodes)
        graph_copy.nodes.push_back(node_copies[node]);

    return {
        /* .buffer           = */ std::move(buffer),
        /* .ctx_allocated    = */ std::move(ctx_allocated),
        /* .ctx_unallocated  = */ std::move(ctx_unallocated),
        /* .graph            = */ std::move(graph_copy),
    };
}

bool ggml_backend_compare_graph_backend(ggml_backend* backend1, ggml_backend* backend2, ggml_cgraph* graph, ggml_backend_eval_callback callback, ggml_tensor* test_node) {
    graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    assert(copy.buffer);

    ggml_cgraph* g1 = graph;
    ggml_cgraph* g2 = &copy.graph;
    assert(g1->nodes.size() == g2->nodes.size());

    auto create_view_graph = [](ggml_tensor* node) {
        ggml_cgraph graph;
        graph.nodes.push_back(node);
        return graph;
    };

    if (test_node != nullptr) {
        // Compute the whole graph and only test the output for a specific tensor
        backend1->graph_compute(g1);
        backend2->graph_compute(g2);

        int test_node_idx = -1;
        for (size_t i = 0; i < g1->nodes.size(); i++) {
            ggml_tensor* t1 = g1->nodes[i];
            if (t1 == test_node) {
                test_node_idx = i;
                break;
            }
        }
        GGML_ASSERT(test_node_idx != -1);

        return callback(g1->nodes[test_node_idx], g2->nodes[test_node_idx]);
    }
    else {
        // clang's views::zip have bug, keep old style here
        for (size_t i = 0; i < g1->nodes.size(); i++) {
            auto t1 = g1->nodes[i];
            auto t2 = g2->nodes[i];
            //printf("eval %d/%d\n", i, g1->n_nodes);
            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));
            ggml_cgraph g1v = create_view_graph(t1);
            ggml_cgraph g2v = create_view_graph(t2);

            backend1->compute(&g1v);
            backend2->compute(&g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(t1, t2)) {
                return false;
            }
        }
        return true;
    }
}

std::string utf16_to_utf8(const std::wstring& str) {
    std::string result;
    result.reserve(str.size() * 2);
    for (wchar_t wc : str) {
        if (wc <= 0x7F) {
            result.push_back(static_cast<char>(wc));
        }
        else if (wc <= 0x7FF) {
            result.push_back(0xC0 | ((wc >> 6) & 0x1F));
            result.push_back(0x80 | (wc & 0x3F));
        }
        else if (wc <= 0xFFFF) {
            result.push_back(0xE0 | ((wc >> 12) & 0x0F));
            result.push_back(0x80 | ((wc >> 6) & 0x3F));
            result.push_back(0x80 | (wc & 0x3F));
        }
        else {
            throw std::runtime_error("Character out of UTF-8 range");
        }
    }
    return result;
}

ggml_tensor* ggml_dup_tensor_layout(ggml_context* ctx, const ggml_tensor* tensor) {
    ggml_tensor* dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}