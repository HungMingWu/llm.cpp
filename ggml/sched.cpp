module;
#include <assert.h>
#include <stdlib.h>
#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS GGML_MAX_SRC
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

module ggml;

ggml_backend_sched::ggml_backend_sched(ggml_backend** backends,
    ggml_backend_buffer_type** bufts,
    int n_backends,
    bool parallel,
    bool op_offload)
{

    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(backends[n_backends - 1]->get_device()->get_type() == GGML_BACKEND_DEVICE_TYPE_CPU);

    const char* GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
    this->n_backends = n_backends;
    n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)

    for (int b = 0; b < n_backends; b++) {
        this->backends[b] = backends[b];
        this->bufts[b] = bufts ? bufts[b] : backends[b]->get_default_buffer_type();
        GGML_ASSERT(backends[b]->supports_buft(this->bufts[b]));

        if (n_copies > 1) {
            for (int c = 0; c < n_copies; c++) {
                events[b][c].reset(backends[b]->get_device()->event_new());
            }
        }
    }
    this->galloc = std::make_unique<ggml_gallocr>(std::span{ this->bufts, static_cast<size_t>(n_backends) });
    this->op_offload = op_offload;
    reset();
}

void ggml_backend_sched::reset()
{
    // reset state for the next run
    if (!is_reset) {
        hv_tensor_backend_ids.clear();
        hv_tensor_copies.clear();
        is_reset = true;
    }
    is_alloc = false;
}

// returns the priority of the backend, lower id is higher priority
std::optional<int> ggml_backend_sched::get_backend_id(ggml_backend* backend) {
    for (int i = 0; i < n_backends; i++) {
        if (backends[i] == backend) {
            return i;
        }
    }
    return std::nullopt;
}

size_t ggml_backend_sched::get_buffer_size(ggml_backend* backend)
{
    auto backend_index = get_backend_id(backend);
    GGML_ASSERT(backend_index.has_value());

    return galloc->get_buffer_size(backend_index.value());
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE * 16 + GGML_SCHED_MAX_SPLITS_DEBUG * GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

int ggml_backend_sched::backend_from_buffer(const ggml_tensor* tensor, const ggml_tensor* op) {
    ggml_backend_buffer* buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < n_backends; i++) {
        if (backends[i]->supports_buft(buffer->get_type()) &&
            backends[i]->supports_op(op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("{}: warning: no backend supports op {} with a weight with buffer type {} used in tensor {}, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), buffer->get_type()->get_name(), tensor->name);
#endif

    return -1;
}

// returns the backend that should be used for the node based on the current locations
int ggml_backend_sched::backend_id_from_cur(ggml_tensor* tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = backend_from_buffer(tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != nullptr) {
        cur_backend_id = backend_from_buffer(tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer* buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
#if 0
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
#endif
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor* src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = backend_from_buffer(src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (op_offload && src_backend_id == n_backends - 1 && src->buffer->is_host()) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (backends[b]->supports_op(tensor) && backends[b]->offload_op(tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

bool ggml_backend_sched::buffer_supported(ggml_tensor* t, int backend_id) {
    ggml_backend_buffer* buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type* buft = nullptr;

    if (buf) {
        // the tensor is already allocated
        buft = buf->get_type();
    }
    else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        const auto tensor_backend_id = [=, this]() -> std::optional<int> {
            auto it = hv_tensor_backend_ids.find(t);
            if (it == hv_tensor_backend_ids.end() && t->view_src) {
                it = hv_tensor_backend_ids.find(t->view_src);
            }
            return it != hv_tensor_backend_ids.end() ? it->second : std::optional<int>{};
        }();
        if (tensor_backend_id.has_value()) {
            buft = bufts[tensor_backend_id.value()];
        }
    }

    return buft != nullptr && backends[backend_id]->supports_buft(buft);
}

ggml_backend* ggml_backend_sched::get_tensor_backend(ggml_tensor* node) {
    auto it = hv_tensor_backend_ids.find(node);
    return (it == hv_tensor_backend_ids.end()) ? nullptr : backends[it->second];
}


void ggml_backend_sched::print_assignments(const ggml_cgraph& graph) {
    const auto fmt_size = [](size_t size) -> std::string {
        if (size >= 1024 * 1024) {
            return std::format("{}M", size / 1024 / 1024);
        }
        else {
            return std::format("{}K", size / 1024);
        }
    };
    const auto get_name = [](ggml_backend* backend) -> const char* {
        return backend ? backend->get_name() : "NULL";
    };
    auto cur_split = splits.cbegin();
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        if (cur_split != std::end(splits) && i == (*cur_split).i_start) {
            ggml_backend* split_backend = backends[(*cur_split).backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #{}: {} # {} inputs",
                std::distance(splits.cbegin(), cur_split), split_backend->get_name(),
                (*cur_split).inputs.size());
            for (bool first = true; const auto &input : (*cur_split).inputs) {
                if (first) {
                    GGML_LOG_DEBUG(": ");
                    first = false;
                }
                GGML_LOG_DEBUG("[{} ({:>5.5})] ", input->name,
                    fmt_size(input->nbytes()));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        ggml_tensor* node = graph.nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (debug > 1) {
            ggml_backend* tensor_backend = get_tensor_backend(node);
            GGML_LOG_DEBUG("node #{:3} ({:>10.10}): {:>20.20} ({:>5.5}) [{:>5.5} {:>8.8}] use={}:", i, ggml_op_name(node->op), node->name,
                fmt_size(node->nbytes()), get_name(tensor_backend), GET_CAUSE(node),
                graph.use_counts.at(node));
            for (auto src : node->src) {
                if (!src) continue;
                ggml_backend* src_backend = get_tensor_backend(src);
                GGML_LOG_DEBUG(" {:>20.20} ({:>5.5}) [{:>5.5} {:>8.8}]", src->name,
                    fmt_size(src->nbytes()), get_name(src_backend), GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

void ggml_backend_sched::synchronize() {
    for (int i = 0; i < n_backends; i++) {
        backends[i]->synchronize();
    }
    if (!is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        cur_copy = 0;
    }
}

ggml_cgraph ggml_graph_view(const ggml_cgraph& cgraph0, int i0, int i1) {
    ggml_cgraph view_graph;
    view_graph.order = cgraph0.order;
    for (int i = i0; i < i1; i++)
        view_graph.nodes.push_back(cgraph0.nodes[i]);
    return view_graph;
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched::split_graph(const ggml_cgraph& graph) {
    // reset splits
    splits.clear();
    graph_inputs.clear();
    is_reset = false;

    ctx = std::make_unique<ggml_context>();
    if (!ctx) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (auto leaf : graph.leafs) {
        auto it = hv_tensor_backend_ids.find(leaf);
        // do not overwrite user assignments
        if (it == end(hv_tensor_backend_ids)) {
            int id = backend_id_from_cur(leaf);
            if (id != -1)
                hv_tensor_backend_ids[leaf] = id;
        }
    }

    for (auto node : graph.nodes) {
        auto it = hv_tensor_backend_ids.find(node);
        // do not overwrite user assignments
        if (it == end(hv_tensor_backend_ids)) {
            int id = backend_id_from_cur(node);
            if (id != -1)
                hv_tensor_backend_ids[node] = id;

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor* src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int* src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    auto if_supported = [this](ggml_tensor* node, int cur_backend_id) -> std::optional<int> {
        if (backends[cur_backend_id]->supports_op(node)) {
            SET_CAUSE(node, "2.sup");
            return cur_backend_id;
        }
        return std::nullopt;
    };

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    for (int cur_backend_id = -1; auto node : graph.nodes) {
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        auto it = hv_tensor_backend_ids.find(node);
        if (it != hv_tensor_backend_ids.end()) {
            if (it->second == n_backends - 1) {
                // skip cpu (lowest prio backend)
                cur_backend_id = -1;
            }
            else {
                cur_backend_id = it->second;
            }
        }
        else if (cur_backend_id != -1) {
            auto result = if_supported(node, cur_backend_id);
            if (result.has_value())
                hv_tensor_backend_ids.insert({ node, result.value()});
        }
    }
    // expand gpu up
    for (int cur_backend_id = -1, i = graph.nodes.size() - 1; i >= 0; i--) {
        ggml_tensor* node = graph.nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        auto it = hv_tensor_backend_ids.find(node);
        if (it != hv_tensor_backend_ids.end()) {
            if (it->second == n_backends - 1) {
                // skip cpu (lowest prio backend)
                cur_backend_id = -1;
            }
            else {
                cur_backend_id = it->second;
            }
        }
        else if (cur_backend_id != -1) {
            auto result = if_supported(node, cur_backend_id);
            if (result.has_value())
                hv_tensor_backend_ids.insert({ node, result.value() });
        }
    }
    // expand rest down
    for (int cur_backend_id = -1; auto node : graph.nodes) {
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        auto it = hv_tensor_backend_ids.find(node);
        if (it != hv_tensor_backend_ids.end()) {
            cur_backend_id = it->second;
        }
        else if (cur_backend_id != -1) {
            auto result = if_supported(node, cur_backend_id);
            if (result.has_value())
                hv_tensor_backend_ids.insert({ node , result.value() });
        }
    }
    // expand rest up
    for (int cur_backend_id = -1, i = graph.nodes.size() - 1; i >= 0; i--) {
        ggml_tensor* node = graph.nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        auto it = hv_tensor_backend_ids.find(node);
        if (it != hv_tensor_backend_ids.end()) {
            cur_backend_id = it->second;
        }
        else if (cur_backend_id != -1) {
            auto result = if_supported(node, cur_backend_id);
            if (result.has_value())
                hv_tensor_backend_ids.insert({ node , result.value() });
        }
    }
    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (auto node : graph.nodes) {
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        auto it = hv_tensor_backend_ids.find(node);
        //int* node_backend_id = &tensor_backend_id(node);
        if (it == hv_tensor_backend_ids.end()) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < n_backends; b++) {
                if (backends[b]->supports_op(node)) {
                    int n_supported = 0;
                    for (auto src : node->src) {
                        if (!src) continue;
                        if ((hv_tensor_backend_ids.count(src) || hv_tensor_backend_ids.count(src->view_src)) && buffer_supported(src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        hv_tensor_backend_ids[node] = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        }
        else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < it->second; b++) {
                if (bufts[b] == bufts[it->second] && backends[b]->supports_op(node)) {
                    bool supported = true;
                    for (auto src : node->src) {
                        if (!src) continue;
                        if (!buffer_supported(src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        it->second = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }
    // pass 4: assign backends to remaining src from dst and view_src
    for (auto node : graph.nodes) {
        auto it = hv_tensor_backend_ids.find(node);
        if (node->view_src != nullptr && it == hv_tensor_backend_ids.end()) {
            auto it2 = hv_tensor_backend_ids.find(node->view_src);
            if (it2 != hv_tensor_backend_ids.end())
                hv_tensor_backend_ids.insert({ node, it2->second });
            SET_CAUSE(node, "4.vsrc");
        }
        for (auto src : node->src) {
            if (!src) continue;
            auto it2 = hv_tensor_backend_ids.find(src);
            if (it2 == hv_tensor_backend_ids.end()) {
                if (src->view_src != nullptr) {
                    // views are always on the same backend as the source
                    auto it3 = hv_tensor_backend_ids.find(src->view_src);
                    if (it3 != hv_tensor_backend_ids.end())
                        it2->second = it3->second;
                    SET_CAUSE(src, "4.vsrc");
                }
                else {
                    hv_tensor_backend_ids.insert({ src, it->second });
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < n_backends && hv_tensor_backend_ids.count(node) == 0; b++) {
            auto node_backend_id = if_supported(node, b);
            if (node_backend_id.has_value())
                hv_tensor_backend_ids.insert({ node, node_backend_id.value()});
        }
        assert(hv_tensor_backend_ids.count(node) != 0);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        ggml_backend_sched_split* cur_split = &splits.emplace_back();
        cur_split->i_start = 0;
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph.nodes.size(); i++) {
            ggml_tensor* node = graph.nodes[i];
            if (!ggml_is_view_op(node->op)) {
                cur_split->backend_id = hv_tensor_backend_ids.at(node);
                break;
            }
        }

        int cur_backend_id = cur_split->backend_id;

        for (; i < graph.nodes.size(); i++) {
            ggml_tensor* node = graph.nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            // all nodes should be assigned by now, this can happen if there is no CPU fallback
            const int node_backend_id = hv_tensor_backend_ids.at(node);

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && !cur_split->inputs.empty()) {
                for (auto src : node->src) {
                    if (!src) continue;
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = hv_tensor_backend_ids.at(src);
                        if (src_backend_id != cur_backend_id && !buffer_supported(src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (cur_split->inputs.size() == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        int src_backend_id = hv_tensor_backend_ids.at(src);
                        bool supported = buffer_supported(src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && hv_tensor_copies[src][cur_backend_id].empty() && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                cur_split->i_end = i;
                cur_split = &splits.emplace_back();
                cur_split->backend_id = node_backend_id;
                cur_split->i_start = i;
                cur_backend_id = node_backend_id;
            }
            // find inputs that are not on the same backend
            for (auto &src : node->src) {
                if (!src) continue;

                // all inputs should be assigned by now
                const int src_backend_id = hv_tensor_backend_ids.at(src);

                if (src->flags & GGML_TENSOR_FLAG_INPUT && n_copies > 1) {
                    auto& hv_tensor_copies_backend = hv_tensor_copies[src][src_backend_id];
                    if (hv_tensor_copies_backend.empty()) {
                        ggml_backend* backend = backends[src_backend_id];
                        for (int c = 0; c < n_copies; c++) {
                            struct ggml_tensor* tensor_copy;
                            if (c == cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            }
                            else {
                                tensor_copy = ggml_dup_tensor_layout(ctx.get(), src);
                                tensor_copy->set_name("{}#{}#{}", backend->get_name(), src->name, c);
                            }
                            if (n_copies > 1) {
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_INPUT);
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_OUTPUT); // prevent ggml-alloc from overwriting the tensor
                            }
                            hv_tensor_copies_backend.push_back(tensor_copy);
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        graph_inputs.push_back(src);
                    }
                }

                if (src_backend_id != cur_backend_id && !buffer_supported(src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    auto& hv_tensor_copies_backend = hv_tensor_copies[src][cur_backend_id];
                    if (hv_tensor_copies_backend.empty()) {
                        ggml_backend* backend = backends[cur_backend_id];
                        for (int c = 0; c < n_copies; c++) {
                            ggml_tensor* tensor_copy = ggml_dup_tensor_layout(ctx.get(), src);
                            tensor_copy->set_name("{}#{}#{}", backend->get_name(), src->name, c);
                            if (n_copies > 1) {
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_INPUT);
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_OUTPUT); // prevent ggml-alloc from overwriting the tensor
                            }
                            hv_tensor_copies_backend.push_back(tensor_copy);
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        cur_split->inputs.push_back(src);
                    }
                    src = hv_tensor_copies_backend[cur_copy];
                }
            }
        }
        cur_split->i_end = graph.nodes.size();
    }

    if (debug) {
        print_assignments(graph);
    }
    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        std::swap(node_backend_ids, prev_node_backend_ids);
        std::swap(leaf_backend_ids, prev_leaf_backend_ids);
    }

    this->graph.nodes.clear();
    this->graph.leafs.clear();
    std::unordered_map<int, int> map_node_backend_ids, map_leaf_backend_ids;
    ggml_cgraph& graph_copy = this->graph;

    for (auto &split : splits) {
        split.graph = ggml_graph_view(graph, split.i_start, split.i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        backends[split.backend_id]->graph_optimize(&split.graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (auto input : split.inputs) {
            ggml_tensor* input_cpy = hv_tensor_copies[input][split.backend_id][cur_copy];

            // add a dependency to the input source so that it is not freed before the copy is done
            ggml_tensor* input_dep = ggml_view_tensor(ctx.get(), input);
            input_dep->src.push_back(input);
            map_node_backend_ids[graph_copy.nodes.size()] = hv_tensor_backend_ids[input];
            graph_copy.nodes.push_back(input_dep);

            // add a dependency to the input copy so that it is allocated at the start of the split
            map_node_backend_ids[graph_copy.nodes.size()] = split.backend_id;
            graph_copy.nodes.push_back(input_cpy);
        }

        for (int j = split.i_start; j < split.i_end; j++) {
            map_node_backend_ids[graph_copy.nodes.size()] = hv_tensor_backend_ids[graph.nodes[j]];
            graph_copy.nodes.push_back(graph.nodes[j]);
        }
    }

    if (n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (auto input : graph_inputs) {
            int backend_id = hv_tensor_backend_ids.at(input);
            for (int c = 0; c < n_copies; c++) {
                ggml_tensor* input_cpy = hv_tensor_copies[input][backend_id][c];
                map_leaf_backend_ids[graph_copy.leafs.size()] = backend_id;
                graph_copy.leafs.push_back(input_cpy);
            }
        }
        for (auto &split : splits) {
            int backend_id = split.backend_id;
            for (auto input : split.inputs) {
                for (int c = 0; c < n_copies; c++) {
                    ggml_tensor* input_cpy = hv_tensor_copies[input][backend_id][c];
                    map_leaf_backend_ids[graph_copy.leafs.size()] = backend_id;
                    graph_copy.leafs.push_back(input_cpy);
                }
            }
        }
    }
    // add leafs from the original graph
    for (auto leaf : graph.leafs) {
        map_leaf_backend_ids[graph_copy.leafs.size()] = hv_tensor_backend_ids.at(leaf);
        graph_copy.leafs.push_back(leaf);
    }

    // trnasform back to vector
    node_backend_ids.resize(map_node_backend_ids.size());
    for (const auto [index, id] : map_node_backend_ids)
        node_backend_ids[index] = id;
    leaf_backend_ids.resize(map_leaf_backend_ids.size());
    for (const auto [index, id] : map_leaf_backend_ids)
        leaf_backend_ids[index] = id;
}

bool ggml_backend_sched::reserve(const ggml_cgraph* measure_graph)
{
    reset();

    synchronize();

    split_graph(*measure_graph);

    if (!galloc->reserve(graph, node_backend_ids, leaf_backend_ids)) {
        return false;
    }
    
    reset();

    return true;
}

bool ggml_backend_sched::alloc_splits() {
    bool backend_ids_changed = [&] {
        if (node_backend_ids.size() != prev_node_backend_ids.size()) return true;
        // clang have bug on std::views::zip right now
        for (size_t i = 0; i < node_backend_ids.size(); i++)
        {
            auto node = node_backend_ids[i];
            auto prev_node = prev_node_backend_ids[i];
            if (node != prev_node && bufts[node] != bufts[prev_node])
                return true;
        }
        return false;
    }();

    if (!backend_ids_changed) {
        backend_ids_changed = [&] {
		    if (leaf_backend_ids.size() != prev_leaf_backend_ids.size()) return true;
		    // clang have bug on std::views::zip right now
            for (size_t i = 0; i < leaf_backend_ids.size(); i++) {
                auto leaf = leaf_backend_ids[i];
                auto prev_leaf = prev_leaf_backend_ids[i];
                if (leaf != prev_leaf && bufts[leaf] != bufts[prev_leaf])
                    return true;
            }
            return false;
        }();
    }

    // allocate graph
    if (backend_ids_changed || !galloc->alloc_graph(&graph)) {
        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < n_backends; i++) {
            backends[i]->synchronize();
        }
#ifndef NDEBUG
        GGML_LOG_DEBUG("{}: failed to allocate graph, reserving (backend_ids_changed = {})", __func__, backend_ids_changed);
#endif
        galloc->reserve(graph, node_backend_ids, leaf_backend_ids);
        if (!galloc->alloc_graph(&graph)) {
            GGML_LOG_ERROR("{}: failed to allocate graph", __func__);
            return false;
        }
    }

    return true;
}

bool ggml_backend_sched::alloc_graph(const ggml_cgraph& graph) {
    split_graph(graph);

    if (!alloc_splits()) {
        return false;
    }
    is_alloc = true;
    return true;
}

ggml_status ggml_backend_sched::compute_splits() {
    ggml_tensor* prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (auto& split : splits) {
        int split_backend_id = split.backend_id;
        ggml_backend* split_backend = backends[split_backend_id];

        // copy the input tensors to the split backend
        for (size_t input_id = 0; input_id < split.inputs.size(); input_id++) {
            auto input = split.inputs[input_id];
            ggml_backend* input_backend = get_tensor_backend(input);
            ggml_tensor* input_cpy = hv_tensor_copies[input][split_backend_id][cur_copy];

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (events[split_backend_id][cur_copy] != nullptr) {
                    events[split_backend_id][cur_copy]->synchronize();
                }
                else {
                    split_backend->synchronize();
                }
                ggml_backend_tensor_copy(input, input_cpy);
            }
            else {
                // wait for the split backend to finish using the input before overwriting it
                if (events[split_backend_id][cur_copy] != nullptr) {
                    split_backend->event_wait(events[split_backend_id][cur_copy].get());
                }
                else {
                    split_backend->synchronize();
                }
                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor* node = split.graph.nodes[0];
                if (!split.graph.nodes.empty() &&
                    input->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    input->buffer->is_host() && (
                        (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                        //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                        )) {

                    const int64_t n_expert = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    input_backend->synchronize();

                    // get the ids
                    ggml_tensor* ids_tensor = node->src[2];
                    ggml_backend* ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (size_t i = input_id + 1; i < split.inputs.size(); i++) {
                        if (ids_tensor == hv_tensor_copies[split.inputs[i]][split_backend_id][cur_copy]) {
                            ids_tensor = split.inputs[i];
                            ids_backend = get_tensor_backend(split.inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ids_tensor->nbytes() / sizeof(int32_t));
                        ids_backend->get_tensor_async(ids_tensor, ids.data(), 0, ids_tensor->nbytes());
                        ids_backend->synchronize();

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1] / sizeof(int32_t) + i0 * ids_tensor->nb[0] / sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy = (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        split_backend->set_tensor_async(
                            input_cpy,
                            (const uint8_t*)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                        };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                }
                else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->cpy_tensor_async(input_backend, input, input_cpy)) {
                        input_backend->synchronize();
                        if (events[split_backend_id][cur_copy] != nullptr) {
                            events[split_backend_id][cur_copy]->synchronize();
                        }
                        else {
                            split_backend->synchronize();
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!callback_eval) {
            ggml_status ec = split_backend->graph_compute(&split.graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        }
        else {
            // similar to ggml_backend_compare_graph_backend
            for (size_t j0 = 0; j0 < split.graph.nodes.size(); j0++) {
                ggml_tensor* t = split.graph.nodes[j0];

                // check if the user needs data from this node
                bool need = callback_eval(t, true);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split.graph.nodes.size() - 1) {
                    t = split.graph.nodes[++j1];
                    need = callback_eval(t, true);
                }

                ggml_cgraph gv = ggml_graph_view(split.graph, j0, j1 + 1);

                ggml_status ec = split_backend->graph_compute(&gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                split_backend->synchronize();

                if (need && !callback_eval(t, false)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (!split.inputs.empty()) {
            if (events[split_backend_id][cur_copy] != nullptr) {
                split_backend->event_record(events[split_backend_id][cur_copy].get());
            }
        }
    }

    cur_copy = (cur_copy + 1) % n_copies;

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_backend_sched::graph_compute_async(const ggml_cgraph& graph) {
    if (!is_reset && !is_alloc) {
        reset();
    }

    if (!is_alloc) {
        if (!alloc_graph(graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }
    return compute_splits();
}

ggml_status ggml_backend_sched::graph_compute(const ggml_cgraph& graph)
{
    ggml_status err = graph_compute_async(graph);
    synchronize();
    return err;
}

void ggml_backend_sched::set_tensor_backend(ggml_tensor* node, ggml_backend* backend)
{
    auto backend_index = get_backend_id(backend);
    GGML_ASSERT(backend_index.has_value());
    hv_tensor_backend_ids[node] = backend_index.value();
    SET_CAUSE(node, "usr");
    is_reset = false;
}

void ggml_backend_sched::dump_dot(const ggml_cgraph* graph, const char* filename)
{

}
