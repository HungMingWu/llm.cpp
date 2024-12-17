module;
#include <assert.h>
#include <stdlib.h>
#include <memory>
#include <optional>
#include <ranges>

#define GGML_ASSERT(...) assert(__VA_ARGS__)

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

ggml_backend_sched::ggml_backend_sched(std::unique_ptr<ggml_backend>* backends,
    ggml_backend_buffer_type_t* bufts,
    int n_backends,
    size_t graph_size,
    bool parallel,
    bool op_offload)
{

    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(backends[n_backends - 1]->get_device()->get_type() == GGML_BACKEND_DEVICE_TYPE_CPU);

    const char* GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
    n_backends = n_backends;
    n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
#if 0
    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2;
    node_backend_ids = (int*)calloc(nodes_size, sizeof(node_backend_ids[0]));
    leaf_backend_ids = (int*)calloc(nodes_size, sizeof(leaf_backend_ids[0]));
    prev_node_backend_ids = (int*)calloc(nodes_size, sizeof(prev_node_backend_ids[0]));
    prev_leaf_backend_ids = (int*)calloc(nodes_size, sizeof(prev_leaf_backend_ids[0]));

    context_buffer_size = ggml_sched_max_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    context_buffer = (char*)malloc(context_buffer_size);

    for (int b = 0; b < n_backends; b++) {
        this->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (n_copies > 1) {
            for (int c = 0; c < n_copies; c++) {
                events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

#endif
    reset();
}

ggml_backend_sched::~ggml_backend_sched()
{
#if 0
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
#endif
}

void ggml_backend_sched::reset()
{
    // reset state for the next run
    if (!is_reset) {
#if 0
        ggml_hash_set_reset(&sched->hash_set);
        memset(hv_tensor_backend_ids, -1, hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(hv_tensor_copies, 0, hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor*));
#endif
        is_reset = true;
    }
    is_alloc = false;
}

// returns the priority of the backend, lower id is higher priority
int ggml_backend_sched::get_backend_id(ggml_backend_t backend) {
    for (int i = 0; i < n_backends; i++) {
        if (backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

size_t ggml_backend_sched::get_buffer_size(ggml_backend_t backend)
{
    int backend_index = get_backend_id(backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < n_backends);

    return galloc->get_buffer_size(backend_index);
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

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor* tensor, const struct ggml_tensor* op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i]->supports_buft(buffer->get_type()) &&
            sched->backends[i]->supports_op(op)) {
            return i;
        }
    }

#if 0 //ndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor* tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != nullptr) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
#if 0
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
#endif
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
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
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && src->buffer->is_host()) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (sched->backends[b]->supports_op(tensor) && sched->backends[b]->offload_op(tensor)) {
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

static bool ggml_is_view_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, ggml_tensor* t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = nullptr;

    if (buf) {
        // the tensor is already allocated
        buft = buf->get_type();
    }
    else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        const auto tensor_backend_id = [=]() -> std::optional<int> {
            auto it = sched->hv_tensor_backend_ids.find(t);
            if (it == sched->hv_tensor_backend_ids.end() && t->view_src) {
                it = sched->hv_tensor_backend_ids.find(t->view_src);
            }
            return it != sched->hv_tensor_backend_ids.end() ? it->second : std::optional<int>{};
        }();
        if (tensor_backend_id.has_value()) {
            buft = sched->bufts[tensor_backend_id.value()];
        }
    }

    return buft != nullptr && sched->backends[backend_id]->supports_buft(buft);
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched::split_graph(ggml_cgraph* graph) {
    // reset splits
    splits.clear();
    n_graph_inputs = 0;
    is_reset = false;

#if 0
    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(ctx);

    ctx = ggml_init(params);
    if (ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }
#endif

    // pass 1: assign backends to ops with pre-allocated inputs
    for (auto leaf : graph->leafs) {
        auto it = hv_tensor_backend_ids.find(leaf);
        // do not overwrite user assignments
        if (it == end(hv_tensor_backend_ids)) {
            int id = ggml_backend_sched_backend_id_from_cur(this, leaf);
            if (id != -1)
                hv_tensor_backend_ids[leaf] = id;
        }
    }

    for (auto node : graph->nodes) {
        auto it = hv_tensor_backend_ids.find(node);
        // do not overwrite user assignments
        if (it == end(hv_tensor_backend_ids)) {
            int id = ggml_backend_sched_backend_id_from_cur(this, node);
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
    for (int cur_backend_id = -1; auto node : graph->nodes) {
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
                it->second = result.value();
        }
    }
    // expand gpu up
    for (int cur_backend_id = -1, i = graph->nodes.size() - 1; i >= 0; i--) {
        ggml_tensor* node = graph->nodes[i];
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
                it->second = result.value();
        }
    }
    // expand rest down
    for (int cur_backend_id = -1; auto node : graph->nodes) {
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
                it->second = result.value();
        }
    }
    // expand rest up
    for (int cur_backend_id = -1, i = graph->nodes.size() - 1; i >= 0; i--) {
        ggml_tensor* node = graph->nodes[i];
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
                it->second = result.value();
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
    for (auto node : graph->nodes) {
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
                        if ((hv_tensor_backend_ids.count(src) || hv_tensor_backend_ids.count(src->view_src)) && ggml_backend_sched_buffer_supported(this, src, b)) {
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
                        if (!ggml_backend_sched_buffer_supported(this, src, b)) {
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
    for (auto node : graph->nodes) {
        auto it = hv_tensor_backend_ids.find(node);
        //int* cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != nullptr && it == hv_tensor_backend_ids.end()) {
            auto it2 = hv_tensor_backend_ids.find(node->view_src);
            if (it2 != hv_tensor_backend_ids.end())
                it->second = it2->second;
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
    }
    // pass 5: split graph, find tensors that need to be copied
    {
        auto &split = splits.emplace_back();
        split.i_start = 0;
        split.n_inputs = 0;
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->nodes.size(); i++) {
            ggml_tensor* node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split.backend_id = hv_tensor_backend_ids.at(node);
                break;
            }
        }

        int cur_backend_id = split.backend_id;

        for (; i < graph->nodes.size(); i++) {
            ggml_tensor* node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            // all nodes should be assigned by now, this can happen if there is no CPU fallback
            const int node_backend_id = hv_tensor_backend_ids.at(node);

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split.n_inputs > 0) {
                for (auto src : node->src) {
                    if (!src) continue;
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = hv_tensor_backend_ids.at(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(this, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split.n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        // To Fix
#if 0
                        int src_backend_id = hv_tensor_backend_ids.at(src);
                        bool supported = ggml_backend_sched_buffer_supported(this, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
#endif
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split.i_end = i;
                split = splits.emplace_back();
                split.backend_id = node_backend_id;
                split.i_start = i;
                split.n_inputs = 0;
                cur_backend_id = node_backend_id;
            }
            // find inputs that are not on the same backend
            for (auto src : node->src) {
                if (!src) continue;

                // all inputs should be assigned by now
                const int src_backend_id = hv_tensor_backend_ids.at(src);

                if (src->flags & GGML_TENSOR_FLAG_INPUT && n_copies > 1) {
                    auto& hv_tensor_copies_backend = hv_tensor_copies[src][src_backend_id];
                    if (hv_tensor_copies_backend.empty()) {
                        ggml_backend_t backend = backends[src_backend_id];
                        for (int c = 0; c < n_copies; c++) {
                            struct ggml_tensor* tensor_copy;
                            if (c == cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            }
                            else {
                                tensor_copy = ggml_dup_tensor_layout(ctx, src);
                                tensor_copy->set_name("{}#{}#{}", backend->get_name(), src->name, c);
                            }
                            if (n_copies > 1) {
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_INPUT);
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_OUTPUT); // prevent ggml-alloc from overwriting the tensor
                            }
                            hv_tensor_copies_backend.push_back(tensor_copy);
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(this, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    auto& hv_tensor_copies_backend = hv_tensor_copies[src][cur_backend_id];
                    if (hv_tensor_copies_backend.empty()) {
                        ggml_backend_t backend = backends[cur_backend_id];
                        for (int c = 0; c < n_copies; c++) {
                            ggml_tensor* tensor_copy = ggml_dup_tensor_layout(ctx, src);
                            tensor_copy->set_name("{}#{}#{}", backend->get_name(), src->name, c);
                            if (n_copies > 1) {
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_INPUT);
                                tensor_copy->set_flag(GGML_TENSOR_FLAG_OUTPUT); // prevent ggml-alloc from overwriting the tensor
                            }
                            hv_tensor_copies_backend.push_back(tensor_copy);
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split.n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split.inputs[n_inputs] = src;
                    }
#if 0
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
#endif
                }
            }
        }
        split.i_end = graph->nodes.size();
    }
#if 0
    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int* tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sched->n_copies;
    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor**)realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor*));
        sched->graph.leafs = (ggml_tensor**)realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor*));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph* graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split* split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor* input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor* input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor* input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor* input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor* input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split* split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor* input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor* input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor* leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }
#endif
}

bool ggml_backend_sched::reserve(ggml_cgraph* measure_graph)
{
#if 0
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
#endif
    split_graph(measure_graph);

#if 0
    ggml_backend_sched_synchronize(sched);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
#endif
    return false;
}