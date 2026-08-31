module;
#include <stdint.h>
#include <string.h>
#include <memory>
#include <unordered_set>
#include <vector>

module ggml;
import :rpc.backend;
import :rpc.device;
import :rpc.ds;
import :rpc.helper;
import :rpc.socket;

static void add_tensor(ggml_tensor* tensor, const ggml_cgraph* cgraph, const std::shared_ptr<rpc_dispatcher>& dispatcher, std::vector<rpc_tensor>& tensors, std::unordered_set<ggml_tensor*>& visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    auto it = cgraph->use_counts.find(tensor);
    if (it != cgraph->use_counts.end()) {
        result.use_count = it->second;
    }
    tensors.push_back(result);
}

static uint8_t* serialize_graph(uint32_t device, const ggml_cgraph* cgraph, const std::shared_ptr<rpc_dispatcher>& dispatcher, size_t* output_size) {
    uint32_t n_nodes = cgraph->nodes.size();
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2 * sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    uint8_t* output = new uint8_t[*output_size]();
    uint8_t* dest = output;
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor* out_tensors = (rpc_tensor*)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
    return output;
}

void ggml_backend_rpc::synchronize()
{
    dispatcher->synchronize();
}

ggml_status ggml_backend_rpc::graph_compute_impl(ggml_cgraph* cgraph)
{
    //GGML_ASSERT(cgraph->nodes.size() > 0);
    ggml_backend_rpc_device* dev = (ggml_backend_rpc_device*)get_device();
    bool reuse = cgraph->uid != 0 && dev->last_graph_uid == cgraph->uid;
    if (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = device;
        dispatcher->send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
    }
    else {
        dev->last_graph_uid = cgraph->uid;
        size_t input_size = 0;
        uint8_t* input = serialize_graph(device, cgraph, dispatcher, &input_size);
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        dispatcher->send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_rpc::event_record(ggml_backend_event* event) {
    dispatcher->event_record(event);
}

void ggml_backend_rpc::event_wait(ggml_backend_event* /*event*/) {
}
