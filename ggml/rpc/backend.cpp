module;
#include <stdint.h>
#include <string.h>
#include <unordered_set>
#include <vector>

module ggml;
import :rpc.backend;
import :rpc.ds;
import :rpc.helper;
import :rpc.socket;

static void add_tensor(ggml_tensor* tensor, std::vector<rpc_tensor>& tensors, std::unordered_set<ggml_tensor*>& visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph* cgraph, std::vector<uint8_t>& output) {
    uint32_t n_nodes = cgraph->nodes.size();
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2 * sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t* dest = output.data();
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
}

void ggml_backend_rpc::synchronize()
{
	// this is no-op because we don't have any async operations
}

ggml_status ggml_backend_rpc::graph_compute_impl(ggml_cgraph* cgraph)
{
    //GGML_ASSERT(cgraph->nodes.size() > 0);
    bool reuse = gc.is_cached(cgraph);
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = device;
        auto sock = get_socket(endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    }
    else {
        gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(device, cgraph, input);
        auto sock = get_socket(endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}
