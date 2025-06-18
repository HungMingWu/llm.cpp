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

static void serialize_graph(const ggml_cgraph* cgraph, std::vector<uint8_t>& output) {
    uint32_t n_nodes = cgraph->nodes.size();
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (auto &node : cgraph->nodes) {
        add_tensor(node, tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + sizeof(n_nodes) + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    uint32_t* out_ntensors = (uint32_t*)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t));
    *out_ntensors = n_tensors;
    rpc_tensor* out_tensors = (rpc_tensor*)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

void ggml_backend_rpc::synchronize()
{
	// this is no-op because we don't have any async operations
}

ggml_status ggml_backend_rpc::graph_compute(ggml_cgraph* cgraph)
{
    std::vector<uint8_t> input;
    serialize_graph(cgraph, input);
    rpc_msg_graph_compute_rsp response;
    auto sock = get_socket(endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return (enum ggml_status)response.result;
}
