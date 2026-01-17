module;
#include <vector>
#include "common.h"

module ggml:cuda.graph;
import :ds;
import :log;

struct ggml_cuda_graph_node_properties {
    void* node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void* src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_cuda_graph_node_properties> props;

    void record_update(bool use_graph, bool update_required) {
        if (use_graph && update_required) {
            number_consecutive_updates++;
        }
        else {
            number_consecutive_updates = 0;
        }
        if (number_consecutive_updates >= 4) {
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
            disable_due_to_too_many_updates = true;
        }
    }

    bool is_enabled() const {
        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
        return !(disable_due_to_gpu_arch || disable_cuda_graphs_due_to_env || disable_due_to_too_many_updates);
    }
};