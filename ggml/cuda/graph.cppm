module;
#include <vector>
#include "common.h"

module ggml:cuda.graph;
import :ds;
import :log;

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
    bool warmup_complete = false;
    uint64_t uid = 0;
    int64_t last_used_time = 0;
    struct node_properties {
        ggml_tensor node;
        void* node_src_data_ptrs[GGML_MAX_SRC];
        int64_t  node_src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
        size_t   node_src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
    };
    std::vector<node_properties> node_props;

    bool is_enabled() const {
        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
        return !(disable_due_to_gpu_arch || disable_cuda_graphs_due_to_env);
    }
};