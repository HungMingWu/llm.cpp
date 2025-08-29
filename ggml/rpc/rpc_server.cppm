module;
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

module ggml:rpc.server;
import :ds;
import :rpc.ds;

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(ggml_backend* backend, const char* cache_dir)
        : backend(backend), cache_dir(cache_dir) {
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp& response);
    void alloc_buffer(const rpc_msg_alloc_buffer_req& request, rpc_msg_alloc_buffer_rsp& response);
    void get_alignment(rpc_msg_get_alignment_rsp& response);
    void get_max_size(rpc_msg_get_max_size_rsp& response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req& request, rpc_msg_buffer_get_base_rsp& response);
    bool free_buffer(const rpc_msg_free_buffer_req& request);
    bool buffer_clear(const rpc_msg_buffer_clear_req& request);
    bool set_tensor(const std::vector<uint8_t>& input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req& request, rpc_msg_set_tensor_hash_rsp& response);
    bool get_tensor(const rpc_msg_get_tensor_req& request, std::vector<uint8_t>& response);
    bool copy_tensor(const rpc_msg_copy_tensor_req& request, rpc_msg_copy_tensor_rsp& response);
    bool graph_compute(const std::vector<uint8_t>& input, rpc_msg_graph_compute_rsp& response);
    bool init_tensor(const rpc_msg_init_tensor_req& request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req& request, rpc_msg_get_alloc_size_rsp& response);

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t>& data);
    ggml_tensor* deserialize_tensor(struct ggml_context* ctx, const rpc_tensor* tensor);
    ggml_tensor* create_node(uint64_t id,
        struct ggml_context* ctx,
        const std::unordered_map<uint64_t, const rpc_tensor*>& tensor_ptrs,
        std::unordered_map<uint64_t, struct ggml_tensor*>& tensor_map);


    ggml_backend* backend;
    const char* cache_dir;
    std::unordered_set<ggml_backend_buffer*> buffers;
};
