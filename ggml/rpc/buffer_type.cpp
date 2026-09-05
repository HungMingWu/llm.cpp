module;
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

module ggml;
import :rpc.buffer;
import :rpc.buffer_type;
import :rpc.ds;
import :rpc.helper;
import :rpc.socket;

std::unique_ptr<ggml_backend_buffer> ggml_rpc_buffer_type::alloc_buffer_impl(size_t size)
{
    auto request = std::make_shared<rpc_msg_alloc_buffer_req>();
    request->device = device;
    request->size = size;
    rpc_msg_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, sizeof(*request), &response, sizeof(response));
    if (response.remote_ptr != 0) {
        return std::make_unique<rpc_backend_buffer>(this, dispatcher, nullptr, response.remote_ptr, response.remote_size);
    }
    else {
        return nullptr;
    }
}

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
    case GGML_OP_FLASH_ATTN_EXT:
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_CUMSUM:
    case GGML_OP_ARGSORT:
    case GGML_OP_TOP_K:
        return true;
    default:
        return false;
    }
}

size_t ggml_rpc_buffer_type::get_alloc_size(const ggml_tensor* tensor)
{
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        struct alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t*)&key, sizeof(key));
        cache_hash = fnv_hash((const uint8_t*)endpoint.data(), endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            if (it != cache.end()) {
                return it->second;
            }
        }

        auto request = std::make_shared<rpc_msg_get_alloc_size_req>();
        request->device = device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        rpc_msg_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, sizeof(*request), &response, sizeof(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        return response.alloc_size;
    }

    return tensor->nbytes();
}