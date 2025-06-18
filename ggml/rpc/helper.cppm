module;
#include <stdint.h>
#include <string.h>

module ggml:rpc.helper;
import :rpc.ds;

rpc_tensor serialize_tensor(const ggml_tensor* tensor) {
    rpc_tensor result;
    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
#if 0
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context* ctx = (ggml_backend_rpc_buffer_context*)buffer->context;
        result.buffer = ctx->remote_ptr;
    }
    else {
        result.buffer = 0;
    }
#endif
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data = reinterpret_cast<uint64_t>(tensor->data);

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    tensor->name.copy(result.name, GGML_MAX_NAME - 1);
    return result;
}

// Computes FNV-1a hash of the data
uint64_t fnv_hash(const uint8_t* data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}