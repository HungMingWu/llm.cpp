module;
#include <string.h>
#include <array>
#include <memory>
#include <vector>

module ggml;
import :rpc.buffer;
import :rpc.helper;
import :rpc.socket;

rpc_backend_buffer::~rpc_backend_buffer()
{
    auto request = std::make_shared<rpc_msg_free_buffer_req>();
    request->remote_ptr = remote_ptr;
    dispatcher->send(RPC_CMD_FREE_BUFFER, request, sizeof(*request));
}

void* rpc_backend_buffer::get_base_impl()
{
    if (base_ptr != nullptr) {
        return base_ptr;
    }
    auto request = std::make_shared<rpc_msg_buffer_get_base_req>();
    request->remote_ptr = remote_ptr;
    rpc_msg_buffer_get_base_rsp response;
    dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, sizeof(*request), &response, sizeof(response));
    base_ptr = reinterpret_cast<void*>(response.base_ptr);
    return base_ptr;
}

ggml_status rpc_backend_buffer::init_tensor(ggml_tensor* tensor)
{
    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        auto request = std::make_shared<rpc_msg_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        dispatcher->send(RPC_CMD_INIT_TENSOR, request, sizeof(*request));
    }
    return GGML_STATUS_SUCCESS;
}

void rpc_backend_buffer::set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > HASH_THRESHOLD) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    uint8_t* input = new uint8_t[input_size]();
    memcpy(input, &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input + sizeof(rpc_tensor) + sizeof(offset), data, size);
    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
    dispatcher->send_async(RPC_CMD_SET_TENSOR, input_ptr, input_size);
}

void rpc_backend_buffer::get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
{
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

bool rpc_backend_buffer::cpy_tensor(const ggml_tensor* src, ggml_tensor* dst)
{
    if (auto src_buffer = dynamic_cast<rpc_backend_buffer*>(src->buffer); src_buffer) {
        // check if src and dst are on the same server
        auto dst_buffer = dynamic_cast<rpc_backend_buffer*>(dst->buffer);
        if (src_buffer->dispatcher != dst_buffer->dispatcher) {
            return false;
        }
        auto request = std::make_shared<rpc_msg_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        dispatcher->send(RPC_CMD_COPY_TENSOR, request, sizeof(*request), &response, sizeof(response));
        return response.result;
    }
    return false;
}

void rpc_backend_buffer::clear_impl(uint8_t value)
{
    auto request = std::make_shared<rpc_msg_buffer_clear_req>();
    request->remote_ptr = remote_ptr;
    request->value = value;
    dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, sizeof(*request));
}

void rpc_backend_buffer::memset_tensor(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size)
{
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    request->value = value;
    dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request));
}
