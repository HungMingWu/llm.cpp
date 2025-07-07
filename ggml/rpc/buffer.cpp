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
    rpc_msg_free_buffer_req request = { remote_ptr };
    bool status = send_rpc_cmd(sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

void* rpc_backend_buffer::get_base_impl()
{
    if (base_ptr != nullptr) {
        return base_ptr;
    }
    rpc_msg_buffer_get_base_req request = { remote_ptr };
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    base_ptr = reinterpret_cast<void*>(response.base_ptr);
    return base_ptr;
}

ggml_status rpc_backend_buffer::init_tensor(ggml_tensor* tensor)
{
    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

void rpc_backend_buffer::set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
}

void rpc_backend_buffer::get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
{
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

bool rpc_backend_buffer::cpy_tensor(const ggml_tensor* src, ggml_tensor* dst)
{
    // check if src and dst are on the same server
    auto src_buffer = dynamic_cast<rpc_backend_buffer*>(src->buffer);
    auto dst_buffer = dynamic_cast<rpc_backend_buffer*>(dst->buffer);
    if (!src_buffer || !dst_buffer || src_buffer->sock != dst_buffer->sock) {
        return false;
    }
    rpc_msg_copy_tensor_req request;
    request.src = serialize_tensor(src);
    request.dst = serialize_tensor(dst);
    rpc_msg_copy_tensor_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.result;
}

void rpc_backend_buffer::clear_impl(uint8_t value)
{
    rpc_msg_buffer_clear_req request = { remote_ptr, value };
    bool status = send_rpc_cmd(sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}
