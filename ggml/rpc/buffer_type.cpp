module;
#include <memory>

module ggml;
import :rpc.buffer;
import :rpc.buffer_type;
import :rpc.ds;
import :rpc.helper;
import :rpc.socket;

std::unique_ptr<ggml_backend_buffer> ggml_rpc_buffer_type::alloc_buffer_impl(size_t size)
{
    rpc_msg_alloc_buffer_req request = { static_cast<uint32_t>(size) };
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
		return std::make_unique<rpc_backend_buffer>(this, sock, nullptr, response.remote_ptr, response.remote_size);
    }
    else {
        return nullptr;
    }
}

size_t ggml_rpc_buffer_type::get_alloc_size(const ggml_tensor* tensor)
{
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {;
        auto sock = get_socket(endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        // TODO: cache the alloc responses to avoid extra RPC calls?
        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        return response.alloc_size;
    }

    return tensor->nbytes();
}
