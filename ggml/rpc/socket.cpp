module;
#include <string.h>
#include <memory>
#include <print>
#include <string>
#include <vector>

module ggml;
import :ds;
import :rpc.server;
import :rpc.socket;

static bool send_msg(socket_ptr sock, const void* msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void* msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t>& input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    }
    catch (const std::bad_alloc& e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size {}", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static void rpc_serve_client(std::span<std::unique_ptr<ggml_backend>> backends, const char* cache_dir,
    socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) ¡X client needs upgrade to protocol v%d.x\n",
            (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
        case RPC_CMD_HELLO: {
            // HELLO command is handled above
            return;
        }
        case RPC_CMD_DEVICE_COUNT: {
            if (!recv_msg(sock, nullptr, 0)) {
                return;
            }
            rpc_msg_device_count_rsp response;
            response.device_count = backends.size();
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_ALLOC_BUFFER: {
            rpc_msg_alloc_buffer_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_alloc_buffer_rsp response;
            if (!server.alloc_buffer(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_ALLOC_SIZE: {
            rpc_msg_get_alloc_size_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_alloc_size_rsp response;
            if (!server.get_alloc_size(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_ALIGNMENT: {
            rpc_msg_get_alignment_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_alignment_rsp response;
            if (!server.get_alignment(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_MAX_SIZE: {
            rpc_msg_get_max_size_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_max_size_rsp response;
            if (!server.get_max_size(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_BUFFER_GET_BASE: {
            rpc_msg_buffer_get_base_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_buffer_get_base_rsp response;
            if (!server.buffer_get_base(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_FREE_BUFFER: {
            rpc_msg_free_buffer_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            if (!server.free_buffer(request)) {
                return;
            }
            if (!send_msg(sock, nullptr, 0)) {
                return;
            }
            break;
        }
        case RPC_CMD_BUFFER_CLEAR: {
            rpc_msg_buffer_clear_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            if (!server.buffer_clear(request)) {
                return;
            }
            if (!send_msg(sock, nullptr, 0)) {
                return;
            }
            break;
        }
        case RPC_CMD_SET_TENSOR: {
            std::vector<uint8_t> input;
            if (!recv_msg(sock, input)) {
                return;
            }
            if (!server.set_tensor(input)) {
                return;
            }
            break;
        }
        case RPC_CMD_SET_TENSOR_HASH: {
            rpc_msg_set_tensor_hash_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_set_tensor_hash_rsp response;
            if (!server.set_tensor_hash(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_INIT_TENSOR: {
            rpc_msg_init_tensor_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            if (!server.init_tensor(request)) {
                return;
            }
            if (!send_msg(sock, nullptr, 0)) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_TENSOR: {
            rpc_msg_get_tensor_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            std::vector<uint8_t> response;
            if (!server.get_tensor(request, response)) {
                return;
            }
            if (!send_msg(sock, response.data(), response.size())) {
                return;
            }
            break;
        }
        case RPC_CMD_COPY_TENSOR: {
            rpc_msg_copy_tensor_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_copy_tensor_rsp response;
            if (!server.copy_tensor(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GRAPH_COMPUTE: {
            std::vector<uint8_t> input;
            if (!recv_msg(sock, input)) {
                return;
            }
            if (!server.graph_compute(input)) {
                return;
            }
            break;
        }
        case RPC_CMD_GRAPH_RECOMPUTE: {
            rpc_msg_graph_recompute_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            if (!server.graph_recompute(request)) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_DEVICE_MEMORY: {
            rpc_msg_get_device_memory_req request;
            if (!recv_msg(sock, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_device_memory_rsp response;
            if (!server.get_device_memory(request, response)) {
                return;
            }
            if (!send_msg(sock, &response, sizeof(response))) {
                return;
            }
            break;
        }
        default: {
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            return;
        }
        }
    }
}

void ggml_backend_rpc_start_server(const char* endpoint, const char* cache_dir,
    size_t n_threads, size_t n_devices, ggml_backend_device** devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<std::unique_ptr<ggml_backend>> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        dev->get_memory(&free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", dev->get_name(), dev->get_description(),
            total / 1024 / 1024, free / 1024 / 1024);
        auto backend = dev->init_backend(nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->get_name());
            return;
        }
        if (auto cpu_backend = dynamic_cast<ggml_cpu_backend*>(backend.get())) {
            cpu_backend->set_n_threads(n_threads);
        }
        backends.push_back(std::move(backend));
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
}
