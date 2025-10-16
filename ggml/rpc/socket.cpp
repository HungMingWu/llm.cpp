module;
#include <string.h>
#include <memory>
#include <print>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif

module ggml;
import :ds;
import :rpc.server;
import :rpc.socket;

static bool send_msg(sockfd_t sockfd, const void* msg, size_t msg_size) {
    if (!send_data(sockfd, &msg_size, sizeof(msg_size))) {
        return false;
    }
    return send_data(sockfd, msg, msg_size);
}

static bool recv_msg(sockfd_t sockfd, void* msg, size_t msg_size) {
    uint64_t size;
    if (!recv_data(sockfd, &size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return recv_data(sockfd, msg, msg_size);
}

static bool recv_msg(sockfd_t sockfd, std::vector<uint8_t>& input) {
    uint64_t size;
    if (!recv_data(sockfd, &size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    }
    catch (const std::bad_alloc& e) {
        std::println(stderr, "Failed to allocate input buffer of size {}", size);
        return false;
    }
    return recv_data(sockfd, input.data(), size);
}

static void rpc_serve_client(std::span<std::unique_ptr<ggml_backend>> backends, const char* cache_dir,
    sockfd_t sockfd, const std::vector<size_t>& free_mem, const std::vector<size_t>& total_mem) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!recv_data(sockfd, &cmd, 1)) {
        return;
    }
    // the first command sent by the client must be HELLO
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }
    if (!recv_msg(sockfd, nullptr, 0)) {
        return;
    }
    rpc_msg_hello_rsp response;
    server.hello(response);
    if (!send_msg(sockfd, &response, sizeof(response))) {
        return;
    }
    while (true) {
        if (!recv_data(sockfd, &cmd, 1)) {
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
            if (!recv_msg(sockfd, nullptr, 0)) {
                return;
            }
            rpc_msg_device_count_rsp response;
            response.device_count = backends.size();
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_ALLOC_BUFFER: {
            rpc_msg_alloc_buffer_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_alloc_buffer_rsp response;
            if (!server.alloc_buffer(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_ALLOC_SIZE: {
            rpc_msg_get_alloc_size_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_alloc_size_rsp response;
            if (!server.get_alloc_size(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_ALIGNMENT: {
            rpc_msg_get_alignment_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_alignment_rsp response;
            if (!server.get_alignment(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_MAX_SIZE: {
            rpc_msg_get_max_size_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_get_max_size_rsp response;
            if (!server.get_max_size(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_BUFFER_GET_BASE: {
            rpc_msg_buffer_get_base_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_buffer_get_base_rsp response;
            if (!server.buffer_get_base(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_FREE_BUFFER: {
            rpc_msg_free_buffer_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            if (!server.free_buffer(request)) {
                return;
            }
            if (!send_msg(sockfd, nullptr, 0)) {
                return;
            }
            break;
        }
        case RPC_CMD_BUFFER_CLEAR: {
            rpc_msg_buffer_clear_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            if (!server.buffer_clear(request)) {
                return;
            }
            if (!send_msg(sockfd, nullptr, 0)) {
                return;
            }
            break;
        }
        case RPC_CMD_SET_TENSOR: {
            std::vector<uint8_t> input;
            if (!recv_msg(sockfd, input)) {
                return;
            }
            if (!server.set_tensor(input)) {
                return;
            }
            break;
        }
        case RPC_CMD_SET_TENSOR_HASH: {
            rpc_msg_set_tensor_hash_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_set_tensor_hash_rsp response;
            if (!server.set_tensor_hash(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_INIT_TENSOR: {
            rpc_msg_init_tensor_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            if (!server.init_tensor(request)) {
                return;
            }
            if (!send_msg(sockfd, nullptr, 0)) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_TENSOR: {
            rpc_msg_get_tensor_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            std::vector<uint8_t> response;
            if (!server.get_tensor(request, response)) {
                return;
            }
            if (!send_msg(sockfd, response.data(), response.size())) {
                return;
            }
            break;
        }
        case RPC_CMD_COPY_TENSOR: {
            rpc_msg_copy_tensor_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_copy_tensor_rsp response;
            if (!server.copy_tensor(request, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GRAPH_COMPUTE: {
            std::vector<uint8_t> input;
            if (!recv_msg(sockfd, input)) {
                return;
            }
            rpc_msg_graph_compute_rsp response;
            if (!server.graph_compute(input, response)) {
                return;
            }
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_DEVICE_MEMORY: {
            rpc_msg_get_device_memory_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            auto dev_id = request.device;
            if (dev_id >= backends.size()) {
                return;
            }
            rpc_msg_get_device_memory_rsp response;
            response.free_mem = free_mem[dev_id];
            response.total_mem = total_mem[dev_id];
#if 0
            LOG_DBG("[get_device_mem] device: {}, free_mem: {}, total_mem: {}\n", dev_id,
                response.free_mem, response.total_mem);
#endif
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        default: {
            GGML_LOG_ERROR("Unknown command: {}\n", cmd);
            return;
        }
        }
    }
}
static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char*)&flag, sizeof(int));
    return ret == 0;
}

static std::shared_ptr<socket_t> create_server_socket(const char* host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock = make_socket(sockfd);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        fprintf(stderr, "Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        fprintf(stderr, "Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return sock;
}

static std::shared_ptr<socket_t> socket_accept(sockfd_t srv_sockfd) {
    auto client_socket_fd = accept(srv_sockfd, NULL, NULL);
    auto client_socket = make_socket(client_socket_fd);
    if (client_socket == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return client_socket;
}

void ggml_backend_rpc_start_server(const char* endpoint, const char* cache_dir,
    size_t n_threads, size_t n_devices,
    ggml_backend_device** devices, size_t* free_mem, size_t* total_mem) {
    if (n_devices == 0 || devices == nullptr || free_mem == nullptr || total_mem == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<std::unique_ptr<ggml_backend>> backends;
    std::vector<size_t> free_mem_vec(free_mem, free_mem + n_devices);
    std::vector<size_t> total_mem_vec(total_mem, total_mem + n_devices);
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", dev->get_name(), dev->get_description(),
            total_mem[i] / 1024 / 1024, free_mem[i] / 1024 / 1024);
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
#ifdef _WIN32
    {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", res);
            return;
        }
    }
#endif
    auto server_socket = create_server_socket(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = socket_accept(server_socket->fd);
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket->fd, free_mem_vec, total_mem_vec);
        printf("Client connection closed\n");
        fflush(stdout);
    }
#ifdef _WIN32
    WSACleanup();
#endif
}
