module;
#include <memory>
#include <print>
#include <string>

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

static void rpc_serve_client(ggml_backend_t backend, const char* cache_dir,
    sockfd_t sockfd, size_t free_mem, size_t total_mem) {
    rpc_server server(backend, cache_dir);
    uint8_t cmd;
    if (!recv_data(sockfd, &cmd, 1)) {
        return;
    }
    // the first command sent by the client must be HELLO
    if (cmd != RPC_CMD_HELLO) {
        std::println(stderr, "Expected HELLO command, update client");
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
            std::println(stderr, "Unknown command: {}", cmd);
            break;
        }
        switch (cmd) {
        case RPC_CMD_HELLO: {
            // HELLO command is handled above
            return;
        }
        case RPC_CMD_ALLOC_BUFFER: {
            rpc_msg_alloc_buffer_req request;
            if (!recv_msg(sockfd, &request, sizeof(request))) {
                return;
            }
            rpc_msg_alloc_buffer_rsp response;
            server.alloc_buffer(request, response);
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
            if (!recv_msg(sockfd, nullptr, 0)) {
                return;
            }
            rpc_msg_get_alignment_rsp response;
            server.get_alignment(response);
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        case RPC_CMD_GET_MAX_SIZE: {
            if (!recv_msg(sockfd, nullptr, 0)) {
                return;
            }
            rpc_msg_get_max_size_rsp response;
            server.get_max_size(response);
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
            if (!recv_msg(sockfd, nullptr, 0)) {
                return;
            }
            rpc_msg_get_device_memory_rsp response;
            response.free_mem = free_mem;
            response.total_mem = total_mem;
            if (!send_msg(sockfd, &response, sizeof(response))) {
                return;
            }
            break;
        }
        default: {
            std::println(stderr, "Unknown command: {}", cmd);
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

void ggml_backend_rpc_start_server(ggml_backend_t backend, const char* endpoint,
    const char* cache_dir,
    size_t free_mem, size_t total_mem) {
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("  backend memory : %zu MB\n", free_mem / (1024 * 1024));

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
        printf("Accepted client connection, free_mem=%zu, total_mem=%zu\n", free_mem, total_mem);
        fflush(stdout);
        rpc_serve_client(backend, cache_dir, client_socket->fd, free_mem, total_mem);
        printf("Client connection closed\n");
        fflush(stdout);
    }
#ifdef _WIN32
    WSACleanup();
#endif
}
