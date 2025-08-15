module;
#include <string.h>
#include <memory>
#include <mutex>
#include <print>
#include <string>
#include <unordered_map>

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

#define GGML_ABORT(...)
#define GGML_PRINT_DEBUG(...)

#define GGML_RPC_MAX_SERVERS       16
#define GGML_UNUSED(x) (void)(x)

module ggml:rpc.socket;
import :rpc.ds;
import :log;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

// cross-platform socket
struct socket_t {
    sockfd_t fd;
    socket_t(sockfd_t fd) : fd(fd) {}
    ~socket_t() {
        GGML_PRINT_DEBUG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// function for nicer error messages on server crash
void RPC_STATUS_ASSERT(bool x) {
    if (!x) GGML_ABORT("Remote RPC server crashed or returned malformed response");
}

bool send_data(sockfd_t sockfd, const void* data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(sockfd, (const char*)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            GGML_LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool recv_data(sockfd_t sockfd, void* data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(sockfd, (char*)data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            GGML_LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            GGML_LOG_ERROR("recv returned 0 (peer closed?)\n");
            return false;
        }
        bytes_recv += (size_t)n;
    }
    return true;
}


// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
bool send_rpc_cmd(const std::shared_ptr<socket_t>& sock, enum rpc_cmd cmd, const void* input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!send_data(sock->fd, &cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!send_data(sock->fd, &input_size, sizeof(input_size))) {
        return false;
    }
    if (!send_data(sock->fd, input, input_size)) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
bool send_rpc_cmd(const std::shared_ptr<socket_t>& sock, enum rpc_cmd cmd, const void* input, size_t input_size, void* output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    // TODO: currently the output_size is always known, do we need support for commands with variable output size?
    // even if we do, we can skip sending output_size from the server for commands with known output size
    uint64_t out_size;
    if (!recv_data(sock->fd, &out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!recv_data(sock->fd, output, output_size)) {
        return false;
    }
    return true;
}

bool check_server_version(const std::shared_ptr<socket_t>& sock) {
    rpc_msg_hello_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        std::println(stderr, "RPC server version mismatch: {}.{}.{}", response.major, response.minor, response.patch);
        return false;
    }
    if (response.minor != RPC_PROTO_MINOR_VERSION || response.patch != RPC_PROTO_PATCH_VERSION) {
        std::println(stderr, "WARNING: RPC server version mismatch: {}.{}.{}", response.major, response.minor, response.patch);
    }
    return true;
}

bool parse_endpoint(const std::string& endpoint, std::string& host, int& port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
}

std::shared_ptr<socket_t> make_socket(sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<socket_t>(fd);
}

bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));
    return ret == 0;
}

std::shared_ptr<socket_t> socket_connect(const char* host, int port) {
    struct sockaddr_in addr;
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock_ptr = make_socket(sockfd);
    if (sock_ptr == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent* server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (connect(sock_ptr->fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return sock_ptr;
}

std::shared_ptr<socket_t> get_socket(const std::string& endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;
    static bool initialized = false;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
#ifdef _WIN32
    if (!initialized) {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            return nullptr;
        }
        initialized = true;
    }
#else
    GGML_UNUSED(initialized);
#endif
    auto sock = socket_connect(host.c_str(), port);
    if (!sock) {
        return nullptr;
    }
    if (!check_server_version(sock)) {
        return nullptr;
    }
    GGML_PRINT_DEBUG("[%s] connected to %s, sockfd=%d\n", __func__, endpoint.c_str(), sock->fd);
    sockets[endpoint] = sock;
    return sock;
}

void get_device_memory(const std::shared_ptr<socket_t>& sock, size_t* free, size_t* total) {
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char* endpoint, size_t* free, size_t* total) {
    auto sock = get_socket(endpoint);
    if (!sock) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, free, total);
}

size_t get_alignment(const std::shared_ptr<socket_t>& sock) {
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

size_t get_max_size(const std::shared_ptr<socket_t>& sock) {
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

void ggml_backend_rpc_start_server(ggml_backend_t backend, const char* endpoint,
    const char* cache_dir,
    size_t free_mem, size_t total_mem);
