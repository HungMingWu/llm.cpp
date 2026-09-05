module;
#include <string.h>
#include <future>
#include <memory>
#include <mutex>
#include <print>
#include <queue>
#include <string>
#include <unordered_map>

#define GGML_ABORT(...)
#define GGML_PRINT_DEBUG(...)

#define GGML_RPC_MAX_SERVERS       16

static const char* RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

module ggml:rpc.socket;
import :rpc.ds;
import :rpc.transport;
import :log;

// function for nicer error messages on server crash
void RPC_STATUS_ASSERT(bool x) {
    if (!x) GGML_ABORT("Remote RPC server crashed or returned malformed response");
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void* input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void* input, size_t input_size, void* output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t>& sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: {}.{}.{}",
            response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
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

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted) {
            return false;
        }
        queue.push(value);
        cvar.notify_all();
        return true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return !queue.empty() || interrupted; });
        if (interrupted) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

struct rpc_msg {
    rpc_cmd                       cmd;
    std::shared_ptr<const void>   input;
    size_t                        input_size;
    void* output;
    size_t                        output_size;
    std::promise<void>            completion;
};

using rpc_msg_ptr = std::shared_ptr<rpc_msg>;
using rpc_msg_queue = message_queue<rpc_msg_ptr>;

struct rpc_event : public ggml_backend_event {
    rpc_msg_ptr              msg;
    std::shared_future<void> sf;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void* output, size_t output_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void* output, size_t output_size);

    ggml_backend_event* event_new(ggml_backend_device* dev);
    void event_free(ggml_backend_event* event);
    void event_synchronize(ggml_backend_event* event);
    void event_record(ggml_backend_event* event);
    void synchronize();

    void start(const std::string& endpoint);
    void work();

    ~rpc_dispatcher();

private:
    rpc_msg_queue    queue;
    socket_ptr       sock;
    std::atomic_bool running;
    std::thread      thread;
};

std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string& endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    if (it != dispatchers.end()) {
        if (auto dispatcher = it->second.lock()) {
            return dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    return dispatcher;
}

void ggml_backend_rpc_get_device_memory(const char* endpoint, uint32_t device, size_t* free, size_t* total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_msg_get_device_memory_req>();
    request->device = device;
    rpc_msg_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, sizeof(*request), &response, sizeof(response));
    *free = response.free_mem;
    *total = response.total_mem;
}

size_t get_alignment(const std::shared_ptr<rpc_dispatcher>& dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_alignment_req>();
    request->device = device;
    rpc_msg_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, sizeof(*request), &response, sizeof(response));
    return response.alignment;
}

size_t get_max_size(const std::shared_ptr<rpc_dispatcher>& dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_max_size_req>();
    request->device = device;
    rpc_msg_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, sizeof(*request), &response, sizeof(response));
    return response.max_size;
}

uint32_t get_device_count(const char* endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    return response.device_count;
}

void ggml_backend_rpc_start_server(const char* endpoint, const char* cache_dir,
    size_t n_threads, size_t n_devices, ggml_backend_device** devices);
