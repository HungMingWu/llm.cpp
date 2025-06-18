module;
#include <assert.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

module ggml;

import :rpc.backend;
import :rpc.buffer_type;
import :rpc.ds;
import :rpc.socket;

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char* endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        fprintf(stderr, "Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock);
    size_t max_size = get_max_size(sock);

    ggml_backend_buffer_type_t buft = new ggml_rpc_buffer_type(
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    );
    buft_map[endpoint] = buft;
    return buft;
}

// device interface
struct ggml_backend_rpc_device : public ggml_backend_device {
    std::string endpoint;
    std::string name;
public:
    ggml_backend_rpc_device(std::string endpoint, std::string name)
		: ggml_backend_device(nullptr), endpoint(std::move(endpoint)), name(std::move(name)) {
	}
    const char* get_name() override { return name.c_str(); }
    const char* get_description() override { return name.c_str(); }
    void get_memory(size_t* free, size_t* total) override {
        ggml_backend_rpc_get_device_memory(endpoint.c_str(), free, total);
    }
    enum ggml_backend_dev_type get_type() override {
        // TODO: obtain value from the server
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    void get_props(struct ggml_backend_dev_props* props) override {
        props->name = get_name();
        props->description = get_description();
        props->type = get_type();
        get_memory(&props->memory_free, &props->memory_total);
        props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ false,
            /* .buffer_from_host_ptr  = */ false,
            /* .events                = */ false,
        };
    }

    ggml_backend_t init_backend(const char* params) override {
        return new ggml_backend_rpc(this, endpoint, "RPC[" + std::string(endpoint) + "]");
    }
    ggml_backend_buffer_type_t get_buffer_type() override {
        return ggml_backend_rpc_buffer_type(endpoint.c_str());
    }

    bool supports_op(const ggml_tensor* op) override {
        //TODO: call the remote backend and cache the results
        return true;
    }

    bool supports_buft(ggml_backend_buffer_type_t buft) override {
        if (auto rpc_buft = dynamic_cast<ggml_rpc_buffer_type*>(buft)) {
            // Check if the buffer type is compatible with this device
            return rpc_buft->get_endpoint() == endpoint;
		}
        return false;
    }
};

ggml_backend_dev_t ggml_backend_rpc_add_device(const char* endpoint)
{
    static std::unordered_map<std::string, ggml_backend_dev_t> dev_map;

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (dev_map.find(endpoint) != dev_map.end()) {
        return dev_map[endpoint];
    }

    ggml_backend_dev_t dev = new ggml_backend_rpc_device(endpoint, "RPC[" + std::string(endpoint) + "]");
    dev_map[endpoint] = dev;

    return dev;
}
