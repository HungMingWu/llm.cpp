module;
#include <assert.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

module ggml;

import :rpc.backend;
import :rpc.buffer_type;
import :rpc.ds;
import :rpc.socket;

ggml_backend_buffer_type* ggml_backend_rpc_buffer_type(const char* endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type*> buft_map;
    auto it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        fprintf(stderr, "Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);

    ggml_backend_buffer_type* buft = new ggml_rpc_buffer_type(
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
    uint32_t device;
    std::string name;
    std::string description;
public:
    ggml_backend_rpc_device(ggml_backend_reg_t reg, std::string endpoint, uint32_t device, std::string name, std::string description)
		: ggml_backend_device(reg),
        endpoint(std::move(endpoint)), device(device), name(std::move(name)), description(std::move(description)) 
    {
	}

    const char* get_name() override { return name.c_str(); }
    const char* get_description() override { return description.c_str(); }
    void get_memory(size_t* free, size_t* total) override {
        ggml_backend_rpc_get_device_memory(endpoint.c_str(), device,  free, total);
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

    std::unique_ptr<ggml_backend> init_backend(const char* params) override {
        return std::make_unique<ggml_backend_rpc>(this, device, endpoint, "RPC[" + std::string(endpoint) + "]");
    }
    ggml_backend_buffer_type* get_buffer_type() override {
        return ggml_backend_rpc_buffer_type(endpoint.c_str(), device);
    }

    bool supports_op(const ggml_tensor* op) override {
        //TODO: call the remote backend and cache the results
        return true;
    }

    bool supports_buft(ggml_backend_buffer_type* buft) override {
        if (auto rpc_buft = dynamic_cast<ggml_rpc_buffer_type*>(buft)) {
            // Check if the buffer type is compatible with this device
            return rpc_buft->get_endpoint() == endpoint;
		}
        return false;
    }
};

struct ggml_backend_rpc_reg : public ggml_backend_reg {
    std::string                     name;
    std::vector<ggml_backend_device*> devices;
public:
    using ggml_backend_reg::ggml_backend_reg;
	std::string_view get_name() override { return name; }
    size_t get_device_count() override { return devices.size(); }
    ggml_backend_device* get_device(size_t index) override {
        return devices.at(index);
    }
};

#define GGML_BACKEND_API_VERSION 2

ggml_backend_reg* ggml_backend_rpc_add_server(const char* endpoint)
{
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg* reg = new ggml_backend_rpc_reg(GGML_BACKEND_API_VERSION, nullptr);
    reg->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device* dev = new ggml_backend_rpc_device{
            reg, 
            /* .endpoint    = */ endpoint,
            /* .device      = */ ind,
            /* .name        = */ dev_name,
            /* .description = */ dev_desc
        };

        reg->devices.push_back(dev);
        dev_id++;
    }
    reg_map[endpoint] = reg;
    return reg;
}
