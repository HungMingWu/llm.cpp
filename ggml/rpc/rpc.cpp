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
import :rpc.device;
import :rpc.ds;
import :rpc.socket;

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
    ggml_backend_rpc_reg* reg = new ggml_backend_rpc_reg();
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
