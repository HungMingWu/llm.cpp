module;
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

module ggml;
import :rpc.backend;
import :rpc.buffer_type;
import :rpc.device;
import :rpc.socket;

ggml_backend_buffer_type* ggml_backend_rpc_buffer_type(const char* endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type*> buft_map;
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto it = buft_map.find(buft_map);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_buffer_type* buft = new ggml_rpc_buffer_type(
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    );
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_rpc_device::ggml_backend_rpc_device(ggml_backend_reg_t reg, std::string endpoint, uint32_t device, std::string name, std::string description)
    : ggml_backend_device(reg),
    endpoint(std::move(endpoint)), device(device), name(std::move(name)), description(std::move(description))
{
}

void ggml_backend_rpc_device::get_memory(size_t* free, size_t* total)
{
    ggml_backend_rpc_get_device_memory(endpoint.c_str(), device, free, total);
}

enum ggml_backend_dev_type ggml_backend_rpc_device::get_type()
{
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void ggml_backend_rpc_device::get_props(struct ggml_backend_dev_props* props)
{
    props->name = get_name();
    props->description = get_description();
    props->type = get_type();
    get_memory(&props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

void ggml_backend_rpc_device::event_free(ggml_backend_event* event) {
    auto dispatcher = get_dispatcher(endpoint);
    dispatcher->event_free(event);
}

void ggml_backend_rpc_device::event_synchronize(ggml_backend_event* event) {
    auto dispatcher = get_dispatcher(endpoint);
    dispatcher->event_synchronize(event);
}

std::unique_ptr<ggml_backend> ggml_backend_rpc_device::init_backend(const char* params)
{
    return std::make_unique<ggml_backend_rpc>(this, device, endpoint, "RPC[" + std::string(endpoint) + "]");
}

ggml_backend_buffer_type* ggml_backend_rpc_device::get_buffer_type()
{
    return ggml_backend_rpc_buffer_type(endpoint.c_str(), device);
}

bool ggml_backend_rpc_device::supports_buft(ggml_backend_buffer_type* buft)
{
    if (auto rpc_buft = dynamic_cast<ggml_rpc_buffer_type*>(buft)) {
        // Check if the buffer type is compatible with this device
        return rpc_buft->get_endpoint() == endpoint;
    }
    return false;
}

ggml_backend_event* ggml_backend_rpc_device::event_new( ) {
    auto dispatcher = get_dispatcher(endpoint);
    return dispatcher->event_new(dev);
}
