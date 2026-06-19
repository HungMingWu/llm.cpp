module;
#include <memory>
#include <string>

module ggml:cuda.device;
import :ds;

class ggml_backend_cuda_device : public ggml_backend_device {
public:
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
    using ggml_backend_device::ggml_backend_device;
    const char* get_name() override { return name.c_str(); }
    const char* get_description() override { return description.c_str(); }
    void get_memory(size_t* free, size_t* total) override;
    ggml_backend_dev_type get_type() override;
    void get_props(ggml_backend_dev_props* props) override;

    std::unique_ptr<ggml_backend> init_backend(const char*) override;
    ggml_backend_buffer_type* get_buffer_type() override;
    ggml_backend_buffer_type* get_host_buffer_type() override;
    bool supports_op(const ggml_tensor* op) override;
    bool supports_buft(ggml_backend_buffer_type* buft) override;
    bool offload_op(const ggml_tensor* op) override;
    ggml_backend_event* event_new() override;
    void event_free(ggml_backend_event* event) override;
    void event_synchronize(ggml_backend_event* event) override;
};