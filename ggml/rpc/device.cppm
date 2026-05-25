module;
#include <memory>
#include <string>

module ggml:rpc.device;
import :ds;

// device interface
struct ggml_backend_rpc_device : public ggml_backend_device {
    std::string endpoint;
    uint32_t device;
    std::string name;
    std::string description;
    uint64_t    last_graph_uid{};
public:
    ggml_backend_rpc_device(ggml_backend_reg_t reg, std::string endpoint, uint32_t device, std::string name, std::string description);

    const char* get_name() override { return name.c_str(); }
    const char* get_description() override { return description.c_str(); }
    void get_memory(size_t* free, size_t* total) override;
    enum ggml_backend_dev_type get_type() override;
    void get_props(struct ggml_backend_dev_props* props) override;

    std::unique_ptr<ggml_backend> init_backend(const char* params) override;
    ggml_backend_buffer_type* get_buffer_type() override;
    bool supports_op(const ggml_tensor* op) override {
        //TODO: call the remote backend and cache the results
        return true;
    }

    bool supports_buft(ggml_backend_buffer_type* buft) override;
};