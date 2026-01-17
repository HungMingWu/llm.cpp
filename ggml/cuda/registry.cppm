module;
#include <span>
#include <string_view>
#include <vector>

module ggml:cuda.registry;
import :ds;
import :log;
import :tensor;
import :cuda.buffer_type;
import :cuda.device;

class backend_cuda_reg : public ggml_backend_reg {
    std::vector<ggml_backend_cuda_device*> devices;
public:
    backend_cuda_reg(int api_version, void* context);

    std::string_view get_name() override;

    ggml_backend_device* get_device(size_t index) override {
        return devices.at(index);
    }

	void* get_proc_address(std::string_view name) override {
#if 0
        if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
            return (void*)ggml_backend_cuda_split_buffer_type;
        }
        if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
            return (void*)ggml_backend_cuda_register_host_buffer;
        }
        if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
            return (void*)ggml_backend_cuda_unregister_host_buffer;
        }
		return nullptr;
#else
		return nullptr;
#endif
	}
    std::span<const ggml_backend_feature> get_features() override;
};

ggml_backend_reg_t ggml_backend_cuda_reg();