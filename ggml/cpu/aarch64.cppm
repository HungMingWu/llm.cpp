module;
#include <stddef.h>
#include <memory>

export module ggml:cpu_aarch64;
import :ds;

struct aarch64_cpu_backend_buffer_type : public ggml_backend_buffer_type {
protected:
    std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) override
    {
#if 0
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

        if (buffer == nullptr) {
            return nullptr;
        }

        buffer->buft = buft;
        buffer->iface.init_tensor = ggml_backend_cpu_aarch64_buffer_init_tensor;
        buffer->iface.set_tensor = ggml_backend_cpu_aarch64_buffer_set_tensor;
        return buffer;
#else
        return nullptr;
#endif
    }
public:
    const char* get_name() override 
    {
        return "CPU_AARCH64";
    }
    size_t get_alignment() override
    {
        return TENSOR_ALIGNMENT;
    }
    bool supports_op(const ggml_tensor* op) override 
    {
        return false;
    }
    tensor_traits* get_tensor_traits(const ggml_tensor* op) override 
    {
        //TODO
#if 0
#else
        return nullptr; 
#endif
    }
};

export
{
	ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type() {
        static aarch64_cpu_backend_buffer_type type;
        return &type;
	}
}
