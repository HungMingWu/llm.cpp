module;
#include <memory>

module ggml;
import :alloc;
import :host_buffer;
import :cpu.buffer_type;

struct cpu_host_backend_buffer : public host_backend_buffer_base {
public:
	using host_backend_buffer_base::host_backend_buffer_base;
	~cpu_host_backend_buffer() override {
		internal::free(context);
	}
};

std::unique_ptr<ggml_backend_buffer> cpu_backend_buffer_type::alloc_buffer_impl(size_t size) {
	void* data = internal::aligned_alloc(64, size);

	if (data == nullptr) {
		GGML_LOG_ERROR("{}: failed to allocate buffer of size {}", __func__, size);
		return nullptr;
	}

	return std::make_unique<cpu_host_backend_buffer>(this, size, data);
}
