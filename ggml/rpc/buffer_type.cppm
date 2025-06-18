module;
#include <memory>
#include <string>

module ggml:rpc.buffer_type;
import :ds;

class ggml_rpc_buffer_type : public ggml_backend_buffer_type {
	std::string endpoint;
	std::string name;
	size_t alignment;
	size_t max_size;
protected:
	std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) override;
public:
	ggml_rpc_buffer_type(std::string endpoint, std::string name, size_t alignment, size_t max_size)
		: endpoint(std::move(endpoint)), name(std::move(name)), alignment(alignment), max_size(max_size) {
	}
	const char* get_name() override { return name.c_str(); }
	size_t get_alignment() override { return alignment; }
	size_t get_max_size() override { return max_size; }
	size_t get_alloc_size(const ggml_tensor* tensor) override;
	const std::string& get_endpoint() const { return endpoint; }
};