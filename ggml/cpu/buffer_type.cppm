module;
#include <memory>

export module ggml:cpu.buffer_type;
import :buffer_type;

export
{
	struct cpu_backend_buffer_type : public ggml_backend_buffer_type {
	protected:
		std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) override;
	public:
		const char* get_name() override { return "CPU"; }
		size_t get_alignment() override {
			return TENSOR_ALIGNMENT;
		}
		bool is_host() override {
			return true;
		}
	};

	struct cpu_backend_buffer_from_ptr_type : public cpu_backend_buffer_type {
	public:
		const char* get_name() override { return "CPU_Mapped"; }
	};
}