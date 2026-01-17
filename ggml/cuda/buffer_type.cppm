module;
#include <string>
#include <memory>
#include "common.h"

module ggml:cuda.buffer_type;
import :cpu.device;
import :ds;

struct cuda_backend_buffer_type : public ggml_backend_buffer_type {
	int device;
	std::string name;
protected:
	std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) override;
public:
	using ggml_backend_buffer_type::ggml_backend_buffer_type;
	const char* get_name() override
	{
		return name.c_str();
	}
	size_t get_alignment() override
	{
		return 128;
	}

	size_t get_alloc_size(const ggml_tensor* tensor) override;
};

struct cuda_split_backend_buffer_type : ggml_backend_buffer_type {
	std::string name;
	int device;
	std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
protected:
	std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) override;
public:
	const char* get_name() override
	{
		return name.c_str();
	}
	size_t get_alignment() override
	{
		return 128;
	}

	size_t get_alloc_size(const ggml_tensor* tensor) override;
};

struct cuda_host_backend_buffer_type : public cpu_backend_buffer_type {
protected:
	std::unique_ptr<ggml_backend_buffer> alloc_buffer_impl(size_t size) override;
public:
	const char* get_name() override { return GGML_CUDA_NAME "_Host"; }
};

bool buffer_type_from_device(ggml_backend_buffer_type* buft, int device);