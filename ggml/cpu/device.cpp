module;
#include <stdlib.h>
#include <memory>

module ggml;
import os;
import :cpu.device;

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type()
{
	static cpu_backend_buffer_type type;
	return &type;
}

void ggml_backend_cpu_device::get_memory(size_t* free, size_t* total)
{
	::get_memory(free, total);
}

ggml_backend_buffer_type_t ggml_backend_cpu_device::get_buffer_type()
{
	return ggml_backend_cpu_buffer_type();
}

ggml_backend_t ggml_backend_cpu_device::init_backend(const char* params)
{
	return std::make_unique<ggml_cpu_backend>(this).release();
}
