module;
#include <stdlib.h>
#include <memory>

module ggml;
import os;

void ggml_backend_cpu_device::get_memory(size_t* free, size_t* total)
{
	::get_memory(free, total);
}

ggml_backend_t ggml_backend_cpu_device::init_backend(const char* params)
{
	return std::make_unique<ggml_cpu_backend>(this).release();
}