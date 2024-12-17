module;
#include <stdlib.h>

module ggml;
import os;

void ggml_backend_cpu_device::get_memory(size_t* free, size_t* total)
{
	::get_memory(free, total);
}