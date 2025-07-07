module;
#include <stdint.h>

module ggml;

void ggml_backend_buffer::clear(uint8_t value)
{
	// clear is optional if the buffer is zero-sized
	if (get_size() == 0) {
		return;
	}

	clear_impl(value);
}