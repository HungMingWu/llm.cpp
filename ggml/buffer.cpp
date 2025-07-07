module;
#include <assert.h>
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

ggml_status ggml_backend_buffer::alloc(ggml_tensor* tensor, void* addr)
{
	assert(tensor->buffer == nullptr);
	assert(tensor->data == nullptr);
	assert(tensor->view_src == nullptr);
	assert(addr >= get_base());
	assert((char*)addr + get_alloc_size(tensor) <=
		(char*)get_base() + get_size());

	tensor->buffer = this;
	tensor->data = addr;
	return init_tensor(tensor);
}