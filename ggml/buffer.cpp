module;
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <vector>
#include <utility>

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

void* ggml_backend_buffer::get_base() {
	// get_base is optional if the buffer is zero-sized
	if (size == 0) {
		return nullptr;
	}

	void* base = get_base_impl();
	//GGML_ASSERT(base != nullptr && "backend buffer base cannot be nullptr");
	return base;
}

multi_backend_buffer::multi_backend_buffer(
	ggml_backend_buffer_type* buft, size_t size, std::vector<std::unique_ptr<ggml_backend_buffer>> buffers)
	: ggml_backend_buffer(buft, size),
	buffers(std::move(buffers))
{
}

void multi_backend_buffer::clear_impl(uint8_t value)
{
	for (auto& buffer : buffers)
		buffer->clear(value);
}
