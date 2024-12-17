module;
#include <stdint.h>
#include <string.h>

module ggml:host_buffer;
import :ds;
import :func;

template <auto free_func>
struct host_backend_buffer : public ggml_backend_buffer {
	void* context;
private:
	void* get_base_impl() override {
		uintptr_t data = (uintptr_t)context;

		// align the buffer
		if (data % TENSOR_ALIGNMENT != 0) {
			data = GGML_PAD(data, TENSOR_ALIGNMENT);
		}

		return (void*)data;
	}
public:
	host_backend_buffer(ggml_backend_buffer_type_t type, size_t size, void* context)
		: ggml_backend_buffer(type, size), context(context)
	{

	}

	~host_backend_buffer() override {
		free_func(context);
	}

	void memset_tensor(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) override
	{
		memset((char*)tensor->data + offset, value, size);
	}

	void set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override
	{
		memcpy((char*)tensor->data + offset, data, size);
	}

	void get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override
	{
		memcpy(data, (const char*)tensor->data + offset, size);
	}

	bool cpy_tensor(const ggml_tensor* src, ggml_tensor* dst) override
	{
		if (src->buffer->is_host()) {
			memcpy(dst->data, src->data, src->nbytes());
			return true;
		}
		return false;
	}

	void clear(uint8_t value) override
	{
		memset(context, value, size);
	}
};
