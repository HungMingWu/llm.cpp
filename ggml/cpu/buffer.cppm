module;
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <memory>

#define GGML_LOG_ERROR(...)
// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

export module ggml:cpu.buffer;
import :alloc;
import :ds;

constexpr size_t GGML_PAD2(size_t x, size_t n) {
	return (x + n - 1) & ~(n - 1);
}

struct cpu_backend_buffer : public ggml_backend_buffer {
	void* context;
public:
	cpu_backend_buffer(ggml_backend_buffer_type_t type, size_t size, void* context)
		: ggml_backend_buffer(type, size), context(context)
	{

	}

	~cpu_backend_buffer() override {
		// TODO
		internal::free(context);
	}

	void* get_base() override {
		uintptr_t data = (uintptr_t)context;

		// align the buffer
		if (data % TENSOR_ALIGNMENT != 0) {
			data = GGML_PAD2(data, TENSOR_ALIGNMENT);
		}

		return (void*)data;
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

export
{
	struct cpu_backend_buffer_type : public ggml_backend_buffer_type {
	public:
		const char* get_name() override { return "CPU"; }
		std::unique_ptr<ggml_backend_buffer> alloc_buffer(size_t size) override {
			void* data = internal::aligned_alloc(64, size);

			if (data == nullptr) {
				GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
				return nullptr;
			}

			return std::make_unique<cpu_backend_buffer>(this, size, data);
		}
		size_t get_alignment() override {
			return TENSOR_ALIGNMENT;
		}
		bool is_host() override {
			return true;
		}
	};
}
