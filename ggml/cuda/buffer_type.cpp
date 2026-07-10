module;
#include <assert.h>
#include <memory>
#include "common.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :host_buffer;
import :cuda.buffer;
import :cuda.buffer_type;
import :cuda.device;
import :cuda.utils;

static void* ggml_cuda_host_malloc(size_t size) {
	if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
		return nullptr;
	}

	void* ptr = nullptr;
	cudaError_t err = cudaMallocHost((void**)&ptr, size);
	if (err != cudaSuccess) {
		// clear the error
		(void)cudaGetLastError();
		GGML_LOG_DEBUG("{}: failed to allocate {:.2} MiB of pinned memory: {}", __func__,
			size / 1024.0 / 1024.0, cudaGetErrorString(err));
		return nullptr;
	}

	return ptr;
}

std::unique_ptr<ggml_backend_buffer> cuda_backend_buffer_type::alloc_buffer_impl(size_t size)
{
	ggml_cuda_set_device(device);

	void* dev_ptr;
	cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, device);
	if (err != cudaSuccess) {
		// clear the error
		cudaGetLastError();
		GGML_LOG_ERROR("{}: allocating {:.2} MiB on device {}: cudaMalloc failed: {}", __func__, size / 1024.0 / 1024.0, device, cudaGetErrorString(err));
		return nullptr;
	}

	return std::make_unique<cuda_backend_buffer>(this, size, device, dev_ptr);
}

size_t cuda_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor)
{
	size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
		? utils::ggml_cuda_flash_attn_ext_get_alloc_size(device, tensor)
		: tensor->nbytes();
	int64_t ne0 = tensor->ne[0];

	if (ggml_is_quantized(tensor->type)) {
		if (ne0 % MATRIX_ROW_PADDING != 0) {
			GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
			size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
		}
	}

	return size;
}

struct cuda_host_buffer : public host_backend_buffer_base {
public:
	using host_backend_buffer_base::host_backend_buffer_base;
	~cuda_host_buffer() override {
		CUDA_CHECK(cudaFreeHost(context));
	}
};

std::unique_ptr<ggml_backend_buffer> cuda_host_backend_buffer_type::alloc_buffer_impl(size_t size)
{
	void* ptr = ggml_cuda_host_malloc(size);

	if (ptr == nullptr) {
		// fallback to cpu buffer
		return cpu_backend_buffer_type::alloc_buffer(size);
	}

	return std::make_unique<cuda_host_buffer>(this, size, ptr);
}