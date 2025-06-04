module;
#include <assert.h>
#include <memory>
#include "common.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_LOG_ERROR(...)
#define GGML_LOG_DEBUG(...)

module ggml;
import :host_buffer;
import :cuda.buffer_type;

static void* ggml_cuda_host_malloc(size_t size) {
	if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
		return nullptr;
	}

	void* ptr = nullptr;
	cudaError_t err = cudaMallocHost((void**)&ptr, size);
	if (err != cudaSuccess) {
		// clear the error
		(void)cudaGetLastError();
		GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
			size / 1024.0 / 1024.0, cudaGetErrorString(err));
		return nullptr;
	}

	return ptr;
}

static void ggml_cuda_host_free(void* ptr) {
	CUDA_CHECK(cudaFreeHost(ptr));
}

std::unique_ptr<ggml_backend_buffer> cuda_backend_buffer_type::alloc_buffer_impl(size_t size)
{
	ggml_cuda_set_device(device);

	void* dev_ptr;
	cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, device);
	if (err != cudaSuccess) {
		// clear the error
		cudaGetLastError();
		GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
		return nullptr;
	}

	return std::make_unique<cuda_backend_buffer>(this, size, device, dev_ptr);
}

size_t cuda_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor)
{
	size_t size = tensor->nbytes();
	int64_t ne0 = tensor->ne[0];

	if (ggml_is_quantized(tensor->type)) {
		if (ne0 % MATRIX_ROW_PADDING != 0) {
			GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
			size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
		}
	}

	return size;
}

std::unique_ptr<ggml_backend_buffer> cuda_split_backend_buffer_type::alloc_buffer_impl(size_t size)
{
	// since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
	// instead, we allocate them for each tensor separately in init_tensor
	// however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
	// as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.

	return std::make_unique<cuda_split_backend_buffer>(this, size);
}

size_t cuda_split_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor)
{
	GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

	size_t total_size = 0;

	const int64_t ne0 = tensor->ne[0];

	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		const auto [row_low, row_high] = get_row_split(ggml_nrows(tensor), tensor_split, id);

		int64_t nrows_split = row_high - row_low;
		if (nrows_split == 0) {
			continue;
		}

		total_size += ggml_nbytes_split(tensor, nrows_split);

		// pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
		if (ne0 % MATRIX_ROW_PADDING != 0) {
			total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
		}
	}

	return total_size;
}

std::unique_ptr<ggml_backend_buffer> cuda_host_backend_buffer_type::alloc_buffer_impl(size_t size)
{
	void* ptr = ggml_cuda_host_malloc(size);

	if (ptr == nullptr) {
		// fallback to cpu buffer
		return cpu_backend_buffer_type::alloc_buffer(size);
	}

	return std::make_unique<host_backend_buffer<ggml_cuda_host_free>>(this, size, ptr);
}

bool buffer_type_from_device(ggml_backend_buffer_type_t buft, int device)
{
	if (auto cuda_buffer_type = dynamic_cast<cuda_backend_buffer_type*>(buft)) {
		return cuda_buffer_type->device == device;
	}
	if (auto cuda_split_buffer_type = dynamic_cast<cuda_split_backend_buffer_type*>(buft)) {
		return cuda_split_buffer_type->device == device;
	}
	const bool integrated = ggml_cuda_info().devices[device].integrated;
	return integrated && (dynamic_cast<cuda_host_backend_buffer_type*>(buft) != nullptr);
}