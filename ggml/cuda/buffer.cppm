module;
#include <assert.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "common.h"

#define GGML_ASSERT(...)

module ggml:cuda.buffer;
import :cuda.buffer_type;
import :ds;
import :tensor;
import :traits;

cudaError_t ggml_cuda_device_malloc(void** ptr, size_t size, int device) {
	ggml_cuda_set_device(device);
	cudaError_t err;
	if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
		err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
		if (err == hipSuccess) {
			// hipMemAdviseSetCoarseGrain is an optional performance hint;
			// ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
			(void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
			(void)hipGetLastError(); // clear any error
		}

		// fall back to cudaMalloc if not supported (e.g. on Windows)
		if (err == hipErrorNotSupported) {
			static bool warned_unsupported = false;
			if (!warned_unsupported) {
				GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.");
				warned_unsupported = true;
			}

			err = cudaMalloc(ptr, size);
		}
#endif // defined(GGML_USE_HIP)
	}
	else {
		err = cudaMalloc(ptr, size);
	}
	return err;
}

struct cuda_backend_buffer : public ggml_backend_buffer {
	int device;
	void* dev_ptr = nullptr;
	std::string name;
private:
	void* get_base_impl() override
	{
		return dev_ptr;
	}
	void clear_impl(uint8_t value) override
	{
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemsetAsync(dev_ptr, value, size, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}
public:
	cuda_backend_buffer(
		ggml_backend_buffer_type* type,
		size_t size,
		int device,
		void* context)
		: ggml_backend_buffer(type, size),
		  device(device),
		  dev_ptr(context)
	{

	}

	~cuda_backend_buffer() override;

	ggml_status init_tensor(ggml_tensor* tensor) override
	{
		if (tensor->view_src != nullptr) {
			assert(tensor->view_src->buffer->get_type() == get_type());
			return GGML_STATUS_SUCCESS;
		}

		if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && usage != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
			// initialize padding to 0 to avoid possible NaN values
			const size_t original_size = tensor->nbytes();
			const size_t padded_size = get_alloc_size(tensor);

			if (padded_size > original_size) {
				ggml_cuda_set_device(device);
				CUDA_CHECK(cudaMemset((char*)tensor->data + original_size, 0, padded_size - original_size));
			}
		}
		return GGML_STATUS_SUCCESS;
	}

	void memset_tensor(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) override
	{
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemsetAsync((char*)tensor->data + offset, value, size, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}

	void set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override
	{
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}

	void get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override
	{
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemcpyAsync(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}

	void set_tensor_2d(ggml_tensor* tensor, const void* data,
		size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) override {
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemcpy2DAsync(
			(char*)tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}

	void get_tensor_2d(const struct ggml_tensor* tensor, void* data,
		size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) override {
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemcpy2DAsync(
			data, stride_data, (const char*)tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}

	bool cpy_tensor(const ggml_tensor* src, ggml_tensor* dst) override;
};

struct ggml_tensor_extra_gpu {
	void* data_device[GGML_CUDA_MAX_DEVICES]{}; // 1 pointer for each device for split tensors
	cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]{}; // events for synchronizing multiple GPUs
	ggml_tensor_extra_gpu() = default;
	~ggml_tensor_extra_gpu();
};

ggml_backend_buffer_type* ggml_backend_cuda_buffer_type(int device);

cuda_backend_buffer_type* to_cuda_buffer_type(ggml_backend_buffer_type* buft);

