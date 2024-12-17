module;
#include <assert.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "common.h"

#define GGML_LOG_DEBUG(...)
#define GGML_LOG_ERROR(...)
#define GGML_ASSERT(...)

export module ggml:cuda.buffer;
import :cuda.buffer_type;
import :ds;
import :tensor;
import :traits;

cudaError_t ggml_cuda_device_malloc(void** ptr, size_t size, int device) {
	ggml_cuda_set_device(device);
#if defined(GGML_USE_HIP) && defined(GGML_HIP_UMA)
	auto res = hipMallocManaged(ptr, size);
	if (res == hipSuccess) {
		// if error we "need" to know why...
		CUDA_CHECK(hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
	}
	return res;
#else

#if !defined(GGML_USE_HIP)
	cudaError_t err;
	if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr)
	{
		err = cudaMallocManaged(ptr, size);
	}
	else
	{
		err = cudaMalloc(ptr, size);
	}
	return err;
#else
	return cudaMalloc(ptr, size);
#endif // !defined(GGML_USE_HIP)

#endif
}

size_t ggml_nbytes_split(const ggml_tensor* tensor, int nrows_split) {
	return nrows_split * ggml_row_size(tensor->type, tensor->ne[0]);
}

int64_t get_row_rounding(const std::array<float, GGML_CUDA_MAX_DEVICES>& tensor_split) {
	int64_t row_rounding = 0;
	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
			continue;
		}

		const int cc = ggml_cuda_info().devices[id].cc;
		row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
	}
	return row_rounding;
}

std::pair<int64_t, int64_t> get_row_split(int64_t nrows, const std::array<float, GGML_CUDA_MAX_DEVICES>& tensor_split, int id) {
	const int64_t rounding = get_row_rounding(tensor_split);

	const int64_t row_low = [&] {
		int64_t row_low = id == 0 ? 0 : nrows * tensor_split[id];
		row_low -= row_low % rounding;
		return row_low;
	}();

	const int64_t row_high = [&] {
		if (id == ggml_backend_cuda_get_device_count() - 1) {
			return nrows;
		}
		else {
			int64_t row_high = nrows * tensor_split[id + 1];
			row_high -= row_high % rounding;
			return row_high;
		}
	}();
	return { row_low, row_high };
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
public:
	cuda_backend_buffer(
		ggml_backend_buffer_type_t type,
		size_t size,
		int device,
		void* context)
		: ggml_backend_buffer(type, size),
		  device(device),
		  dev_ptr(context)
	{

	}

	~cuda_backend_buffer() override
	{
		CUDA_CHECK(cudaFree(dev_ptr));
	}

	void init_tensor(ggml_tensor* tensor) override
	{
		if (tensor->view_src != nullptr) {
			assert(tensor->view_src->buffer->get_type() == get_type());
			return;
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

	bool cpy_tensor(const ggml_tensor* src, ggml_tensor* dst) override
	{
#if 0
		if (ggml_backend_buffer_is_cuda(src->buffer)) {
			ggml_backend_cuda_buffer_context* src_ctx = (ggml_backend_cuda_buffer_context*)src->buffer->context;
			ggml_backend_cuda_buffer_context* dst_ctx = (ggml_backend_cuda_buffer_context*)dst->buffer->context;
			if (src_ctx->device == dst_ctx->device) {
				CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
			}
			else {
#ifdef GGML_CUDA_NO_PEER_COPY
				return false;
#else
				CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
			}
			CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
			return true;
		}
#endif
		return false;
	}

	void clear(uint8_t value) override
	{
		ggml_cuda_set_device(device);
		CUDA_CHECK(cudaMemsetAsync(dev_ptr, value, size, cudaStreamPerThread));
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}
};

struct ggml_tensor_extra_gpu {
	void* data_device[GGML_CUDA_MAX_DEVICES]{}; // 1 pointer for each device for split tensors
	cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]{}; // events for synchronizing multiple GPUs
	ggml_tensor_extra_gpu() = default;
	~ggml_tensor_extra_gpu();
};

struct cuda_split_backend_buffer : public ggml_backend_buffer {
	std::map<ggml_tensor*, ggml_tensor_extra_gpu> tensor_extras;
private:
	void* get_base_impl() override
	{
		// the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
		return (void*)0x1000;
	}
public:
	using ggml_backend_buffer::ggml_backend_buffer;
	void init_tensor(ggml_tensor* tensor) override;
	void set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size) override;
	void get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size) override;

	void clear(uint8_t) override {}
};

export
{
	ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
		if (device >= ggml_backend_cuda_get_device_count()) {
			return nullptr;
		}

		static cuda_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];
		static std::once_flag initialized;

		std::call_once(initialized, [&]() { 
			for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
				ggml_backend_cuda_buffer_types[i].device = i;
				ggml_backend_cuda_buffer_types->name = GGML_CUDA_NAME + std::to_string(i);
			}
		});
		return &ggml_backend_cuda_buffer_types[device];
	}

	cuda_backend_buffer_type* to_cuda_buffer_type(ggml_backend_buffer_type_t buft)
	{
		return dynamic_cast<cuda_backend_buffer_type*>(buft);
	}

	cuda_split_backend_buffer_type* to_split_buffer_type(ggml_backend_buffer_type_t buft) {
		return dynamic_cast<cuda_split_backend_buffer_type*>(buft);
	};
}
