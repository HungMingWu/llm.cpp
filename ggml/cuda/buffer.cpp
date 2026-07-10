module;
#include "common.h"
#include <map>
#include <mutex>
#include <string>
#define GGML_ASSERT(...)

module ggml;
import :cuda.buffer;
import :cuda.buffer_type;
import :cuda.device;
import :cuda.registry;

cuda_backend_buffer::~cuda_backend_buffer()
{
	CUDA_CHECK(cudaFree(dev_ptr));
}

bool cuda_backend_buffer::cpy_tensor(const ggml_tensor* src, ggml_tensor* dst)
{
	cuda_backend_buffer* cuda_buf_src = dynamic_cast<cuda_backend_buffer*>(src->buffer);
	cuda_backend_buffer* cuda_buf_dst = dynamic_cast<cuda_backend_buffer*>(dst->buffer);
	if (cuda_buf_src && cuda_buf_dst) {
		// compare the backing physical devices: distinct virtual devices may share one physical GPU,
		// in which case a same-device copy (not a peer copy) is required
		const int src_physical = ggml_cuda_get_physical_device(cuda_buf_src->device);
		const int dst_physical = ggml_cuda_get_physical_device(cuda_buf_dst->device);
		if (src_physical == dst_physical) {
			CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, src->nbytes(), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
		}
		else {
			if constexpr (ggml_cuda_no_peer_copy_v) {
				return false;
			}
			else {
				CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, src->nbytes(), cudaStreamPerThread));
			}
		}
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
		return true;
	}
	return false;
}

ggml_tensor_extra_gpu::~ggml_tensor_extra_gpu()
{
	for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
		for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
			if (events[id][is] != nullptr) {
				CUDA_CHECK(cudaEventDestroy(events[id][is]));
			}
		}
		if (data_device[id] != nullptr) {
			CUDA_CHECK(cudaFree(data_device[id]));
		}
	}
}

ggml_backend_buffer_type* ggml_backend_cuda_buffer_type(int device) {
	if (device >= ggml_backend_cuda_get_device_count()) {
		return nullptr;
	}

	static cuda_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];
	static std::once_flag initialized;

	std::call_once(initialized, [&]() {
		for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
			ggml_backend_cuda_buffer_types[i].device = i;
			ggml_backend_cuda_buffer_types[i].set_device(ggml_backend_cuda_reg()->get_device(i));
			ggml_backend_cuda_buffer_types->name = GGML_CUDA_NAME + std::to_string(i);
		}
	});
	return &ggml_backend_cuda_buffer_types[device];
}

cuda_backend_buffer_type* to_cuda_buffer_type(ggml_backend_buffer_type* buft)
{
	return dynamic_cast<cuda_backend_buffer_type*>(buft);
}
