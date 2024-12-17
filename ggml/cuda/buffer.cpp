module;
#include "common.h"
#define GGML_ASSERT(...)

module ggml;

void cuda_split_backend_buffer::init_tensor(ggml_tensor* tensor)
{
	GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

	auto split_buft = static_cast<cuda_split_backend_buffer_type*>(buft);

	const int64_t ne0 = tensor->ne[0];

	ggml_tensor_extra_gpu* extra = new ggml_tensor_extra_gpu{};
	tensor_extras.push_back(extra);

	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		int64_t row_low, row_high;
		get_row_split(&row_low, &row_high, tensor, split_buft->tensor_split, id);

		int64_t nrows_split = row_high - row_low;
		if (nrows_split == 0) {
			continue;
		}

		size_t size = ggml_nbytes_split(tensor, nrows_split);
		const size_t original_size = size;

		// pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
		if (ne0 % MATRIX_ROW_PADDING != 0) {
			size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
		}

		// FIXME: do not crash if cudaMalloc fails
		// currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
		ggml_cuda_set_device(id);
		char* buf;
		CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, size, id));

		// set padding to 0 to avoid possible NaN values
		if (size > original_size) {
			CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
		}

		extra->data_device[id] = buf;

		for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
			CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
		}
	}
	tensor->extra = extra;
}

void cuda_split_backend_buffer::set_tensor(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
	// split tensors must always be set in their entirety at once
	GGML_ASSERT(offset == 0);
	GGML_ASSERT(size == ggml_nbytes(tensor));

	auto split_buft = static_cast<cuda_split_backend_buffer_type*>(buft);

	const int64_t ne0 = tensor->ne[0];
	const size_t nb1 = tensor->nb[1];
	ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)tensor->extra;

	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		int64_t row_low, row_high;
		get_row_split(&row_low, &row_high, tensor, split_buft->tensor_split, id);

		int64_t nrows_split = row_high - row_low;
		if (nrows_split == 0) {
			continue;
		}

		const size_t offset_split = row_low * nb1;
		size_t size = ggml_nbytes_split(tensor, nrows_split);
		const size_t original_size = size;

		// pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
		if (ne0 % MATRIX_ROW_PADDING != 0) {
			size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
		}

		const char* buf_host = (const char*)data + offset_split;
		CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
	}

	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}
}

void cuda_split_backend_buffer::get_tensor(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
{
	// split tensors must always be set in their entirety at once
	GGML_ASSERT(offset == 0);
	GGML_ASSERT(size == ggml_nbytes(tensor));

	auto split_buft = static_cast<cuda_split_backend_buffer_type*>(buft);

	const int64_t ne0 = tensor->ne[0];
	const size_t nb1 = tensor->nb[1];
	ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)tensor->extra;

	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		int64_t row_low, row_high;
		get_row_split(&row_low, &row_high, tensor, split_buft->tensor_split, id);

		int64_t nrows_split = row_high - row_low;
		if (nrows_split == 0) {
			continue;
		}

		const size_t offset_split = row_low * nb1;
		size_t size = ggml_nbytes_split(tensor, nrows_split);
		const size_t original_size = size;

		// pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
		if (ne0 % MATRIX_ROW_PADDING != 0) {
			size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
		}

		char* buf_host = (char*)data + offset_split;
		CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
	}

	for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
		CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
	}
}