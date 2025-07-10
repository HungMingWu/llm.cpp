module;
#include <memory>
#include <vector>

module ggml;

std::unique_ptr<ggml_backend_buffer> ggml_backend_buffer_type::alloc_buffer(size_t size) {
    if (size == 0) return nullptr;
    return alloc_buffer_impl(size);
}

size_t ggml_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor) { return tensor->nbytes(); }

template <typename Iter>
static bool alloc_tensor_range(ggml_context* ctx,
	Iter first, Iter last,
	ggml_backend_buffer_type_t buft, size_t size,
	std::vector<std::unique_ptr<ggml_backend_buffer>>* buffers) {

	std::unique_ptr<ggml_backend_buffer> buffer = buft->alloc_buffer(size);

	if (buffer == NULL) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("{}: failed to allocate {} buffer of size {}", __func__, buft->get_name(), size);
#endif
		return false;
	}

	ggml_tallocr tallocr(buffer.get());

	for (auto it = first; it != last; ++it) {
		auto& t = *it;
		if (t->data == NULL) {
			if (t->view_src == NULL) {
				tallocr.alloc(t);
			}
			else if (t->buffer == NULL) {
				ggml_backend_view_init(t);
			}
		}
		else {
			if (t->view_src != NULL && t->buffer == NULL) {
				// view of a pre-allocated tensor
				ggml_backend_view_init(t);
			}
		}
	}

	buffers->push_back(std::move(buffer));
	return true;
}

std::unique_ptr<ggml_backend_buffer> ggml_backend_buffer_type::alloc_tensors(ggml_context* ctx)
{
	size_t alignment = get_alignment();
	size_t max_size = get_max_size();

	std::vector<std::unique_ptr<ggml_backend_buffer>> buffers;

	size_t cur_buf_size = 0;
	auto first = ctx->getTensors().begin();
	for (auto it = ctx->getTensors().begin(); it != ctx->getTensors().end(); ++it) {
		auto& t = *it;
		size_t this_size = 0;
		if (t->data == nullptr && t->view_src == nullptr) {
			this_size = GGML_PAD(get_alloc_size(t), alignment);
		}
#if 0
		if (this_size > max_size) {
			GGML_LOG_ERROR("{}: tensor {} is too large to fit in a {} buffer (tensor size: {}, max buffer size: {})",
				__func__, t->name,
				ggml_backend_buft_name(buft),
				this_size, max_size);
			return nullptr;
		}
#endif
		if ((cur_buf_size + this_size) > max_size) {
			// allocate tensors in the current buffer
			if (!alloc_tensor_range(ctx, first, it, this, cur_buf_size, &buffers)) {
				return NULL;
			}
			first = it;
			cur_buf_size = this_size;
		}
		else {
			cur_buf_size += this_size;
		}
	}

	// allocate remaining tensors
	if (cur_buf_size > 0) {
		if (!alloc_tensor_range(ctx, first, ctx->getTensors().end(), this, cur_buf_size, &buffers)) {
			return nullptr;
		}
	}

	if (buffers.empty()) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("{}: all tensors in the context are already allocated", __func__);
#endif
		return nullptr;
	}

	if (buffers.size() == 1) {
		return std::move(buffers[0]);
	}
	else {
		size_t total_size = 0;
		for (const auto& buffer : buffers) {
			total_size += buffer->get_size();
		}
		return std::make_unique<multi_backend_buffer>(buffers[0]->get_type(), total_size, std::move(buffers));
	}
}
