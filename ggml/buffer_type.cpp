module;
#include <iterator>
#include <memory>
#include <vector>

module ggml;

std::unique_ptr<ggml_backend_buffer> ggml_backend_buffer_type::alloc_buffer(size_t size) {
    if (size == 0) return nullptr;
    return alloc_buffer_impl(size);
}

size_t ggml_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor) { return tensor->nbytes(); }

std::vector<ggml_backend_buffer_type::alloc_buf_info> ggml_backend_buffer_type::calc_alloc_info(const ggml_context* ctx)
{
	std::vector<alloc_buf_info> result;

	size_t alignment = get_alignment();
	size_t max_size = get_max_size();

	size_t cur_buf_size = 0;
	auto first = ctx->getTensors().begin();
	for (auto it = ctx->getTensors().begin(); it != ctx->getTensors().end(); ++it) {
		auto& t = *it;
		size_t this_size = 0;
		if (t->data == nullptr && t->view_src == nullptr) {
			this_size = GGML_PAD(get_alloc_size(t), alignment);
		}

		if ((cur_buf_size + this_size) > max_size) {
			auto& new_buf_info = result.emplace_back();
			new_buf_info.buffer_size = cur_buf_size;
			std::copy(first, it, std::back_inserter(new_buf_info.allocated_tensors));
			first = it;
			cur_buf_size = this_size;
		}
		else {
			cur_buf_size += this_size;
		}
	}

	// allocate remaining tensors
	if (cur_buf_size > 0) {
		auto& new_buf_info = result.emplace_back();
		new_buf_info.buffer_size = cur_buf_size;
		std::copy(first, ctx->getTensors().end(), std::back_inserter(new_buf_info.allocated_tensors));
	}

	return result;
}

size_t ggml_backend_buffer_type::calc_needed_size(const ggml_context* ctx)
{
	auto result = calc_alloc_info(ctx);
	size_t total_size = 0;
	for (const auto& buf_info : result) {
		total_size += buf_info.buffer_size;
	}
	return total_size;
}

std::unique_ptr<ggml_backend_buffer> ggml_backend_buffer_type::alloc_tensors(const ggml_context* ctx)
{
	auto result = calc_alloc_info(ctx);

	if (result.empty()) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("{}: all tensors in the context are already allocated", __func__);
#endif
		return nullptr;
	}

	auto create = [this](size_t size, const std::vector<ggml_tensor*> &tensors)-> std::unique_ptr<ggml_backend_buffer> {
		std::unique_ptr<ggml_backend_buffer> buffer = alloc_buffer(size);
		ggml_tallocr tallocr(buffer.get());
		for (auto t : tensors) {
			if (t->data == nullptr) {
				if (t->view_src == nullptr) {
					tallocr.alloc(t);
				}
				else if (t->buffer == nullptr) {
					ggml_backend_view_init(t);
				}
			}
			else {
				if (t->view_src != nullptr && t->buffer == nullptr) {
					// view of a pre-allocated tensor
					ggml_backend_view_init(t);
				}
			}
		}
		return buffer;
	};

	if (result.size() == 1)
		return create(result[0].buffer_size, result[0].allocated_tensors);

	size_t total_size = 0;
	std::vector<std::unique_ptr<ggml_backend_buffer>> buffers;
	for (const auto& buf_info : result) {
		total_size += buf_info.buffer_size;
		buffers.push_back(create(buf_info.buffer_size, buf_info.allocated_tensors));
	}
	return std::make_unique<multi_backend_buffer>(buffers[0]->get_type(), total_size, std::move(buffers));
}