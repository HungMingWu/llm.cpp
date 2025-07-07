module;
#include <memory>

module ggml;

std::unique_ptr<ggml_backend_buffer> ggml_backend_buffer_type::alloc_buffer(size_t size) {
    if (size == 0) return nullptr;
    return alloc_buffer_impl(size);
}

size_t ggml_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor) { return tensor->nbytes(); }