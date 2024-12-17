module;
#include <memory>

module ggml;

std::unique_ptr<ggml_backend_buffer> ggml_backend_buffer_type::alloc_buffer(size_t size) {
    if (size == 0) return nullptr;
    return alloc_buffer_impl(size);
}
