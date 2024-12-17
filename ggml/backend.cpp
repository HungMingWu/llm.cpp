module;
#include <assert.h>
#include <stddef.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;

void ggml_backend::set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    ggml_backend_tensor_set(tensor, data, offset, size);
}

void ggml_backend::get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    ggml_backend_tensor_get(tensor, data, offset, size);
}

void ggml_backend::set_tensor_async(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != nullptr && "tensor not allocated");
    GGML_ASSERT(offset + size <= tensor->nbytes() && "tensor write out of bounds");
    set_tensor_async_impl(tensor, data, offset, size);
}

void ggml_backend::get_tensor_async(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= tensor->nbytes() && "tensor read out of bounds");

    get_tensor_async_impl(tensor, data, offset, size);
}