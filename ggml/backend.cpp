module;
#include <assert.h>
#include <stddef.h>
#include <memory>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;

void ggml_backend::set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    synchronize();
    ggml_backend_tensor_set(tensor, data, offset, size);
}

void ggml_backend::get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    synchronize();
    ggml_backend_tensor_get(tensor, data, offset, size);
}

void ggml_backend::set_tensor_2d_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data)
{
    for (size_t i = 0; i < n_copies; i++) {
        set_tensor_async(tensor, (const char*)data + i * stride_data, offset + i * stride_tensor, size);
    }
}

void ggml_backend::get_tensor_2d_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data)
{
    for (size_t i = 0; i < n_copies; i++) {
        get_tensor_async(tensor, (char*)data + i * stride_data, offset + i * stride_tensor, size);
    }
}

void ggml_backend::set_tensor_2d_async(ggml_tensor* tensor, const void* data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data)
{
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies - 1) * stride_tensor + size <= tensor->nbytes() && "tensor write out of bounds");
    return set_tensor_2d_async_impl(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend::get_tensor_2d_async(const ggml_tensor* tensor, void* data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data)
{
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies - 1) * stride_tensor + size <= tensor->nbytes() && "tensor write out of bounds");
    return get_tensor_2d_async_impl(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
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

ggml_status ggml_backend::graph_compute(ggml_cgraph* cgraph) {
    enum ggml_status err = graph_compute_impl(cgraph);
    synchronize();
    return err;
}

std::unique_ptr<ggml_backend_buffer> ggml_backend::alloc_tensors(const ggml_context* ctx)
{
    return get_default_buffer_type()->alloc_tensors(ctx);
}

std::unique_ptr<ggml_backend_buffer> ggml_backend::alloc_buffer(size_t size)
{
    return device->get_buffer_type()->alloc_buffer(size);
}