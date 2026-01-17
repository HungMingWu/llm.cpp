module;
#include <memory>
#include "cuda_config.h"
#include "common.h"

module ggml;

import :cuda.buffer;
import :cuda.buffer_type;
import :cuda.device;

static int64_t get_op_batch_size(const ggml_tensor* op) {
    switch (op->op) {
    case GGML_OP_GET_ROWS:
        return 0;
    case GGML_OP_MUL_MAT:
        return op->ne[1];
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_ROPE:
        return op->ne[2];
    default:
        return ggml_nrows(op);
    }
}

ggml_backend_dev_type ggml_backend_cuda_device::get_type() {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void ggml_backend_cuda_device::get_props(ggml_backend_dev_props* props) {
    props->name = get_name();
    props->description = get_description();
    props->type = get_type();
    props->device_id = pci_bus_id.empty() ? nullptr : pci_bus_id.c_str();
    get_memory(&props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
    const bool events = !ggml_cuda_no_peer_copy_v;

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

std::unique_ptr<ggml_backend> ggml_backend_cuda_device::init_backend(const char*)
{
    return ggml_backend_cuda_init(device);
}

ggml_backend_buffer_type* ggml_backend_cuda_device::get_buffer_type()
{
    return ggml_backend_cuda_buffer_type(device);
}

ggml_backend_buffer_type* ggml_backend_cuda_device::get_host_buffer_type()
{
    static cuda_host_backend_buffer_type type;
    return &type;
}

bool ggml_backend_cuda_device::supports_buft(ggml_backend_buffer_type* buft)
{
    return buffer_type_from_device(buft, device);
}

bool ggml_backend_cuda_device::offload_op(const ggml_tensor* op)
{
    return get_op_batch_size(op) >= op_offload_min_batch_size;
}

ggml_backend_event* ggml_backend_cuda_device::event_new()
{
    if constexpr (ggml_cuda_no_peer_copy_v) {
        return nullptr;
    }
    else {
        ggml_cuda_set_device(device);

        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

        return new ggml_backend_event{
            /* .device  = */ this,
            /* .context = */ event,
        };
    }
}

void ggml_backend_cuda_device::event_free(ggml_backend_event* event)
{
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

void ggml_backend_cuda_device::event_synchronize(ggml_backend_event* event)
{
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}
