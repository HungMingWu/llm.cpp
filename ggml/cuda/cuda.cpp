module;
#include <memory>
#include <mutex>
#include <string>
#include "common.h"

module ggml;

import :cuda;
import :cuda.backend;
import :cuda.registry;

std::unique_ptr<ggml_backend> ggml_backend_cuda_init(int device)
{
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("{}: invalid device {}", __func__, device);
        return nullptr;
    }
    auto cuda_device = (ggml_backend_cuda_device*)ggml_backend_cuda_reg()->get_device(device);
    auto backend = std::make_unique<ggml_backend_cuda>(cuda_device);
    backend->device = device;
    backend->name = GGML_CUDA_NAME + std::to_string(device);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    std::lock_guard<std::mutex> lock(cuda_device->device_mutex);
    cuda_device->active_count++;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    return backend;
}