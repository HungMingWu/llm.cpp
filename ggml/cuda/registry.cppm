module;
#include <format>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "common.h"
#include "vendors/cuda.h"
#define GGML_ASSERT(...)
#define GGML_ABORT(...)
#define GGML_BACKEND_API_VERSION 2

export module ggml:cuda.registry;
import :ds;
import :log;
import :tensor;
import :cuda.buffer;
import :cuda.buffer_type;
import :cuda.backend;

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

export {
    std::unique_ptr<ggml_backend> ggml_backend_cuda_init(int device);
}

class ggml_backend_cuda_device : public ggml_backend_device {
public:
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
    using ggml_backend_device::ggml_backend_device;
    const char* get_name() override { return name.c_str(); }
    const char* get_description() override { return description.c_str(); }
    void get_memory(size_t* free, size_t* total) override;
    enum ggml_backend_dev_type get_type() override {
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    void get_props(struct ggml_backend_dev_props* props) override {
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

    std::unique_ptr<ggml_backend> init_backend(const char*) override
    {
        return ggml_backend_cuda_init(device);
    }

    ggml_backend_buffer_type* get_buffer_type() override
    {
        return ggml_backend_cuda_buffer_type(device);
    }

    ggml_backend_buffer_type* get_host_buffer_type() override
    {
        static cuda_host_backend_buffer_type type;
        return &type;
    }

    bool supports_op(const ggml_tensor* op) override;

    bool supports_buft(ggml_backend_buffer_type* buft) override
    {
        return buffer_type_from_device(buft, device);
    }

    bool offload_op(const ggml_tensor* op) override
    {
        return get_op_batch_size(op) >= op_offload_min_batch_size;
    }

    ggml_backend_event* event_new() override
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

    void event_free(ggml_backend_event* event) override
    {
        CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
        delete event;
    }

    void event_synchronize(ggml_backend_event* event) override
    {
        CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
    }
};

class backend_cuda_reg : public ggml_backend_reg {
    std::vector<ggml_backend_cuda_device*> devices;
public:
    backend_cuda_reg(int api_version, void* context)
        : ggml_backend_reg(api_version, context)
    {
        const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
        for (int i = 0; i < ggml_cuda_info().device_count; i++) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            if (prop.major < 3) {
                GGML_LOG_INFO("{}: skipping device {} {} with compute capability {}.{} (minimum is 3.0)",
					__func__, i, prop.name, prop.major, prop.minor);
                continue;
            } else {
                ggml_backend_cuda_device* dev = new ggml_backend_cuda_device(this);
                dev->device = i;
                dev->name = GGML_CUDA_NAME + std::to_string(i);
                dev->description = prop.name;
                dev->pci_bus_id = std::format("{:04x}:{:02x}:{:02x}.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                dev->op_offload_min_batch_size = min_batch_size;
                devices.push_back(dev);
            }
        }
    }

    std::string_view get_name() override {
        return GGML_CUDA_NAME;
    }

    ggml_backend_device* get_device(size_t index) override {
        return devices.at(index);
    }

	void* get_proc_address(std::string_view name) override {
#if 0
        if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
            return (void*)ggml_backend_cuda_split_buffer_type;
        }
        if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
            return (void*)ggml_backend_cuda_register_host_buffer;
        }
        if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
            return (void*)ggml_backend_cuda_unregister_host_buffer;
        }
		return nullptr;
#else
		return nullptr;
#endif
	}
    std::span<const ggml_backend_feature> get_features() override;
};

export
{
	ggml_backend_reg_t ggml_backend_cuda_reg() {
        static backend_cuda_reg ggml_backend_cuda_reg = {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .context     = */ nullptr,
        };
        return &ggml_backend_cuda_reg;
	}

    std::unique_ptr<ggml_backend> ggml_backend_cuda_init(int device) {
		if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
			GGML_LOG_ERROR("{}: invalid device {}", __func__, device);
			return nullptr;
		}
        auto cuda_device = ggml_backend_cuda_reg()->get_device(device);
        auto backend = std::make_unique<ggml_backend_cuda>(cuda_device);
        backend->device = device;
        backend->name = GGML_CUDA_NAME + std::to_string(device);
        return backend;
	}
}
