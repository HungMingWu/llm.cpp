module;
#include <bit>
#include <memory>
#include <string>
#include "common.h"
#include "vendors/cuda.h"
#define GGML_ASSERT(...)
#define GGML_LOG_ERROR(...)
#define GGML_ABORT(...)
#define GGML_BACKEND_API_VERSION 1
#define WARP_SIZE 32

export module ggml:cuda.registry;
import :ds;
import :tensor;
import :cuda.buffer;
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

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
#if 0
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
#else
    return nullptr;
#endif
}

class ggml_backend_cuda_device : public ggml_backend_device {
public:
    int device;
    std::string name;
    std::string description;
    using ggml_backend_device::ggml_backend_device;
    const char* get_name() override { return name.c_str(); }
    const char* get_description() override { return description.c_str(); }
    void get_memory(size_t* free, size_t* total) override {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMemGetInfo(free, total));
    }
    enum ggml_backend_dev_type get_type() override {
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    void get_props(struct ggml_backend_dev_props* props) override {
        props->name = get_name();
        props->description = get_description();
        props->type = get_type();
        get_memory(&props->memory_free, &props->memory_total);

        bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
        bool events = false;
#else
        bool events = true;
#endif

        props->caps = {
            /* .async                 = */ true,
            /* .host_buffer           = */ host_buffer,
            /* .buffer_from_host_ptr  = */ false,
            /* .events                = */ events,
        };
    }

    ggml_backend_t init_backend(const char*) override
    {
        return ggml_backend_cuda_init(device).release();
    }

    ggml_backend_buffer_type_t get_buffer_type() override
    {
        return ggml_backend_cuda_buffer_type(device);
    }

    ggml_backend_buffer_type_t get_host_buffer_type() override
    {
        return ggml_backend_cuda_host_buffer_type();
    }

    bool supports_op(const ggml_tensor* op) override
    {
        // split buffers can only be used with GGML_OP_MUL_MAT
        if (op->op != GGML_OP_MUL_MAT) {
            for (auto &src : op->src) {
                if (!src) continue;
                if (!src->buffer) continue;
                if (ggml_backend_buft_is_cuda_split(src->buffer->get_type())) {
                    return false;
                }
            }
        }

        // check if all the sources are allocated on this device
        for (auto& src : op->src) {
            if (!src) continue;
            if (!src->buffer) continue;
            if (ggml_backend_buft_is_cuda(src->buffer->get_type())) {
#if 0
                ggml_backend_cuda_buffer_type_context* buft_ctx = (ggml_backend_cuda_buffer_type_context*)op->src[i]->buffer->buft->context;
                if (buft_ctx->device != device) {
                    return false;
                }
#endif
            }
        }

        switch (op->op) {
        case GGML_OP_UNARY:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        {
            const ggml_tensor* a = op->src[0];
            const ggml_tensor* b = op->src[1];
            // for small weight matrices the active device can end up without any rows, don't use row split in those cases
            // this avoids some edge cases (and the performance would not be good anyways)
            if (a->buffer && ggml_backend_buft_is_cuda_split(a->buffer->get_type())) {
#if 0
                ggml_backend_cuda_split_buffer_type_context* buft_ctx = (ggml_backend_cuda_split_buffer_type_context*)a->buffer->buft->context;
                int64_t row_low;
                int64_t row_high;
                get_row_split(&row_low, &row_high, a, buft_ctx->tensor_split, dev_ctx->device);
                if (row_low == row_high) {
                    return false;
                }
#endif
            }
            if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                return false;
            }
#ifdef GGML_USE_MUSA
            if (b->type == GGML_TYPE_F16 && b->ne[2] * b->ne[3] > 1 &&
                !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                return false;
            }
#endif // GGML_USE_MUSA
            switch (a->type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_K:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_BF16:
#ifdef GGML_USE_MUSA
                if (a->type == GGML_TYPE_Q3_K) {
                    return false;
                }
#endif // GGML_USE_MUSA
                return true;
            default:
                return false;
            }
        } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
        {
            switch (op->src[0]->type) {
            case GGML_TYPE_F16:
            case GGML_TYPE_F32:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
                return true;
            default:
                return false;
            }
        } break;
        case GGML_OP_GET_ROWS_BACK:
        {
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        } break;
        case GGML_OP_CPY:
        {
            ggml_type src0_type = op->src[0]->type;
            ggml_type src1_type = op->src[1]->type;
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                return true;
            }
            if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                return true;
            }
            if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                return true;
            }
            if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                return true;
            }
            if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                return true;
            }
            return false;
        } break;
        case GGML_OP_DUP:
        {
            ggml_type src0_type = op->src[0]->type;
            return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
        } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
        {
            return true;
        } break;
        case GGML_OP_REPEAT:
        {
            ggml_type src0_type = op->src[0]->type;
            return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
        } break;
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2] * op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
        {
            ggml_type src0_type = op->src[0]->type;
            return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
        } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
        {
            ggml_type src0_type = op->src[0]->type;
            ggml_type src1_type = op->src[1]->type;
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_SSM_CONV:
            return true;
        case GGML_OP_CONT:
            return op->src[0]->type != GGML_TYPE_BF16;
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK:
            return std::bit_cast<float>(op->op_params[1]) == 0.0f;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            const size_t ts = ggml_type_size(op->src[0]->type);
            const int64_t ne0_012 = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2];
            return op->src[0]->nb[0] == ts && op->src[0]->nb[3] == ne0_012 * ts;
        }
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST;
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_FLASH_ATTN_EXT: {
#ifndef FLASH_ATTN_AVAILABLE
            return false;
#endif // FLASH_ATTN_AVAILABLE
            if (op->src[1]->type == GGML_TYPE_BF16 || op->src[2]->type == GGML_TYPE_BF16) {
                return false;
            }
            if (op->src[0]->ne[0] == 64 && op->src[1]->type == GGML_TYPE_F16) {
                return true;
            }
            if (op->src[0]->ne[0] == 128) {
                return true;
            }
            if (op->src[0]->ne[0] == 256 && op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16) {
                return true;
            }
            return fp16_mma_available(ggml_cuda_info().devices[device].cc) &&
                op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16;
        }
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
            return true;
        default:
            return false;
        }
    }

    bool supports_buft(ggml_backend_buffer_type_t buft) override
    {
#if 0
        return (ggml_backend_buft_is_cuda(buft) || ggml_backend_buft_is_cuda_split(buft)) && buft->device == this;
#else
        return false;
#endif
    }

    bool offload_op(const struct ggml_tensor* op) override
    {
#if 0
        const int min_batch_size = 32;

        return get_op_batch_size(op) >= min_batch_size;
#else
        return false;
#endif
    }

    ggml_backend_event_t event_new() override
    {
#ifdef GGML_CUDA_NO_PEER_COPY
        return nullptr;
#else
        ggml_cuda_set_device(device);

        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

        return new ggml_backend_event{
            /* .device  = */ this,
            /* .context = */ event,
        };
#endif
    }

    void event_free(ggml_backend_event_t event) override
    {
        CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
        delete event;
    }

    void event_synchronize(ggml_backend_event_t event) override
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
        for (int i = 0; i < ggml_cuda_info().device_count; i++) {
            ggml_backend_cuda_device* dev = new ggml_backend_cuda_device(this);
            dev->device = i;
            dev->name = GGML_CUDA_NAME + std::to_string(i);

            ggml_cuda_set_device(i);
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            dev->description = prop.name;
            devices.push_back(dev);
        }
    }

	const char* get_name() override {
		return GGML_CUDA_NAME;
	}
	size_t get_device_count() override {
		return devices.size();
	}
	ggml_backend_dev_t get_device(size_t index) override {
        GGML_ASSERT(index < devices.size());
        return devices[index];
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
        if (strcmp(name, "ggml_backend_get_features") == 0) {
            return (void*)ggml_backend_cuda_get_features;
        }
		return nullptr;
#else
		return nullptr;
#endif
	}
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
			GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
			return nullptr;
		}
        // There is a bug in VC Module, modify when it fixes
        auto backend = std::make_unique<ggml_backend_cuda>(ggml_backend_cuda_reg()->get_device(device));
        backend->device = device;
        backend->name = GGML_CUDA_NAME + std::to_string(device);
        return backend;
	}
}
