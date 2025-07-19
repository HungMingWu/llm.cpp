module;
#include <bit>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "common.h"
#include "vendors/cuda.h"
#define GGML_ASSERT(...)
#define GGML_ABORT(...)
#define GGML_BACKEND_API_VERSION 1
#define WARP_SIZE 32

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

    std::unique_ptr<ggml_backend> init_backend(const char*) override
    {
        return ggml_backend_cuda_init(device);
    }

    ggml_backend_buffer_type_t get_buffer_type() override
    {
        return ggml_backend_cuda_buffer_type(device);
    }

    ggml_backend_buffer_type_t get_host_buffer_type() override
    {
        static cuda_host_backend_buffer_type type;
        return &type;
    }

    bool supports_op(const ggml_tensor* op) override
    {
        // split buffers can only be used with GGML_OP_MUL_MAT
        if (op->op != GGML_OP_MUL_MAT) {
            for (auto &src : op->src) {
                if (!src) continue;
                if (!src->buffer) continue;
                if (to_split_buffer_type(src->buffer->get_type())) {
                    return false;
                }
            }
        }

        // check if all the sources are allocated on this device
        for (auto& src : op->src) {
            if (!src) continue;
            if (!src->buffer) continue;
            if (auto cuda_buffer_type = to_cuda_buffer_type(src->buffer->get_type())) {
                if (cuda_buffer_type->device != device) {
                    return false;
                }
            }
        }

        switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
            case GGML_UNARY_OP_ABS:
            case GGML_UNARY_OP_SGN:
            case GGML_UNARY_OP_NEG:
            case GGML_UNARY_OP_STEP:
            case GGML_UNARY_OP_GELU:
            case GGML_UNARY_OP_SILU:
            case GGML_UNARY_OP_RELU:
            case GGML_UNARY_OP_SIGMOID:
            case GGML_UNARY_OP_HARDSIGMOID:
            case GGML_UNARY_OP_HARDSWISH:
            case GGML_UNARY_OP_GELU_ERF:
            case GGML_UNARY_OP_GELU_QUICK:
            case GGML_UNARY_OP_TANH:
            case GGML_UNARY_OP_EXP:
            case GGML_UNARY_OP_ELU:
                return ggml_is_contiguous(op->src[0]);
            default:
                return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
            case GGML_GLU_OP_REGLU:
            case GGML_GLU_OP_GEGLU:
            case GGML_GLU_OP_SWIGLU:
            case GGML_GLU_OP_GEGLU_ERF:
            case GGML_GLU_OP_GEGLU_QUICK:
                return ggml_is_contiguous_1(op->src[0]);
            default:
                return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        {
            const ggml_tensor* a = op->src[0];
            const ggml_tensor* b = op->src[1];
            // for small weight matrices the active device can end up without any rows, don't use row split in those cases
            // this avoids some edge cases (and the performance would not be good anyways)
            cuda_split_backend_buffer_type* split_bufer_type = (a->buffer) ? to_split_buffer_type(a->buffer->get_type()) : nullptr;
            if (split_bufer_type) {
                if (a->ne[2] > 1 || a->ne[3] > 1) {
                    return false;
                }
                // for small weight matrices the active device can end up without any rows, don't use row split in those cases
                // this avoids some edge cases (and the performance would not be good anyways)
                auto [row_low, row_high] = get_row_split(ggml_nrows(a), split_bufer_type->tensor_split, device);
                if (row_low == row_high) {
                    return false;
                }
            }
            if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                return false;
            }
#ifdef GGML_USE_MUSA
            const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
            if (b->ne[2] * b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                    a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                    return false;
                }
                if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                    a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                    return false;
                }
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
            case GGML_TYPE_BF16:
            case GGML_TYPE_I32:
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
        case GGML_OP_SET_ROWS:
        {
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_I64;
        } break;
        case GGML_OP_CPY:
        {
            ggml_type src0_type = op->src[0]->type;
            ggml_type src1_type = op->src[1]->type;
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
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
            return true;
        case GGML_OP_SSM_SCAN: {
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            }
            else {
                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return op->src[0]->type != GGML_TYPE_BF16;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK:
            return std::bit_cast<float>(op->op_params[1]) == 0.0f;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_UPSCALE:
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
            if (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                const int cc = ggml_cuda_info().devices[device].cc;
                if (!new_mma_available(cc)) {
                    return false;
                }
                const int gqa_ratio = op->src[0]->ne[2] / op->src[1]->ne[2];
                return op->src[1]->ne[0] == 576 && op->src[2]->ne[0] == 512 && op->src[3] && gqa_ratio % 16 == 0;
            }
            if (op->src[0]->ne[0] == 192) {
                return false;
            }
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
            if (op->src[3] && op->src[3]->ne[2] != 1) {
                return false;
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
        return buffer_type_from_device(buft, device);
    }

    bool offload_op(const ggml_tensor* op) override
    {
        const int min_batch_size = 32;

        return get_op_batch_size(op) >= min_batch_size;
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
    std::vector<ggml_backend_device*> devices_span;
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
            devices_span.push_back(dev);
        }
    }

    std::string_view get_name() override {
        return GGML_CUDA_NAME;
    }

    std::span<ggml_backend_dev_t> get_devices() override {
        return devices_span;
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
			GGML_LOG_ERROR("{}: invalid device {}", __func__, device);
			return nullptr;
		}
        auto cuda_device = ggml_backend_cuda_reg()->get_devices()[device];
        auto backend = std::make_unique<ggml_backend_cuda>(cuda_device);
        backend->device = device;
        backend->name = GGML_CUDA_NAME + std::to_string(device);
        return backend;
	}
}
