module;
#include <bit>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "cuda_config.h"
#include "common.h"
#include "vendors/cuda.h"
#if defined(__linux__)
#include <stdio.h>
#include <memory>
#endif
#define WARP_SIZE 32

module ggml;
import :cuda.op;
import :cuda.backend;

static int register_ok = []() {
    get_reg().register_backend(ggml_backend_cuda_reg());
    return 0;
}();

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long* available_memory_kb, long* free_swap_kb) {
    FILE* meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char* line = file_buffer.get();
    char* line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        }
        else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        }
        else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        }
        else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        }
        else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        }
        else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

ggml_backend_buffer_type* ggml_backend_cuda_device::get_buffer_type()
{
    return ggml_backend_cuda_buffer_type(device);
}

std::span<const ggml_backend_feature> backend_cuda_reg::get_features()
{
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
#define _STRINGIFY(...) #__VA_ARGS__
#define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

#ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
#endif

        if constexpr (not ggml_cuda_force_mmq_v) {
            features.push_back({ "FORCE_MMQ", "1" });
        }

        if constexpr (not force_enable_cuda_blas_v) {
            features.push_back({ "FORCE_CUBLAS", "1" });
        }

        if constexpr (not ggml_use_vmm_v) {
            features.push_back({ "NO_VMM", "1" });
        }

        if constexpr (ggml_cuda_no_peer_copy_v) {
            features.push_back({ "NO_PEER_COPY", "1" });
        }

        if constexpr (ggml_cuda_use_graphs_v) {
            features.push_back({ "USE_GRAPHS", "1" });
        }

        features.push_back({ "PEER_MAX_BATCH_SIZE", std::to_string(ggml_cuda_peer_max_batch_size_v) });

        if constexpr (ggml_cuda_fa_all_quants_v) {
            features.push_back({ "FA_ALL_QUANTS", "1" });
        }

        {
            const auto& info = ggml_cuda_info();
            for (int id = 0; id < info.device_count; ++id) {
                if (blackwell_mma_available(info.devices[id].cc)) {
                    features.push_back({ "BLACKWELL_NATIVE_FP4", "1" });
                    break;
                }
            }
        }

#undef _STRINGIFY
#undef STRINGIFY

        return features;
    }();

    return features;
}

void ggml_backend_cuda_device::get_memory(size_t* free, size_t* total)
{
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemGetInfo(free, total));

    // ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, this->device));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        }
        else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__)
}

bool ggml_backend_cuda_device::supports_op(const ggml_tensor* op)
{
    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (auto& src : op->src) {
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
        case GGML_UNARY_OP_EXPM1:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_XIELU:
        case GGML_UNARY_OP_FLOOR:
        case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_TRUNC:
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
        case GGML_GLU_OP_SWIGLU_OAI:
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
        case GGML_TYPE_MXFP4:
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
            (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
    } break;
    case GGML_OP_SET:
    {
        const ggml_type t = op->type;
        return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
            t == op->src[0]->type &&
            t == op->src[1]->type;
    } break;
    case GGML_OP_CPY:
    {
        ggml_type src0_type = op->src[0]->type;
        ggml_type src1_type = op->src[1]->type;
        if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
            (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
            ) {
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
        if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
            return true;
        }
        if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
            return true;
        }
        if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
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
        if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16) && src1_type == GGML_TYPE_F32) {
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
    case GGML_OP_ADD_ID:
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
        return true;
    case GGML_OP_DIAG_MASK_INF:
        return true;
    case GGML_OP_SOFT_MAX:
        return true;
    case GGML_OP_SOFT_MAX_BACK:
        return std::bit_cast<float>(op->op_params[1]) == 0.0f;
    case GGML_OP_ROLL:
        if (op->src[0]->type == GGML_TYPE_F32) {
            return true;
        }
        return false;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK: {
        return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
    }
    case GGML_OP_IM2COL:
    case GGML_OP_IM2COL_3D:
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_CONV_TRANSPOSE_2D:
    case GGML_OP_POOL_2D:
    case GGML_OP_ACC:
        return true;
    case GGML_OP_TOP_K:
    case GGML_OP_SUM:
        return ggml_is_contiguous_rows(op->src[0]);
    case GGML_OP_ARGSORT:
        return enable_cuda_cub_v ? true : op->src[0]->ne[0] <= 1024;
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_PAD:
        return ggml_is_contiguous(op->src[0]);
    case GGML_OP_UPSCALE:
    case GGML_OP_PAD_REFLECT_1D:
    case GGML_OP_ARANGE:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_LEAKY_RELU:
    case GGML_OP_RWKV_WKV6:
    case GGML_OP_GATED_LINEAR_ATTN:
    case GGML_OP_RWKV_WKV7:
        return true;
    case GGML_OP_FLASH_ATTN_EXT:
        return op::ggml_cuda_flash_attn_ext_supported(device, op);
    case GGML_OP_CROSS_ENTROPY_LOSS:
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
    case GGML_OP_OPT_STEP_ADAMW:
    case GGML_OP_OPT_STEP_SGD:
    case GGML_OP_FILL:
    case GGML_OP_CUMSUM:
    case GGML_OP_TRI:
    case GGML_OP_DIAG:
    case GGML_OP_SOLVE_TRI:
        return true;
    default:
        return false;
    }
}


std::unique_ptr<ggml_backend> ggml_backend_cuda_init(int device)
{
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