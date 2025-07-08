#include <mutex>
#include "common.h"
#include "cuda_config.h"
#include "vendor_constant.h"

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

import ggml;

[[noreturn]]
void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s", msg);
    GGML_LOG_ERROR("  current device: {}, in function {} at {}:{}", id, func, file, line);
    GGML_LOG_ERROR("  {}", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("{}: failed to initialize " GGML_CUDA_NAME ": {}", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_LOG_INFO("{}: GGML_CUDA_FORCE_MMQ:    yes", __func__);
#else
    GGML_LOG_INFO("{}: GGML_CUDA_FORCE_MMQ:    no", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_LOG_INFO("{}: GGML_CUDA_FORCE_CUBLAS: yes", __func__);
#else
    GGML_LOG_INFO("{}: GGML_CUDA_FORCE_CUBLAS: no", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_LOG_INFO("{}: found {} " GGML_CUDA_NAME " devices:", __func__, info.device_count);
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        GGML_LOG_INFO("  Device {}: {}, compute capability {}.{}, VMM: {}", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info.devices[id].integrated = prop.integrated;
        info.devices[id].nsm = prop.multiProcessorCount;
        info.devices[id].smpb = prop.sharedMemPerBlock;
        info.devices[id].warp_size = prop.warpSize;
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device {}: {}, {} (0x{:x}), VMM: {}, Wave Size: {}",
            id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
            device_vmm ? "yes" : "no", prop.warpSize);
#elif defined(GGML_USE_MUSA)
        // TODO: refine the .cc to reflect MUSA's actual CC capabilities
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100 * prop.major + 10 * prop.minor;
        GGML_LOG_INFO("  Device {}: {}, compute capability {}.{}, VMM: {}",
            id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100 * prop.major + 10 * prop.minor;
        GGML_LOG_INFO("  Device {}: {}, compute capability {}.{}, VMM: {}",
            id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    return info;
}

const ggml_cuda_device_info& ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
    if constexpr (force_enable_cuda_blas_v) {
        return false;
    }

    bool mmq_supported;

    switch (type) {
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
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_IQ4_NL:
        mmq_supported = true;
        break;
    default:
        mmq_supported = false;
        break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (int8_mma_available(cc)) {
        return true;
    }

    if (cc < GGML_CUDA_CC_DP4A) {
        return false;
    }

    if constexpr (force_enable_cuda_mmq_v) {
        return true;
    }

    if (cc < GGML_CUDA_CC_OFFSET_AMD) {
        return cc < GGML_CUDA_CC_VOLTA || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    return (cc < GGML_CUDA_CC_RDNA3 && cc != GGML_CUDA_CC_CDNA && cc != GGML_CUDA_CC_VEGA20) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

static bool fp32_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

bool bf16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) || GGML_CUDA_CC_IS_CDNA(cc) || cc >= GGML_CUDA_CC_RDNA3;
}

bool ggml_cuda_should_use_mmv(enum ggml_type type, int cc, const int64_t* src0_ne, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }
    switch (type) {
    case GGML_TYPE_F32:
        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return ne11 <= 8;
            }
            if (cc >= GGML_CUDA_CC_TURING) {
                return ne11 <= 4;
            }
            return ne11 <= 3;
        }
        else if (GGML_CUDA_CC_IS_AMD(cc)) {
            if (fp32_mma_hardware_available(cc)) {
                return ne11 <= 3;
            }
            return ne11 <= 8;
        }
        return ne11 <= 8;
    case GGML_TYPE_F16:
        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
            if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return src0_small && ne11 <= 4;
            }
            if (fp16_mma_hardware_available(cc)) {
                return src0_small && ne11 <= 3;
            }
            return ne11 <= 8;
        }
        else if (GGML_CUDA_CC_IS_AMD(cc)) {
            if (fp16_mma_hardware_available(cc)) {
                if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                    return ne11 <= 5;
                }
                return ne11 <= 2;
            }
            return ne11 <= 8;
        }
        return ne11 <= 8;
    case GGML_TYPE_BF16:
        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
            if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return src0_small && ne11 <= 4;
            }
            if (bf16_mma_hardware_available(cc)) {
                return src0_small && ne11 <= 3;
            }
            return ne11 <= 8;
        }
        else if (GGML_CUDA_CC_IS_AMD(cc)) {
            if (bf16_mma_hardware_available(cc)) {
                return ne11 <= 3;
            }
            return ne11 <= 8;
        }
        return ne11 <= 8;
    default:
        return false;
    }
}

bool fast_fp16_hardware_available(const int cc)
{
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc != 610) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}