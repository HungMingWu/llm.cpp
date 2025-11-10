#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "common.h"
#include "cuda_config.h"

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

import ggml;

[[noreturn]]
void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: {}", msg);
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

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char* pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    }
    else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
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

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if defined(GGML_USE_VMM)
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
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm = prop.multiProcessorCount;
        info.devices[id].smpb = prop.sharedMemPerBlock;
        info.devices[id].warp_size = prop.warpSize;
#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device {} {}: {}  cc {}.{}",
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
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device {}: {}, compute capability {}.{}, VMM: {}",
            id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100 * prop.major + 10 * prop.minor;
        GGML_LOG_INFO("  Device {}: {}, compute capability {}.{}, VMM: {}",
            id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        }
        else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        }
        else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device {}: {}", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second);
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.");
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

bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

bool fp32_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc);
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

void CUDA_SET_SHARED_MEMORY_LIMIT(const void* kernel, size_t nbytes)
{
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };
    const int id = ggml_cuda_get_device();
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes));
        shared_memory_limit_raised[id] = true;
    }
#else
#endif // !(defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool amd_mfma_available(const int cc)
{
#if !defined(GGML_HIP_NO_MMQ_MFMA)
    return GGML_CUDA_CC_IS_CDNA(cc);
#else
    return false;
#endif //!defined(GGML_HIP_NO_MMQ_MFMA)
}

bool volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

bool ampere_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}
