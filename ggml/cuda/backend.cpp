module;
#include <assert.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#include "block.h"
#include "common.h"
#include "cuda_pool.h"
#include "op/convert.cuh"
#include "op/cuda_func.h"
#include "cuda_config.h"

#define GGML_ABORT(...)
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :fused;
import :cuda.backend;
import :cuda.device;
import :cuda.op;

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

namespace 
{
    // pool with virtual memory
    struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
        static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

        int device;
        int physical_device;
        CUdeviceptr pool_addr = 0;
        size_t pool_used = 0;
        size_t pool_size = 0;
        size_t granularity;

        explicit ggml_cuda_pool_vmm(int device) :
            device(device),
            physical_device(ggml_cuda_get_physical_device(device)),
            granularity(ggml_cuda_info().devices[device].vmm_granularity) {
        }

        ~ggml_cuda_pool_vmm() {
            if (pool_addr != 0) {
                CU_CHECK(cuMemUnmap(pool_addr, pool_size));
                CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
            }
        }

        void* alloc(size_t size, size_t* actual_size) override {
            // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
            const size_t alignment = 128;
            size = alignment * ((size + alignment - 1) / alignment);

            size_t avail = pool_size - pool_used;

            if (size > avail) {
                // round up to the next multiple of the granularity
                size_t reserve_size = size - avail;
                reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

                GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

                // allocate more physical memory
                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = physical_device;
                CUmemGenericAllocationHandle handle;
                CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

                // reserve virtual address space (if not already reserved)
                if (pool_addr == 0) {
                    CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
                }

                // map at the end of the pool
                CUdeviceptr start_ptr = (CUdeviceptr)((char*)(pool_addr)+pool_size);
                CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
                mappings.push_back({ start_ptr, reserve_size });
#endif

                // the memory allocation handle is no longer needed after mapping
                CU_CHECK(cuMemRelease(handle));

                // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
                bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
                use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

                if (use_peer_access) {
                    // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                    // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                    // allocations do not become peer-accessible from that alone, so access
                    // must be granted explicitly here. With virtual devices, grant access
                    // on the backing *physical* devices (deduplicated, since several
                    // virtual devices can map to the same physical GPU).
                    std::vector<CUmemAccessDesc> access_descs;
                    bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                    const int device_count = ggml_cuda_info().device_count;
                    for (int id = 0; id < device_count; ++id) {
                        const int id_physical = ggml_cuda_get_physical_device(id);
                        if (id_physical != physical_device) {
                            int can_access_peer = 0;
                            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                            if (!can_access_peer) {
                                continue;
                            }
                        }
                        if (physical_seen[id_physical]) {
                            continue;
                        }
                        physical_seen[id_physical] = true;
                        CUmemAccessDesc access = {};
                        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        access.location.id = id_physical;
                        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                        access_descs.push_back(access);
                    }
                    CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
                }
                else {
                    // set access for non P2P
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = physical_device;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
                }

                // add to the pool
                pool_size += reserve_size;

                //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                //       device, (unsigned long long) (pool_size/1024/1024),
                //       (unsigned long long) (reserve_size/1024/1024));
            }

            GGML_ASSERT(pool_addr != 0);

            void* ptr = (void*)(pool_addr + pool_used);
            *actual_size = size;
            pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
            printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

            return ptr;
        }

        void free(void* ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
            printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

            pool_used -= size;

            // all deallocations must be in reverse order of the allocations
            GGML_ASSERT(ptr == (void*)(pool_addr + pool_used));
        }
    };

    // buffer pool for cuda (legacy)
    struct ggml_cuda_pool_leg : public ggml_cuda_pool {
        static const int MAX_BUFFERS = 256;

        int device;
        struct ggml_cuda_buffer {
            void* ptr = nullptr;
            size_t size = 0;
        };

        ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
        size_t pool_size = 0;

        explicit ggml_cuda_pool_leg(int device) :
            device(device) {
        }

        ~ggml_cuda_pool_leg() {
            clear_pool();
            GGML_ASSERT(pool_size == 0);
        }

        void clear_pool() {
            ggml_cuda_set_device(device);
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                ggml_cuda_buffer& b = buffer_pool[i];
                if (b.ptr != nullptr) {
                    CUDA_CHECK(cudaFree(b.ptr));
                    pool_size -= b.size;
                    b.ptr = nullptr;
                    b.size = 0;
                }
            }
        }

        void* alloc(size_t size, size_t* actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
            int nnz = 0;
            size_t max_size = 0;
#endif
            size_t best_diff = 1ull << 36;
            int ibest = -1;
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                ggml_cuda_buffer& b = buffer_pool[i];
                if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                    ++nnz;
                    if (b.size > max_size) max_size = b.size;
#endif
                    if (b.size >= size) {
                        size_t diff = b.size - size;
                        if (diff < best_diff) {
                            best_diff = diff;
                            ibest = i;
                            if (!best_diff) {
                                void* ptr = b.ptr;
                                *actual_size = b.size;
                                b.ptr = nullptr;
                                b.size = 0;
                                return ptr;
                            }
                        }
                    }
                }
            }
            if (ibest >= 0) {
                ggml_cuda_buffer& b = buffer_pool[ibest];
                void* ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                return ptr;
            }
            void* ptr;
            size_t look_ahead_size = (size_t)(1.05 * size);
            look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
            ggml_cuda_set_device(device);
            cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaErrorMemoryAllocation) {
                (void)cudaGetLastError();
                const size_t cached_bytes = pool_size;
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[{}]: alloc of {:.2f} MiB failed, flushing {:.2f} MiB of cached buffers and retrying",
                    device, look_ahead_size / 1024.0 / 1024.0, cached_bytes / 1024.0 / 1024.0);
                CUDA_CHECK(cudaDeviceSynchronize());
                clear_pool();
                err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
                if (err == cudaSuccess) {
                    GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
                }
            }
            CUDA_CHECK(err);
            *actual_size = look_ahead_size;
            pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
            GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
            return ptr;
        }

        void free(void* ptr, size_t size) override {
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                ggml_cuda_buffer& b = buffer_pool[i];
                if (b.ptr == nullptr) {
                    b.ptr = ptr;
                    b.size = size;
                    return;
                }
            }
            GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS");
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaFree(ptr));
            pool_size -= size;
        }
    };

    std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device, int /*stream_no*/) {
        if constexpr (ggml_use_vmm_v) {
            if (ggml_cuda_info().devices[device].vmm) {
                return std::make_unique<ggml_cuda_pool_vmm>(device);
            }
        }
        return std::make_unique<ggml_cuda_pool_leg>(device);
    }

    using ggml_cuda_op_mul_mat_t = void(*)(
        ggml_backend_cuda& ctx,
        ggml_tensor* dst,
        const char* src0_dd_i,
        const float* src1_ddf_i,
        const char* src1_ddq_i,
        float* dst_dd_i,
        const int64_t row_low,
        const int64_t row_high,
        const int64_t src1_ncols,
        const int64_t src1_padded_row_size,
        cudaStream_t stream);
}

template <ggml_type type>
const float alphaVal = 1.0f;

template <ggml_type type>
const float betaVal = 0.0f;

template <>
const half alphaVal<GGML_TYPE_F16> = 1.0;

template <>
const half betaVal<GGML_TYPE_F16> = 0.0;

constexpr cudaDataType_t getDataType(ggml_type compute_type)
{
    if (compute_type == GGML_TYPE_F32)
        return CUDA_R_32F;
    else if (compute_type == GGML_TYPE_BF16)
        return CUDA_R_16BF;
    else if (compute_type == GGML_TYPE_F16)
        return CUDA_R_16F;
    else
        std::unreachable();
}

constexpr cublasComputeType_t getComputeType(ggml_type compute_type)
{
    if (compute_type == GGML_TYPE_F32)
        return CUBLAS_COMPUTE_32F;
    else if (compute_type == GGML_TYPE_BF16)
        return CUBLAS_COMPUTE_32F;
    else if (compute_type == GGML_TYPE_F16)
        return CUBLAS_COMPUTE_16F;
    else
        std::unreachable();
}

template<ggml_type compute_type, typename compute_t>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    const int64_t ne_dst = dst->nelements();
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(src0->nb[0] == src0_ts);
    int64_t s01 = src0->nb[1] / src0_ts;
    int64_t s02 = src0->nb[2] / src0_ts;
    int64_t s03 = src0->nb[3] / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(src1->nb[0] == src1_ts);
    int64_t s11 = src1->nb[1] / src1_ts;
    int64_t s12 = src1->nb[2] / src1_ts;
    int64_t s13 = src1->nb[3] / src1_ts;

    float* dst_ddf = (float*)dst->data;

    const compute_t* src0_ptr = nullptr;
    const compute_t* src1_ptr = nullptr;

    ggml_cuda_pool_alloc<compute_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<compute_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const compute_t*)src0->data;
    }
    else {
        src0_alloc.alloc(src0->nelements());

        convert_context ctx{
            .src_type = std::bit_cast<internal::ggml_type>(src0->type),
            .src_ne = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]},
            .src_nb = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]}
        };
        convert_to_cuda(ctx, src0->data, src0_alloc.get(), main_stream);
        // convert_to_cuda writes elements in logical tensor order, so the
        // converted buffer is contiguous even when the source allocation is
        // dense but permuted.
        s01 = src0->ne[0];
        s02 = src0->ne[1] * s01;
        s03 = src0->ne[2] * s02;
        is_src0_cont_2 = true;
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const compute_t*)src1->data;
    }
    else {
        src1_alloc.alloc(src1->nelements());

        convert_context ctx{
            .src_type = std::bit_cast<internal::ggml_type>(src1->type),
            .src_ne = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]},
            .src_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]}
        };
        convert_to_cuda(ctx, src1->data, src1_alloc.get(), main_stream);
        s11 = src1->ne[0];
        s12 = src1->ne[1] * s11;
        s13 = src1->ne[2] * s12;
        is_src1_cont_2 = true;
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<compute_t> dst_temp(ctx.pool());
    char* dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = getComputeType(compute_type);
    cudaDataType_t cu_data_type = getDataType(compute_type);
    cudaDataType_t cu_data_type_a = getDataType(compute_type);
    cudaDataType_t cu_data_type_b = getDataType(compute_type);
    const void* alpha = &alphaVal<compute_type>;
    const void* beta = &betaVal<compute_type>;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    }
    else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char*)dst_ddf;
        cu_compute_type = getComputeType(GGML_TYPE_F32);
        cu_data_type = getDataType(GGML_TYPE_F32);
        alpha = &alphaVal<GGML_TYPE_F32>;
        beta = &betaVal<GGML_TYPE_F32>;
    }
    else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char*)dst_ddf;  // Direct F32 output
        }
        else {
            dst_ptr = (char*)dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(compute_t);
            nbd3 /= sizeof(float) / sizeof(compute_t);
        }
    }

    GGML_ASSERT(src1->ne[2] % src0->ne[2] == 0);
    GGML_ASSERT(src1->ne[3] % src0->ne[3] == 0);

    // broadcast factors
    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                src0->ne[1], src1->ne[1], src1->ne[0],
                (const float*)alpha, (const float*)src0_ptr, s01,
                (const float*)src1_ptr, s11,
                (const float*)beta, (float*)dst_ptr, dst->ne[0]));
    }
    else if (src1->ne[2] == 1 && src1->ne[3] == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                src0->ne[1], src1->ne[1], src1->ne[0],
                alpha, src0_ptr, cu_data_type_a, s01,
                src1_ptr, cu_data_type_b, s11,
                beta, dst_ptr, cu_data_type, dst->ne[0],
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and src0->ne[2]==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = src0->ne[2] == 1 ? s03 : s02;
        const int64_t smb = src1->ne[2] == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
            cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                src0->ne[1], src1->ne[1], src1->ne[0],
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta, dst_ptr, cu_data_type, dst->ne[0], dst->ne[1] * dst->ne[0], // strideC
                src1->ne[2] * src1->ne[3],
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = src1->ne[2] * src1->ne[3];

        ggml_cuda_pool_alloc<const void*> ptrs_src(ctx.pool(), 2 * ne23);
        ggml_cuda_pool_alloc<      void*> ptrs_dst(ctx.pool(), 1 * ne23);

        const size_t src_type_size = sizeof(compute_t);

        k_compute_batched_ptrs_cuda(
            src0_ptr, src1_ptr, dst_ptr,
            ptrs_src.get(), ptrs_dst.get(),
            src1->ne[2], src1->ne[3],
            ne23,
            s02 * src_type_size, s03 * src_type_size,
            s12 * src_type_size, s13 * src_type_size,
            nbd2, nbd3,
            r2, r3, main_stream);

        CUBLAS_CHECK(
            cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                src0->ne[1], src1->ne[1], src1->ne[0],
                alpha, (const void**)(ptrs_src.get() + 0 * ne23), cu_data_type_a, s01,
                (const void**)(ptrs_src.get() + 1 * ne23), cu_data_type_b, s11,
                beta, (void**)(ptrs_dst.get() + 0 * ne23), cu_data_type, dst->ne[0],
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const int64_t ne = ne_dst;
        const size_t type_size = ggml_type_size(compute_type);
        convert_context ctx{
            .src_type = std::bit_cast<internal::ggml_type>(compute_type),
            .src_ne = { ne, 1, 1, 1 },
            .src_nb = { type_size, type_size * ne, type_size * ne, type_size * ne }
        };
        convert_to_cuda(ctx, dst_temp.get(), dst_ddf, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    }
    else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
        compute_type = GGML_TYPE_F32;
    }
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char* env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char& c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        }
        else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        }
        else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        }
        else if (env_cpp != "auto") {
            GGML_LOG_WARN("{}: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: {}", __func__, env_cpp);
        }
    }
    switch (compute_type) {
    case GGML_TYPE_F32:
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32, float>(ctx, src0, src1, dst);
        break;
    case GGML_TYPE_BF16:
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16, nv_bfloat16>(ctx, src0, src1, dst);
        break;
    case GGML_TYPE_F16:
        ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16, half>(ctx, src0, src1, dst);
        break;
    default:
        GGML_ABORT("Unsupported type");
    }
}

void ggml_backend_cuda::mul_mat(ggml_tensor* dst)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && op::fwht(stream(), src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && src0->nbytes() != src0->buffer->get_alloc_size(src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(*this, src0, src1, dst);
        return;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    if (utils::should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1])) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        op::mul_mat_vec_f(stream(), src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. src0->ne[1] == 1)
    if (src0->ne[1] == 1 && src1->ne[1] > MMVF_MAX_BATCH_SIZE && dst->ne[2] == 1 && dst->ne[3] == 1
        && src0->type == GGML_TYPE_F32
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
        && utils::should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = src1->ne[1];
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0] * src1->ne[1];
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        op::mul_mat_vec_f(stream(), src1, src0, nullptr, &dst_vec);
        return;
    }
    if (utils::should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id =*/ false)) {
        op::mul_mat_f(pool(), stream(), nullptr, dst);
        return;
    }
    if (utils::should_use_mmvq(src0->type, cc, src1->ne[1])) {
        op::mul_mat_vec_q(pool(), stream(), src0, src1, nullptr, dst);
        return;
    }
    if (utils::should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts =*/ 0)) {
        op::mul_mat_q(pool(), stream(), nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(*this, src0, src1, dst);
}

void ggml_backend_cuda::set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
    ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, stream()));
}

void ggml_backend_cuda::get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
{
    ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost, stream()));
}

void ggml_backend_cuda::set_tensor_2d_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data)
{
    ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char*)tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, stream()));
}

void ggml_backend_cuda::get_tensor_2d_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data)
{
    ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char*)tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, stream()));
}

bool ggml_backend_cuda::cpy_tensor_async(ggml_backend* backend_src, const ggml_tensor* src, ggml_tensor* dst)
{
    ggml_backend_buffer* buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer* buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cuda* cuda_backend_src = dynamic_cast<ggml_backend_cuda*>(backend_src);
    if (!cuda_backend_src) return false;

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    cuda_backend_buffer* cuda_buf_src = dynamic_cast<cuda_backend_buffer*>(buf_src);
    cuda_backend_buffer* cuda_buf_dst = dynamic_cast<cuda_backend_buffer*>(buf_dst);

    if (cuda_backend_src->device != cuda_buf_src->device || this->device != cuda_buf_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("{}: backend and buffer devices do not match", __func__);
#endif
        return false;
    }

    if (cuda_backend_src != this) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_backend_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToDevice, cuda_backend_src->stream()));
        }
        else {
            if constexpr (ggml_cuda_no_peer_copy_v) {
                return false;
            }
            else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, dst->nbytes(), cuda_backend_src->stream()));
            }
        }

        // record event on src stream after the copy
        if (!cuda_backend_src->copy_event) {
            ggml_cuda_set_device(cuda_backend_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_backend_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_backend_src->copy_event, cuda_backend_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(stream(), cuda_backend_src->copy_event, 0));
    }
    else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToDevice, cuda_backend_src->stream()));
    }

    return true;
}

void ggml_backend_cuda::synchronize()
{
    CUDA_CHECK(cudaStreamSynchronize(stream()));
}

static void ggml_cuda_graph_update_executable(ggml_cuda_graph& graph) {
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph.instance, graph.graph, &result_info);

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("{}: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph.instance));
        graph.instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph.instance, graph.graph, NULL, NULL, 0));
    }
    else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

static bool ggml_cuda_topk_moe_fusion(const ggml_cgraph* cgraph, int node_idx, ggml_cuda_topk_moe_args& args) {
    args.sigmoid = false;
    args.sqrt_softplus = false;
    args.softmax = false;
    args.delayed_softmax = false;
    args.prob_bias = false;
    args.norm = false;

    const int      n_nodes = cgraph->nodes.size();
    const auto nodes = cgraph->nodes.data();

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        }
        else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
            nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        }
        else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor* probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    }
    else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor* probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            }
            else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor* t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
        t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
    const ggml_cgraph* cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache& fused_state_cpy) {
    const ggml_tensor* gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor* src_v = gdn->src[2];
    const int64_t       S_v = src_v->ne[0];
    const int64_t       H = src_v->ne[1];
    const int64_t       n_tokens = src_v->ne[2];
    const int64_t       n_seqs = src_v->ne[3];
    const int64_t       D = S_v * S_v * H;
    const int64_t       K = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor* cpy = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->nodes.size() && cpy == nullptr; ++j) {
        const ggml_tensor* n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor* src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor* dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne.begin()) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t)ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data = (float*)dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t)(dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_cuda_pool& pool, cudaStream_t stream, ggml_cgraph* cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor* node = cgraph->nodes[i];

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
            if constexpr (ggml_cuda_debug_v) {
                GGML_LOG_INFO("{}: fused gated_delta_net snapshot copies for {} (skipped {} nodes)",
                    __func__, node->name, nodes_to_skip);
            }
            const gated_delta_net_context ctx = utils::build_gated_delta_net_context(node);
            gated_delta_net_fused_cache(ctx, fused_state_cpy, stream);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
        cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor* logits = node->src[0];
            ggml_tensor* weights = nullptr;
            ggml_tensor* ids = nullptr;
            const ggml_tensor* bias = nullptr;
            const ggml_tensor* clamp = nullptr;
            const ggml_tensor* scale = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                }
                else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                }
                else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int)ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                }
                else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                        { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (fused::ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                    fused::should_use_topk_moe(node, logits, weights, ids) &&
                    fused::ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    fused::topk_moe(stream, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
            else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights = cgraph->nodes[i + 5];
                ids = cgraph->nodes[i + 1];
                const ggml_tensor* softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (fused::ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                    fused::should_use_topk_moe(softmax, logits, weights, ids) &&
                    fused::ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    fused::topk_moe(stream, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor* rope = cgraph->nodes[i];
        ggml_tensor* set_rows = cgraph->nodes[i + 2];

        op::rope(stream, rope, true, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (fused::ggml_can_fuse_subgraph(cgraph, i,
        { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
        { i + 4 })) {
        const ggml_tensor* mul0 = cgraph->nodes[i];
        const ggml_tensor* sqr = cgraph->nodes[i + 2];
        const ggml_tensor* mul1 = cgraph->nodes[i + 3];
        ggml_tensor* add = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor* x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor* a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor* inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor* x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok = (x->ne[2] == 1 && x->ne[3] == 1) &&
            (add->ne[2] == 1 && add->ne[3] == 1) &&
            (a->ne[2] == 1 && a->ne[3] == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor* sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
            (a->type == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
            (mul0->type == x->type) && (sin1->type == x->type) &&
            (sqr->type == x->type) && (mul1->type == x->type) &&
            (add->type == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
            ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            op::snake_fused(stream, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!fused::ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node = *node;
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src.push_back(cgraph->nodes[i + j + 1]->src[1]);
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                fused::add(stream, &fused_node, n_fuse);
            }
            else {
                fused::mul(stream, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count = 0;

    auto get_mul_mat_scale = [](const ggml_tensor* scale_node, const ggml_tensor* mm_node) -> const ggml_tensor* {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor* scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || scale->nelements() != 1 ||
            !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor* reshape, const ggml_tensor* repeat, const ggml_tensor* getrows,
        const ggml_tensor* scale_node, const ggml_tensor* mm_node) -> const ggml_tensor* {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
            (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor* scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || scale->nelements() != mm_node->src[0]->ne[2] ||
            !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor* bias_node, const ggml_tensor* mul_node, ggml_op op_bias) -> const ggml_tensor* {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx = with_bias ? i + 2 : -1;
                const int up_idx = with_bias ? i + 3 : i + 2;
                const int up_scale_idx = up_idx + 1;
                const int up_bias_idx = with_bias ? up_idx + 2 : -1;
                const int glu_idx = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                }
                else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!fused::ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !fused::ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor* gate_n = cgraph->nodes[gate_idx];
                ggml_tensor* gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor* gate_out_n = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor* up_n = cgraph->nodes[up_idx];
                ggml_tensor* up_scale_n = cgraph->nodes[up_scale_idx];
                ggml_tensor* up_out_n = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor* glu = cgraph->nodes[glu_idx];

                if (!fused::ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                    with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor* gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor* up_scale = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor* up_bias = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor* gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                    !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor* src0 = up_n->src[0];
                const ggml_tensor* src1 = up_n->src[1];
                const ggml_tensor* ids = up_n->src[2];

                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate = gate_n->src[0];
                fusion_data.x_bias = up_bias;
                fusion_data.gate_bias = gate_bias;
                fusion_data.x_scale = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op = ggml_get_glu_op(glu);
                fusion_data.glu_limit = std::bit_cast<float>(glu->op_params[3]);

                if (fused::should_mul_mat_vec_q(up_n)) {
                    op::mul_mat_vec_q(pool, stream, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }
        else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx = with_bias ? i + 5 : -1;
                const int up_idx = with_bias ? i + 6 : i + 5;
                const int up_scale_idx = up_idx + 4;
                const int up_bias_idx = with_bias ? up_idx + 5 : -1;
                const int glu_idx = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = op;
                    ops[7] = GGML_OP_RESHAPE;
                    ops[8] = GGML_OP_REPEAT;
                    ops[9] = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                }
                else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = op;
                    ops[6] = GGML_OP_RESHAPE;
                    ops[7] = GGML_OP_REPEAT;
                    ops[8] = GGML_OP_GET_ROWS;
                    ops[9] = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!fused::ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !fused::ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor* gate_n = cgraph->nodes[gate_idx];
                ggml_tensor* gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor* gate_out_n = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor* up_n = cgraph->nodes[up_idx];
                ggml_tensor* up_scale_n = cgraph->nodes[up_scale_idx];
                ggml_tensor* up_out_n = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor* glu = cgraph->nodes[glu_idx];

                if (!fused::ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                    with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor* gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                    cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor* up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                    cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor* up_bias = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor* gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor* src0 = up_n->src[0];
                const ggml_tensor* src1 = up_n->src[1];
                const ggml_tensor* ids = up_n->src[2];

                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate = gate_n->src[0];
                fusion_data.x_bias = up_bias;
                fusion_data.gate_bias = gate_bias;
                fusion_data.x_scale = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op = ggml_get_glu_op(glu);
                fusion_data.glu_limit = std::bit_cast<float>(glu->op_params[3]);

                if (fused::should_mul_mat_vec_q(up_n)) {
                    op::mul_mat_vec_q(pool, stream, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (fused::ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor* glu = cgraph->nodes[i + 4];
            ggml_tensor* gate_bias_n = glu->src[0];
            ggml_tensor* up_bias_n = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor* gate_n = nullptr;
            ggml_tensor* up_n = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n = cgraph->nodes[i + 2];
            }
            else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n = cgraph->nodes[i];
            }
            else {
                continue;
            }

            const ggml_tensor* up_bias_tensor = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor* gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor* src0 = up_n->src[0];
            const ggml_tensor* src1 = up_n->src[1];
            const ggml_tensor* ids = up_n->src[2];

            if (fused::should_mul_mat_vec_f(up_n)) {
                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate = gate_n->src[0];
                fusion_data.x_bias = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op = ggml_get_glu_op(glu);
                fusion_data.glu_limit = std::bit_cast<float>(glu->op_params[3]);

                op::mul_mat_vec_f(stream, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 5;
                break;
            }

            if (fused::should_mul_mat_vec_q(up_n)) {
                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate = gate_n->src[0];
                fusion_data.x_bias = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op = ggml_get_glu_op(glu);
                fusion_data.glu_limit = std::bit_cast<float>(glu->op_params[3]);

                op::mul_mat_vec_q(pool, stream, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 5;
                break;
            }
        }
        else if (fused::ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor* glu = cgraph->nodes[i + 2];
            ggml_tensor* gate = glu->src[0];
            ggml_tensor* up = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor* src0 = up->src[0];
            const ggml_tensor* src1 = up->src[1];
            const ggml_tensor* ids = up->src[2];

            if (fused::should_mul_mat_vec_f(up)) {
                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate = gate->src[0];
                fusion_data.glu_op = ggml_get_glu_op(glu);
                fusion_data.glu_limit = std::bit_cast<float>(glu->op_params[3]);

                op::mul_mat_vec_f(stream, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 3;
                break;
            }

            if (fused::should_mul_mat_vec_q(up)) {
                op::ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate = gate->src[0];
                fusion_data.glu_op = ggml_get_glu_op(glu);
                fusion_data.glu_limit = std::bit_cast<float>(glu->op_params[3]);

                op::mul_mat_vec_q(pool, stream, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                }
                else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            }
            else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                }
                else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!fused::ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                !fused::ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor* mm_node = cgraph->nodes[i];
            ggml_tensor* scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor* out_node = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor* scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            }
            else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor* bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor* src0 = mm_node->src[0];
            const ggml_tensor* src1 = mm_node->src[1];
            const ggml_tensor* ids = mm_node->src[2];

            op::ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias = bias;
            fusion_data.x_scale = scale;

            if (fused::should_mul_mat_vec_q(mm_node)) {
                op::mul_mat_vec_q(pool, stream, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!fused::ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor* mm_node = cgraph->nodes[i];
        ggml_tensor* bias_node = cgraph->nodes[i + 1];

        ggml_tensor* bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            }
            else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            }
            else {
                continue;
            }
        }
        else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor* src0 = mm_node->src[0];
        const ggml_tensor* src1 = mm_node->src[1];
        const ggml_tensor* ids = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        op::ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (fused::should_mul_mat_vec_f(mm_node)) {
            op::mul_mat_vec_f(stream, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count = 2;
            break;
        }

        if (fused::should_mul_mat_vec_q(mm_node)) {
            op::mul_mat_vec_q(pool, stream, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        op::rms_norm_mul_rope_fused(stream, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        op::rms_norm_mul_rope_fused(stream, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        fused::rms_norm_add(stream, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        fused::rms_norm(stream, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        op::ssm_conv(stream, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        op::ssm_conv(stream, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        op::unary_mul(stream, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        op::relu_sqr(stream, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (fused::ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        fused::softcap(stream, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}

bool supported(int device, bool integrated, const ggml_tensor* node) {
    if (node->buffer->get_type() == ggml_backend_cuda_buffer_type(device)) {
        return true;
    }
    if (auto cuda_host_buft = dynamic_cast<cuda_host_backend_buffer_type*>(node->buffer->get_type())) {
        return integrated;
    }
    return false;
}

void ggml_backend_cuda::graph_evaluate_and_capture(ggml_cgraph* cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void* graph_key)
{
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated = ggml_cuda_info().devices[device].integrated;

    ggml_cuda_stream_context& stream_ctx = stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event* concurrent_event = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor* node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching {} streams at {}\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = stream();  // this should be stream 0
            GGML_ASSERT(curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = this->stream(this->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {

            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto& [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }
            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor*, int> node_to_idx;
                node_to_idx.reserve(cgraph->nodes.size());
                for (int i = 0; i < cgraph->nodes.size(); ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto& [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor* orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        }
                        else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::ranges::sort(sorted_positions);

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i - 1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor*>(event.original_order[i]);
                    }
                }
            }
            else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->nodes.size(); i++) {
                auto node = cgraph->nodes[i];

                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                stream(this->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event = nullptr;
                    }
                    else {
                        GGML_ASSERT(concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to {} for node {}\n", curr_stream_no, node->name);
                    }
                }
                else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor* prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to {} for node {}\n", curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(pool(), stream(), cgraph, i);

                if (nodes_to_skip != 0) {
                    if constexpr (ggml_cuda_debug_v) {
                        const int last_fused = i + nodes_to_skip;
                        GGML_LOG_INFO("nodes_fused: {}, first: {} ({}), last: {} ({})",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
                    }
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(supported(device, integrated, node));
                for (auto& src : node->src) {
                    if (!src) continue;
                    assert(src->buffer);
                    assert(supported(device, integrated, src));
                }
#endif // NDEBUG

                bool ok = compute_forward(node);
                if (!ok) {
                    GGML_LOG_ERROR("{}: op not supported {} ({})", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
                }
            }
        }

        if constexpr (use_cuda_graph_v) {
            ggml_cuda_graph* graph = cuda_graph(graph_key);
            if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
                if (graph->graph != nullptr) {
                    CUDA_CHECK(cudaGraphDestroy(graph->graph));
                    graph->graph = nullptr;
                }

                CUDA_CHECK(cudaStreamEndCapture(stream(), &graph->graph));
                graph_evaluated_or_captured = true; // CUDA graph has been captured

                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                    ggml_cuda_lock_cv.notify_all();
                }
            }
            else {
                graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
            }
        } else {
            graph_evaluated_or_captured = true;
        }
    }

    if constexpr (use_cuda_graph_v) {
        if (use_cuda_graph) {
            ggml_cuda_graph* graph = cuda_graph(graph_key);
            if (graph->instance == nullptr) { // Create executable graph from captured graph.
                CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
            }
            if (cuda_graph_update_required) { // Update graph executable
                ggml_cuda_graph_update_executable(*graph);
            }
            // Launch graph
            CUDA_CHECK(cudaGraphLaunch(graph->instance, stream()));
        }
    }
}

static const void* ggml_cuda_graph_get_key(ggml_cgraph* cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_cuda_graph& graph, ggml_cgraph* cgraph) {
    bool res = false;

    const void* graph_key = ggml_cuda_graph_get_key(cgraph);

    if (graph.instance == nullptr) {
        res = true;
    }

    if (cgraph->uid != 0 &&
        cgraph->uid == graph.uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph.node_props.size() == cgraph->nodes.size());
        return false;
    }

    graph.uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph.node_props.size() != cgraph->nodes.size()) {
        res = true;
        graph.node_props.resize(cgraph->nodes.size());
    }

    for (int i = 0; i < cgraph->nodes.size(); i++) {
        ggml_cuda_graph::node_properties prop = {};
        prop.node = *cgraph->nodes[i];

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne.data(), sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb.data(), sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph.node_props[i], &prop, sizeof(prop)) != 0) {
            graph.node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph* cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->nodes.size(); i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (utils::ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("{}: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

enum ggml_status ggml_backend_cuda::graph_compute_impl(ggml_cgraph* cgraph)
{
    ggml_cuda_set_device(device);

    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
    const void* graph_key = nullptr;

    if constexpr (use_cuda_graph_v) {
        graph_key = ggml_cuda_graph_get_key(cgraph);

        ggml_cuda_graph& graph = *cuda_graph(graph_key);

        graph_set_enabled(graph_key);

        if (graph.is_enabled()) {
            const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
            if (graph_compatible) {
                const bool properties_changed = ggml_cuda_graph_update_required(graph, cgraph);

                if (!graph.warmup_complete) {
                    // Warmup: need at least 2 calls with no property change on the 2nd call
                    if (!properties_changed) {
                        graph.warmup_complete = true;
                        GGML_LOG_DEBUG("{}: CUDA graph warmup complete\n", __func__);
                        use_cuda_graph = true;
                        cuda_graph_update_required = true;
                    }
                    // else: properties changed or first call - execute directly (use_cuda_graph stays false)
                }
                else {
                    // Post-warmup: normal CUDA graph operation
                    if (properties_changed) {
                        // Properties changed - reset warmup, execute directly until stable again
                        graph.warmup_complete = false;
                        GGML_LOG_DEBUG("{}: CUDA graph warmup reset\n", __func__);
                    }
                    else {
                        use_cuda_graph = true;
                        cuda_graph_update_required = graph.instance == nullptr;
                    }
                }
            }
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(stream(), cudaStreamCaptureModeRelaxed));
    }

    graph_evaluate_and_capture(cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_cuda::event_record(ggml_backend_event* event)
{
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, stream()));
}

void ggml_backend_cuda::event_wait(ggml_backend_event* event)
{
    if (true) {
        CUDA_CHECK(cudaStreamWaitEvent(stream(), (cudaEvent_t)event->context, 0));
    }
    else {
#if 0
        // untested
        auto wait_fn = [](void* user_data) {
            ggml_backend_event* event = (ggml_backend_event*)user_data;
            ggml_backend_event_synchronize(event);
            };

        CUDA_CHECK(cudaLaunchHostFunc(stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

bool ggml_backend_cuda::compute_forward(ggml_tensor* dst) {
    switch (dst->op) {
    case GGML_OP_ARGMAX:
        op::argmax(stream(), dst);
        break;
    case GGML_OP_COUNT_EQUAL:
        op::count_equal(stream(), dst);
        break;
    case GGML_OP_REPEAT:
        op::repeat(stream(), dst);
        break;
    case GGML_OP_REPEAT_BACK:
        op::repeat_back(stream(), dst);
        break;
    case GGML_OP_GET_ROWS:
        op::get_rows(stream(), dst);
        break;
    case GGML_OP_GET_ROWS_BACK:
        op::get_rows_back(stream(), dst);
        break;
    case GGML_OP_SET_ROWS:
        op::set_rows(stream(), dst);
        break;
    case GGML_OP_SET:
        op::set(stream(), dst);
        break;
    case GGML_OP_DUP:
    case GGML_OP_CONT:
        op::dup(stream(), dst);
        break;
    case GGML_OP_CPY:
        op::cpy(stream(), dst);
        break;
    case GGML_OP_ADD:
    case GGML_OP_ADD1: // TODO: more efficient implementation
        op::add(stream(), dst);
        break;
    case GGML_OP_ADD_ID:
        op::add_id(stream(), dst);
        break;
    case GGML_OP_SUB:
        op::sub(stream(), dst);
        break;
    case GGML_OP_ACC:
        op::acc(stream(), dst);
        break;
    case GGML_OP_MUL:
        op::mul(stream(), dst);
        break;
    case GGML_OP_DIV:
        op::div(stream(), dst);
        break;
    case GGML_OP_UNARY:
        op::unary(stream(), dst);
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            op::glu(stream(), dst);
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
            op::swiglu_oai(stream(), dst);
			break;
        case GGML_GLU_OP_SWIGLU_CLAMP:
            op::swiglu_clamp(stream(), dst);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_NORM:
        op::norm(stream(), dst);
        break;
    case GGML_OP_GROUP_NORM:
        op::group_norm(stream(), dst);
        break;
    case GGML_OP_L2_NORM:
        op::l2_norm(stream(), dst);
        break;
    case GGML_OP_CONCAT:
        op::concat(stream(), dst);
        break;
    case GGML_OP_UPSCALE:
        op::upscale(stream(), dst);
        break;
    case GGML_OP_PAD:
        op::pad(stream(), dst);
        break;
    case GGML_OP_PAD_REFLECT_1D:
        op::pad_reflect_1d(stream(), dst);
        break;
    case GGML_OP_ARANGE:
        op::arange(stream(), dst);
        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        op::timestep_embedding(stream(), dst);
        break;
    case GGML_OP_LEAKY_RELU:
        op::leaky_relu(stream(), dst);
        break;
    case GGML_OP_SILU_BACK:
        op::silu_back(stream(), dst);
        break;
    case GGML_OP_RMS_NORM:
        op::rms_norm(stream(), dst);
        break;
    case GGML_OP_RMS_NORM_BACK:
        op::rms_norm_back(stream(), dst);
        break;
    case GGML_OP_MUL_MAT:
        mul_mat(dst);
        break;
    case GGML_OP_MUL_MAT_ID:
        op::mul_mat_id(pool(), stream(), dst, [this](ggml_tensor* dst) {
            mul_mat(dst);
        });
        break;
    case GGML_OP_OUT_PROD:
        op::out_prod(pool(), stream(), cublas_handle(), dst);
        break;
    case GGML_OP_SCALE:
        op::scale(stream(), dst);
        break;
    case GGML_OP_SQR:
        op::sqr(stream(), dst);
        break;
    case GGML_OP_SQRT:
        op::sqrt(stream(), dst);
        break;
    case GGML_OP_SIN:
        op::sin(stream(), dst);
        break;
    case GGML_OP_COS:
        op::cos(stream(), dst);
        break;
    case GGML_OP_CLAMP:
        op::clamp(stream(), dst);
        break;
    case GGML_OP_LOG:
        op::log(stream(), dst);
        break;
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        break;
    case GGML_OP_DIAG:
        op::diag(stream(), dst);
        break;
    case GGML_OP_DIAG_MASK_INF:
        op::diag_mask_inf(stream(), dst);
        break;
    case GGML_OP_SOFT_MAX:
        op::soft_max(pool(), stream(), dst);
        break;
    case GGML_OP_SOFT_MAX_BACK:
        op::soft_max_back(stream(), dst);
        break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        op::rope(stream(), dst, (dst->op == GGML_OP_ROPE) ? true : false);
        break;
    case GGML_OP_ROLL:
        op::roll(stream(), dst);
        break;
    case GGML_OP_IM2COL:
        op::im2col(stream(), dst);
        break;
    case GGML_OP_IM2COL_3D:
        op::im2col_3d(stream(), dst);
        break;
    case GGML_OP_CONV_2D:
        op::conv2d(stream(), dst);
        break;
    case GGML_OP_CONV_2D_DW:
        op::conv2d_dw(stream(), dst);
        break;
    case GGML_OP_CONV_TRANSPOSE_2D:
        op::conv_2d_transpose_p0(stream(), dst);
        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        op::conv_transpose_1d(stream(), dst);
        break;
    case GGML_OP_COL2IM_1D:
        op::col2im_1d(stream(), dst);
        break;
    case GGML_OP_POOL_2D:
        op::pool2d(stream(), dst);
        break;
    case GGML_OP_POOL_1D:
        op::pool1d(stream(), dst);
        break;
    case GGML_OP_SUM:
        op::sum(pool(), stream(), dst);
        break;
    case GGML_OP_CUMSUM:
        op::cumsum(pool(), stream(), dst);
        break;
    case GGML_OP_SUM_ROWS:
        op::sum_rows(stream(), dst);
        break;
    case GGML_OP_MEAN: {
        const bool any_cuda_graph_has_instance = [=, this]() {
            if constexpr (use_cuda_graph_v) {
                for (const auto& [_, graph] : cuda_graphs) {
                    if (graph && graph->instance != nullptr) {
                        return true;
                    }
                }
                return false;
            }
            else {
                return false;
            }
        }();
        const bool any_cuda_graph_enabled = [=, this]() {
            if constexpr (use_cuda_graph_v) {
                for (const auto& [key, graph] : cuda_graphs) {
                    if (graph && graph->is_enabled()) {
                        return true;
                    }
                }
                return false;
            }
            else {
                return false;
            }
        }();
        op::mean(pool(), stream(), any_cuda_graph_has_instance, any_cuda_graph_enabled, dst);
        break;
    }
    case GGML_OP_SSM_CONV:
        op::ssm_conv(stream(), dst);
        break;
    case GGML_OP_TOP_K:
        op::top_k(pool(), stream(), dst);
        break;
    case GGML_OP_SSM_SCAN:
        op::ssm_scan(pool(), cublas_handle(), stream(), dst);
        break;
    case GGML_OP_ARGSORT:
        op::argsort(pool(), stream(), dst);
        break;
    case GGML_OP_FLASH_ATTN_EXT:
        op::flash_attn_ext(device, pool(), stream(), dst);
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS:
        op::cross_entropy_loss(pool(), stream(), dst);
        break;
    case GGML_OP_TRI:
        op::tri(stream(), dst);
        break;
    case GGML_OP_RWKV_WKV6:
        op::rwkv_wkv6(stream(), dst);
        break;
    case GGML_OP_GATED_LINEAR_ATTN:
        op::gated_linear_attn(stream(), dst);
        break;
    case GGML_OP_GATED_DELTA_NET:
        op::gated_delta_net(stream(), dst);
        break;
    case GGML_OP_DSV4_HC_COMB:
        op::dsv4_hc_comb(stream(), dst);
        break;
    case GGML_OP_DSV4_HC_PRE:
        op::dsv4_hc_pre(stream(), dst);
        break;
    case GGML_OP_DSV4_HC_POST:
        op::dsv4_hc_post(stream(), dst);
        break;
    case GGML_OP_RWKV_WKV7:
        op::rwkv_wkv7(stream(), dst);
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        op::cross_entropy_loss_back(pool(), stream(), dst);
        break;
    case GGML_OP_OPT_STEP_ADAMW:
        op::opt_step_adamw(stream(), dst);
        break;
    case GGML_OP_OPT_STEP_SGD:
        op::opt_step_sgd(stream(), dst);
        break;
    case GGML_OP_SOLVE_TRI: {
        const int id = ggml_cuda_get_device();
        op::solve_tri(pool(id), cublas_handle(), stream(), dst);
        break;
    }
    case GGML_OP_FILL:
        op::fill(stream(), dst);
        break;
    case GGML_OP_LIGHTNING_INDEXER:
        op::lightning_indexer(stream(), dst);
        break;
    default:
        return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("{}: {} failed", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

ggml_backend_cuda::~ggml_backend_cuda()
{
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, [] { return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}

bool ggml_backend_cuda::graph_set_enabled(const void* graph_key) {
    if constexpr (use_cuda_graph_v) {
        ggml_cuda_graph* graph = cuda_graph(graph_key);

        if (graph->graph == nullptr) {
            if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_VOLTA) {
                graph->disable_due_to_gpu_arch = true;
                GGML_LOG_DEBUG("{}: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
        }

        return graph->is_enabled();
    }
    else {
        return false;
    }
}

cudaStream_t ggml_backend_cuda::stream(int device, int stream) {
    if (streams[device][stream] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
    }
    return streams[device][stream];
}

ggml_cuda_pool& ggml_backend_cuda::pool(int device) {
    if (pools[device][curr_stream_no] == nullptr) {
        pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no);
    }
    return *pools[device][curr_stream_no];
}

cublasHandle_t ggml_backend_cuda::cublas_handle() {
    if (cublas_handles[device][curr_stream_no] == nullptr) {
        ggml_cuda_set_device(device);
        CUBLAS_CHECK(cublasCreate(&cublas_handles[device][curr_stream_no]));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device][curr_stream_no], CUBLAS_TF32_TENSOR_OP_MATH));
        CUBLAS_CHECK(cublasSetStream(cublas_handles[device][curr_stream_no], stream()));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUBLAS_VER_MAJOR > 11 || (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR >= 2))
        if (cublas_workspace_sizes[device] == 0) {
            const int cc = ggml_cuda_info().devices[device].cc;
            cublas_workspace_sizes[device] = (cc >= GGML_CUDA_CC_HOPPER) ? 32 * 1024 * 1024 : 4 * 1024 * 1024;
        }
        CUDA_CHECK(cudaMalloc(&cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
        CUBLAS_CHECK(cublasSetWorkspace(cublas_handles[device][curr_stream_no], cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
#endif
    }
    return cublas_handles[device][curr_stream_no];
}

void ggml_backend_cuda::graph_optimize(ggml_cgraph* cgraph) {

#ifdef USE_CUDA_GRAPH
    const void* graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(graph_key);
#else
    const bool use_cuda_graph = false;
#endif

    static bool enable_graph_optimization = [] {
        const char* env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context& stream_context = this->stream_context();
    stream_context.reset();

    if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
        return;
    }

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor*, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor*, int> node_indices;

    const auto& is_noop = [](const ggml_tensor* node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto& depends_on = [](const ggml_tensor* dst, const ggml_tensor* src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor* dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor* src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->nodes.size(); node_idx++) {
        const ggml_tensor* node = cgraph->nodes[node_idx];
        node_indices[node] = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor* src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto& [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!root_node->name.starts_with("attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto& [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor*>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->nodes.size(); ++i) {
                const ggml_tensor* node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t)count);

            //find the join point
            const ggml_tensor* join_node = nullptr;

            const auto& belongs_to_branch = [&](const ggml_tensor* node,
                                                const std::vector<const ggml_tensor*>& branch) -> bool {
                for (const ggml_tensor* n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->nodes.size(); ++i) {
                const ggml_tensor* curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor*>& branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor* n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx = fork_node_idx + 1;
                const int n_branches = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor*> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor*, ggml_cuda_concurrent_event>& concurrent_events = this->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                //GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor*>& branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor*> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor*>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor*>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}
