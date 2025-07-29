module;
#include <assert.h>
#include <bit>
#include <condition_variable>
#include <memory>
#include <span>
#include <vector>
#include "block.h"
#include "common.h"
#include "cuda_pool.h"
#include "config.h"
#include "op/convert.cuh"
#include "op/cuda_func.h"
#include "cuda_config.h"
#include "vendor_constant.h"

#define GGML_ABORT(...)
#define GGML_ASSERT(...) assert(__VA_ARGS__)

#define GGML_UNUSED(x) (void)(x)
#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);

#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

module ggml;
import :cuda.backend;
import :cuda.op;

static void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* /*src1_ddf_i*/,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                              || (GGML_CUDA_CC_IS_AMD(cc) && GGML_CUDA_CC_IS_CDNA3(cc)))
                              && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int*)src1_ddq_i, nullptr, nullptr, dst_dd_i,
        src0->ne[0], row_diff, src1_ncols, stride01, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k };

    ggml_cuda_mul_mat_q_switch_type(ctx.pool(id), args, stream);
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char* /*src1_ddq_i*/, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t /*src1_padded_row_size*/, cudaStream_t stream)
{

    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(src0_dd_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;

    const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);

    const bool use_fp16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT;

    if (supports_bf16 && src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            size_t ne = src1_ncols * ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1->type, src1_ddf_i, src1_as_bf16.get(), ne, stream);
        }
        const nv_bfloat16* src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16*)src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16* src0_ptr = (const nv_bfloat16*)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff * src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                row_diff, src1_ncols, ne10,
                &alpha_f32, src0_ptr, CUDA_R_16BF, ne00,
                src1_ptr, CUDA_R_16BF, ne10,
                &beta_f32, dst_bf16.get(), CUDA_R_16BF, ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        to_fp32_cuda(GGML_TYPE_BF16, dst_bf16.get(), dst_dd_i, row_diff * src1_ncols, stream);
    }
    else if (fast_fp16_hardware_available(cc) && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            size_t ne = row_diff * ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0->type, src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const half* src0_ptr = src0->type == GGML_TYPE_F16 ? (const half*)src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            size_t ne = src1_ncols * ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1->type, src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half* src1_ptr = src1->type == GGML_TYPE_F16 ? (const half*)src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ptr, CUDA_R_16F, ne00,
                    src1_ptr, CUDA_R_16F, ne10,
                    &beta, dst_dd_i, CUDA_R_32F, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff * src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr, CUDA_R_16F, ne00,
                    src1_ptr, CUDA_R_16F, ne10,
                    &beta_f16, dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            to_fp32_cuda(GGML_TYPE_F16, dst_f16.get(), dst_dd_i, row_diff * src1_ncols, stream);
        }
    }
    else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            src0_ddq_as_f32.alloc(row_diff * ne00);
            to_fp32_cuda(src0->type, src0_dd_i, src0_ddq_as_f32.get(), row_diff * ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            src1_ddq_as_f32.alloc(src1_ncols * ne10);
            to_fp32_cuda(src1->type, src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols * ne10, stream);
        }

        const float* src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float*)src0_dd_i : src0_ddq_as_f32.get();
        const float* src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float*)src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                row_diff, src1_ncols, ne10,
                &alpha, src0_ddf_i, ne00,
                src1_ddf1_i, ne10,
                &beta, dst_dd_i, ldc));
    }
}

void ggml_cuda_op_mul_mat_vec(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char*, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t, cudaStream_t stream)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0 = dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_col_dst = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer

    mul_mat_vec_context ctx1 {
        .src0_type = src0->type,
        .prec = prec,
        .src0_d = src0_dd_i,
        .src1_d = (const float*)src1_ddf_i,
        .ids_d = nullptr,
        .dst_d = (float*)dst->data,
        .ncols = ne00,
        .nrows = row_diff,
		.ncols_dst = src1_ncols,
        .stride_row = ne00,
        .stride_col_y = ne10,
        .stride_col_dst = stride_col_dst,
        .nchannels_x = 1,
        .nchannels_y = 1,
        .nchannels_dst = 1,
        .stride_channel_x = 0,
        .stride_channel_y = 0,
        .stride_channel_dst = 0,
        .nsamples_x = 1,
        .nsamples_dst = 1,
        .stride_sample_x = 0,
        .stride_sample_y = 0,
        .stride_sample_dst = 0
    };
    mul_mat_vec_cuda(&ctx1, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream)
{
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    mat_vec_q_switch_context ctx1 {
        .type_x = src0->type,
        .vx = src0_dd_i,
        .vy = src1_ddq_i,
        .ids = nullptr,
		.dst = dst_dd_i,
        .ncols_x = ne00,
        .nrows_x = row_diff,
		.ncols_dst = src1_ncols,
        .stride_row_x = stride_row_x,
        .stride_col_y = stride_col_y,
        .stride_col_dst = nrows_dst,
        .nchannels_x = 1,
        .nchannels_y = 1,
        .nchannels_dst = 1,
        .stride_channel_x = 1,
        .stride_channel_y = 1,
        .stride_channel_dst = 1,
        .nsamples_x = 1,
        .nsamples_dst = 1,
        .stride_sample_x = 1,
        .stride_sample_y = 1,
        .stride_sample_dst = 1
    };

    mul_mat_vec_q_switch_type(&ctx1, stream);
}

// pool with virtual memory
#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
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
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

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
#endif // !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)

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
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
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
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
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

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(dynamic_cast<cuda_split_backend_buffer_type*>(src0->buffer->get_type()) == nullptr);
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = dst->nelements();
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float* dst_ddf = (float*)dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t* src0_ptr = nullptr;
    const cuda_t* src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    // Handle src0
    src0_ptr = (const cuda_t*)src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t*)src1->data;
    }
    else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = src1->nelements();
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11 * s11;
        s13 = ne12 * s12;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char* dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void* alpha = traits::get_alpha();
    const void* beta = traits::get_beta();
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char*)dst_ddf;  // Direct F32 output
        }
        else {
            dst_t = (char*)dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }
    else {
        dst_t = (char*)dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        cu_compute_type = CUBLAS_COMPUTE_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    if (r2 == 1 && r3 == 1 && ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1)) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
            cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01 / nb00, nb02 / nb00, // strideA
                src1_ptr, cu_data_type_b, s11, s12,       // strideB
                beta, dst_t, cu_data_type, ne0, ne1 * ne0,   // strideC
                ne12 * ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12 * ne13;

        ggml_cuda_pool_alloc<const void*> ptrs_src(ctx.pool(), 2 * ne23);
        ggml_cuda_pool_alloc<      void*> ptrs_dst(ctx.pool(), 1 * ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        k_compute_batched_ptrs_cuda(
            src0_ptr, src1_ptr, dst_t,
            ptrs_src.get(), ptrs_dst.get(),
            ne12, ne13,
            ne23,
            nb02, nb03,
            (src1->type == src0_type) ? nb12 : s12 * src1_stride_size,
            (src1->type == src0_type) ? nb13 : s13 * src1_stride_size,
            nbd2, nbd3,
            r2, r3, main_stream);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
            cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void**)(ptrs_src.get() + 0 * ne23), cu_data_type_a, nb01 / nb00,
                (const void**)(ptrs_src.get() + 1 * ne23), cu_data_type_b, s11,
                beta, (void**)(ptrs_dst.get() + 0 * ne23), cu_data_type, ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        to_fp32_cuda(traits::ggml_type_val, dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
    case GGML_TYPE_F32:
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
        break;
    case GGML_TYPE_BF16:
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
        break;
    case GGML_TYPE_F16:
        ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
        break;
    default:
        GGML_ABORT("Unsupported type");
    }
}

void ggml_backend_cuda::mul_mat(ggml_tensor* dst)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const bool split = to_split_buffer_type(src0->buffer->get_type()) != nullptr;

    bool use_mul_mat_vec = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16 = false;

    if (split) {
        auto buft_ctx = (cuda_split_backend_buffer_type*)src0->buffer->get_type();
        auto& tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc = ggml_cuda_info().devices[id].cc;
            use_mul_mat_q = use_mul_mat_q && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
            use_mul_mat_vec = use_mul_mat_vec && ggml_cuda_should_use_mmv(src0->type, cc, src0->ne.data(), src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_hardware_available(cc);
        }
    }
    else {
        const int cc = ggml_cuda_info().devices[device].cc;
        use_mul_mat_q = use_mul_mat_q && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
        use_mul_mat_vec = use_mul_mat_vec && ggml_cuda_should_use_mmv(src0->type, cc, src0->ne.data(), src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_hardware_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    //TODO update for generic tensor parallelism
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16 = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32 = src0->type == GGML_TYPE_F32;

    if (!split && use_mul_mat_vec) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        op::mul_mat_vec(this->stream(), nullptr, dst);
    }
    else if (!split && use_mul_mat_vec_q) {
        op::mul_mat_vec_q(stream(), nullptr, dst, pool());
    }
    else if (!split && use_mul_mat_q) {
        op::mul_mat_q(stream(), pool(), nullptr, dst);
    }
    else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(*this, src0, src1, dst);
    }
    else if (use_mul_mat_vec) {
        op_mul_mat(dst, ggml_cuda_op_mul_mat_vec, nullptr);
    }
    else if (use_mul_mat_vec_q) {
        op_mul_mat(dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    }
    else if (use_mul_mat_q) {
        op_mul_mat(dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    }
    else {
        op_mul_mat(dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void* dst, const struct ggml_tensor* src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    const char* src_ptr = (const char*)src->data;
    char* dst_ptr = (char*)dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char* x = src_ptr + i1_low * nb1 + i2 * nb2 + i3 * nb3;
    if (nb0 == ts && nb1 == ts * ne0 / bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff * nb1, cudaMemcpyDeviceToDevice, stream);
    }
    else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts * ne0 / bs, x, nb1, ts * ne0 / bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    }
    else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void* rx = (const void*)((const char*)x + i1 * nb1);
            void* rd = (void*)(dst_ptr + i1 * ts * ne0 / bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts / bs, rx, nb0, ts / bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

void ggml_backend_cuda::op_mul_mat(
    ggml_tensor* dst,
    ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    auto src1_ctx = (cuda_backend_buffer*)src1->buffer;
    auto dst_ctx = (cuda_backend_buffer*)dst->buffer;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;
    const int64_t i03_divisor = ne13 / ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = to_split_buffer_type(src0->buffer->get_type()) != nullptr;
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    ggml_tensor_extra_gpu* src0_extra = split ? (ggml_tensor_extra_gpu*)src0->extra : nullptr;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        auto buft_ctx = (cuda_split_backend_buffer_type*)src0->buffer->get_type();
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char* src0_dd = nullptr;
        float* src1_ddf = nullptr; // float
        char* src1_ddq = nullptr; // q8_1
        float* dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(tensor_split);

            if (id != 0) {
                dev[id].row_low = ne01 * tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high = ne01 * tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = this->stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char*)src0_extra->data_device[id] : (char*)src0->data;
        }
        else {
            // If src0 is not contiguous it will be copied to a temporary buffer.
            // This buffer needs to be cleared entirely because multiple regions will function as padding.
            const size_t nbytes_data = src0->nbytes();
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(pool(id), nbytes_data + nbytes_padding);
            // TODO: remove this for MUSA once the Guilty Lockup issue is resolved
#ifndef GGML_USE_MUSA
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
#else // GGML_USE_MUSA
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
#endif // !GGML_USE_MUSA
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && src0->buffer->getUsage() == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            const size_t nbytes_data = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low) * ne00);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float*)src1->data;
        }
        else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(pool(id), src1->nelements());
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1 * src1_padded_col_size * q8_1_ts / q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc) * sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(pool(id), src_1_ddq_size);
            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(
                    dev[id].src1_ddf, nullptr, dev[id].src1_ddq, src0->type, src1->ne[0],
                    src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                    src1_padded_col_size, src1->ne[1], src1->ne[2], src1->ne[3], stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float*)dst->data;
        }
        else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low) * ne1 : dst->nelements();
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(pool(id), size_dst_ddf);
        }

        // if multiple devices are used they need to wait for the main device
        // here an event is recorded that signals that the main device has finished calculating the input data
        if (split && used_devices > 1) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaEventRecord(src0_extra->events[device][0], this->stream()));
        }
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0 / src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = this->stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13 * ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0 * ne11 * src1_padded_col_size * q8_1_ts / q8_1_bs;

                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                }
                else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size * q8_1_ts / q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                const size_t nbytes_src0_matrix = ne01 * ne00 * src0_ts / src0_bs;
                char* src0_dd_i = dev[id].src0_dd + ((i03 / i03_divisor) * ne02 + (i02 / i02_divisor)) * nbytes_src0_matrix;
                float* src1_ddf_i = dev[id].src1_ddf + (i0 * ne11 + src1_col_0) * ne10;
                char* src1_ddq_i = dev[id].src1_ddq + src1_ddq_i_offset;
                float* dst_dd_i = dev[id].dst_dd + (i0 * ne1 + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != device) {
                        // TODO
                        assert(false);
#if 0
                        if (quantize_src1) {
                            char* src1_ddq_i_source = dev[device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11 * sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols * sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size / (4 * QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, device, pitch, width, height, stream));
                            }
                            else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs, stream));
                            }
                        }
                        else {
                            float* src1_ddf_i_source = (float*)src1->data;
                            src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                src1_ncols * ne10 * sizeof(float), stream));
                        }
#endif
                    }
                }
                else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0 + src1_ncols, stream));
                }
                else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(
                        src1_ddf_i, nullptr, src1_ddq_i, src0->type, src1->ne[0], src1->ne[0],
                        src1->ne[1] * src1->ne[0], src1->ne[2] * src1->ne[1] * src1->ne[0],
                        src1_padded_col_size, src1_ncols, 1, 1, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02 / i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(*this, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    assert(false);
#if 0
                    void* dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float* dhf_dst_i = (float*)((char*)dst_off_device + i02 * nb2 + i03 * nb3);
                        GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                        dhf_dst_i += src1_col_0 * ne0 + dev[id].row_low;
                        CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                            dhf_dst_i, device, ne0 * sizeof(float), dst_dd_i, id, row_diff * sizeof(float), row_diff * sizeof(float), src1_ncols, stream));
                    }
                    else {
                        float* dhf_dst_i = (float*)((char*)dst_off_device + i02 * nb2 + i03 * nb3);
                        GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                        dhf_dst_i += src1_col_0 * ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols * ne0 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
#endif
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != device || is != 0)) {
                    assert(false);
                    //CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }

            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(this->stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

void ggml_backend_cuda::set_tensor_async_impl(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, stream()));
}

void ggml_backend_cuda::get_tensor_async_impl(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
{
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost, stream()));
}

bool ggml_backend_cuda::cpy_tensor_async(ggml_backend_t backend_src, const ggml_tensor* src, ggml_tensor* dst)
{
#if 0
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(this)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context* cuda_ctx_src = (ggml_backend_cuda_context*)backend_src->context;

    ggml_backend_cuda_buffer_context* buf_ctx_src = (ggml_backend_cuda_buffer_context*)buf_src->context;
    ggml_backend_cuda_buffer_context* buf_ctx_dst = (ggml_backend_cuda_buffer_context*)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("{}: backend and buffer devices do not match", __func__);
#endif
        return false;
    }

    if (backend_src != this) {
        // copy on src stream
        if (cuda_ctx_src->device == device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, dst->nbytes(), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        }
        else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(stream(), cuda_ctx_src->copy_event, 0));
    }
    else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
#endif
    return true;
}

void ggml_backend_cuda::synchronize()
{
    CUDA_CHECK(cudaStreamSynchronize(stream()));
}

// nicer C++ syntax for ggml_can_fuse
inline bool ggml_can_fuse(const struct ggml_cgraph* cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph* cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor* rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor* mul = cgraph->nodes[node_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm->src[1])) {
            return false;
        }

        //rms_norm kernel assumes contigous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    }

    return true;
}

void ggml_backend_cuda::evaluate_and_capture_cuda_graph(ggml_cgraph* cgraph,
    [[maybe_unused]] std::vector<void*>& ggml_cuda_cpy_fn_ptrs, bool& graph_evaluated_or_captured, bool& use_cuda_graph,
    bool& cuda_graph_update_required) {
    // flag used to determine whether it is an integrated_gpu
    const bool integrated = ggml_cuda_info().devices[device].integrated;

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            for (size_t i = 0; i < cgraph->nodes.size(); i++) {
                auto node = cgraph->nodes[i];
                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

                static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
                if (!disable_fusion && ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                    op::rms_norm_fused(stream(), node, cgraph->nodes[i + 1]);
                    i++;
                    continue;
                }
#ifndef NDEBUG
                assert(node->buffer->get_type() == ggml_backend_cuda_buffer_type(device));
                for (auto& src : node->src) {
                    if (!src) continue;
                    assert(src->buffer);
                    assert(buffer_type_from_device(src->buffer->get_type(), device));
                }
#else
                GGML_UNUSED(integrated);
#endif // NDEBUG

                bool ok = compute_forward(node);
                if (!ok) {
                    GGML_LOG_ERROR("{}: op not supported {} ({})", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }
        }

        if constexpr (use_cuda_graph_v) {
#ifdef USE_CUDA_GRAPH
            if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
                if (cuda_ctx->cuda_graph->graph != nullptr) {
                    CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                    cuda_ctx->cuda_graph->graph = nullptr;
                }

                CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));
                graph_evaluated_or_captured = true; // CUDA graph has been captured

                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                    ggml_cuda_lock_cv.notify_all();
                }
            }
            else {
                graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
            }
        }

        if (use_cuda_graph) {
            if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
                CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
            }

            // Perform update to graph (if required for this token), and change copy parameter (required for every token)
            maintain_cuda_graph(cuda_ctx, ggml_cuda_cpy_fn_ptrs, cuda_graph_update_required);

            // Update graph executable
            update_cuda_graph_executable(cuda_ctx);

            // Launch graph
            CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#endif  // USE_CUDA_GRAPH
        }
        else {
            graph_evaluated_or_captured = true;
        }
    }
}

enum ggml_status ggml_backend_cuda::graph_compute_impl(ggml_cgraph* cgraph)
{
    ggml_cuda_set_device(device);

    // vector of pointers to CUDA cpy kernels, which are required to identify
    // kernel parameters which need updated in the graph for each token
    std::vector<void*> ggml_cuda_cpy_fn_ptrs;

    auto [use_cuda_graph, cuda_graph_update_required] = [&]() -> std::pair<bool, bool> {
        if constexpr (not use_cuda_graph_v) {
			return { false, false };
        }
        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

        // Objects required for CUDA Graph
        bool use_cuda_graph = true;
        bool cuda_graph_update_required = false;

        if (cuda_graph.graph == nullptr) {
            if (ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE) {
                    cuda_graph.disable_due_to_gpu_arch = true;
#ifndef NDEBUG
                    GGML_LOG_DEBUG("{}: disabling CUDA graphs due to GPU architecture", __func__);
#endif
            }
        }

        // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
        // or previous graph capture failure.
        // Also disable for multi-gpu for now. TO DO investigate
        if (disable_cuda_graphs_due_to_env
            || cuda_graph.disable_due_to_gpu_arch
            || cuda_graph.disable_due_to_too_many_updates
            || cuda_graph.disable_due_to_failed_graph_capture) {
            use_cuda_graph = false;
        }
        return { use_cuda_graph, cuda_graph_update_required };
#if 0
        if (use_cuda_graph) {
            cuda_graph_update_required = is_cuda_graph_update_required(cuda_ctx, cgraph);

            use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, cgraph,
                ggml_cuda_cpy_fn_ptrs, use_cuda_graph);

            // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
            if (use_cuda_graph && cuda_graph_update_required) {
                cuda_ctx->cuda_graph->number_consecutive_updates++;
            }
            else {
                cuda_ctx->cuda_graph->number_consecutive_updates = 0;
            }

            if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
                cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
                GGML_LOG_DEBUG("{}: disabling CUDA graphs due to too many consecutive updates", __func__);
#endif
            }
        }
        if (use_cuda_graph && cuda_graph_update_required) {
            // Start CUDA graph capture
            {
                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
            }

            CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
        }

        if (!use_cuda_graph) {
            cuda_ctx->cuda_graph->use_cpy_indirection = false;
        }
#endif
    }();

    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cgraph, ggml_cuda_cpy_fn_ptrs, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

void ggml_backend_cuda::event_record(ggml_backend_event_t event)
{
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, stream()));
}

void ggml_backend_cuda::event_wait(ggml_backend_event_t event)
{
    if (ggml_backend_is_cuda(this)) {
        CUDA_CHECK(cudaStreamWaitEvent(stream(), (cudaEvent_t)event->context, 0));
    }
    else {
#if 0
        // untested
        auto wait_fn = [](void* user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
            };

        CUDA_CHECK(cudaLaunchHostFunc(stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

bool ggml_backend_cuda::compute_forward(ggml_tensor* dst) {
    // why is this here instead of mul_mat?
    if (dst->src.size() > 1 && dst->src[0] != nullptr && to_cuda_buffer_type(dst->src[0]->buffer->get_type())) {
        const ggml_tensor* src1 = dst->src[1];
        if (src1) ggml_cuda_set_peer_access(src1->ne[1], device);
    }

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
        op::mul_mat_id(stream(), dst, pool(), [this](ggml_tensor* dst) {
            mul_mat(dst);
        });
        break;
    case GGML_OP_OUT_PROD:
        op::out_prod(stream(), cublas_handle(), dst);
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
    case GGML_OP_DIAG_MASK_INF:
        op::diag_mask_inf(stream(), dst);
        break;
    case GGML_OP_SOFT_MAX:
        op::soft_max(stream(), dst);
        break;
    case GGML_OP_SOFT_MAX_BACK:
        op::soft_max_back(stream(), dst);
        break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        op::rope(stream(), dst, (dst->op == GGML_OP_ROPE) ? true : false);
        break;
    case GGML_OP_IM2COL:
        op::im2col(stream(), dst);
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
    case GGML_OP_POOL_2D:
        op::pool2d(stream(), dst);
        break;
    case GGML_OP_SUM:
        op::sum(pool(), stream(), dst);
        break;
    case GGML_OP_SUM_ROWS:
        op::sum_rows(stream(), dst);
        break;
    case GGML_OP_MEAN:
        op::mean(stream(), dst);
        break;
    case GGML_OP_SSM_CONV:
        op::ssm_conv(stream(), dst);
        break;
    case GGML_OP_SSM_SCAN:
        op::ssm_scan(stream(), dst);
        break;
    case GGML_OP_ARGSORT:
        op::argsort(stream(), dst);
        break;
    case GGML_OP_FLASH_ATTN_EXT:
        op::flash_attn_ext(device, pool(), stream(), dst);
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS:
        op::cross_entropy_loss(pool(), stream(), dst);
        break;
    case GGML_OP_RWKV_WKV6:
        op::rwkv_wkv6(stream(), dst);
        break;
    case GGML_OP_GATED_LINEAR_ATTN:
        op::gated_linear_attn(stream(), dst);
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

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

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
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }
}

