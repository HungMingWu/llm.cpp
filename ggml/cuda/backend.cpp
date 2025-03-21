module;
#include <assert.h>
#include <bit>
#include <memory>
#include <span>
#include "block.h"
#include "common.h"
#include "cuda_pool.h"
#include "cu/convert.cuh"
#include "cu/cuda_func.h"

#define GGML_ABORT(...)
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_LOG_DEBUG(...)
#define GGML_LOG_ERROR(...)
#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
// maximum number of src0 rows with which to use mul_mat_vec over cuBLAS if FP16 tensor cores are available
#define MMV_MAX_ROWS 512
#define MUL_MAT_SRC1_COL_STRIDE 128
#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor co
#define GGML_CUDA_CC_IS_CDNA(cc)  (cc >= GGML_CUDA_CC_CDNA && cc < GGML_CUDA_CC_RDNA1)

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

static constexpr bool int8_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && cc >= GGML_CUDA_CC_TURING;
}

static constexpr bool fp16_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && cc >= GGML_CUDA_CC_VOLTA;
}

module ggml;
import :cuda.config;
import :cuda.backend;
import :cuda.op;

static bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
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

static void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* /*src1_ddf_i*/,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
    const int64_t row_diff = row_high - row_low;
    int id = ggml_cuda_get_device();

    mat_q_context context{
        .id = id,
        .pool = &ctx.pool(id),
        .src0_dd_i = src0_dd_i,
        .src0_type = src0->type,
        .src1_ddq_i = src1_ddq_i,
        .src1_ncols = src1_ncols,
        .dst_dd_i = dst_dd_i,
        // the main device has a larger memory buffer to hold the results from all GPUs
        // nrows_dst == nrows of the matrix that the kernel writes into
        .nrows_dst = (id == ctx.device) ? dst->ne[0] : row_diff,
        .ne00 = src0->ne[0],
        .ne11 = src1->ne[1],
        .row_diff = row_diff,
        .stride00 = src0->ne[0] / (int64_t)ggml_blck_size(src0->type),
        .src1_padded_row_size = src1_padded_row_size
    };
    mul_mat_q_cuda(&context, stream);
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream)
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

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (compute_capability >= GGML_CUDA_CC_VOLTA && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff * ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
        }

        const half* src0_ptr = src0->type == GGML_TYPE_F16 ? (const half*)src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols * ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half* src1_ptr = src1->type == GGML_TYPE_F16 ? (const half*)src1_ddf_i : src1_as_f16.get();
        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff * src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        cublasComputeType_t cu_compute_type = CUBLAS_COMPUTE_16F;
        if (ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_CDNA) {
            cu_compute_type = CUBLAS_COMPUTE_32F;
        }

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                row_diff, src1_ncols, ne10,
                &alpha_f16, src0_ptr, CUDA_R_16F, ne00,
                src1_ptr, CUDA_R_16F, ne10,
                &beta_f16, dst_f16.get(), CUDA_R_16F, ldc,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff * src1_ncols, stream);

    }
    else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff * ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff * ne00, stream);
        }

        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols * ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols * ne10, stream);
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

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(src1_ncols == 1);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row = ne00;
    const int64_t nchannels_x = 1;
    const int64_t nchannels_y = 1;
    const int64_t channel_stride_x = 0;
    const int64_t channel_stride_y = 0;
    const int64_t channel_stride_dst = 0;

    mul_mat_vec_cuda((const half*)src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stride_row,
        nchannels_x, nchannels_y, channel_stride_x, channel_stride_y, channel_stride_dst, prec, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda& ctx,
    ggml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
    const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
    const int64_t row_diff = row_high - row_low;
    int id = ggml_cuda_get_device();

    mat_vec_context context{
        .src0_type = src0->type,
        .vx = src0_dd_i,
        .vy = src1_ddq_i,
        .dst = dst_dd_i,
        .ncols_x = src0->ne[0],
        .nrows_x = row_diff,
        .ncols_y = src1_ncols,
        .nrows_y = src1_padded_row_size,
        // the main device has a larger memory buffer to hold the results from all GPUs
        // nrows_dst == nrows of the matrix that the kernel writes into
        .nrows_dst = id == ctx.device ? dst->ne[0] : row_diff
    };
    mul_mat_vec_q_cuda(&context, stream);
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
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
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

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = dst->nelements();

    cudaStream_t main_stream = ctx.stream();

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    void* src0_ddq = src0->data;
    half* src0_f16 = (half*)src0_ddq;
    float* src1_ddf = (float*)src1->data;
    float* dst_ddf = (float*)dst->data;

    // convert src1 to fp16
    ggml_cuda_pool_alloc<half> src1_f16_alloc(ctx.pool());
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        const int64_t ne_src1 = src1->nelements();
        src1_f16_alloc.alloc(ne_src1);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda(src1_ddf, src1_f16_alloc.get(), ne_src1, main_stream);
    }
    half* src1_f16 = src1->type == GGML_TYPE_F16 ? (half*)src1_ddf : src1_f16_alloc.get();

    ggml_cuda_pool_alloc<half> dst_f16(ctx.pool());
    char* dst_t;

    cublasComputeType_t cu_compute_type = CUBLAS_COMPUTE_16F;
    cudaDataType_t      cu_data_type = CUDA_R_16F;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const half  alpha_f16 = 1.0f;
    const half  beta_f16 = 0.0f;

    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    const void* alpha = &alpha_f16;
    const void* beta = &beta_f16;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char*)dst_f16.alloc(ne_dst);

        nbd2 /= sizeof(float) / sizeof(half);
        nbd3 /= sizeof(float) / sizeof(half);
    }
    else {
        dst_t = (char*)dst_ddf;

        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;

        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    if (GGML_CUDA_CC_IS_CDNA(ggml_cuda_info().devices[ctx.device].cc)) {
        cu_compute_type = CUBLAS_COMPUTE_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if 0
    // use cublasGemmEx
    {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                int i03 = i13 / r3;
                int i02 = i12 / r2;

                CUBLAS_CHECK(
                    cublasGemmEx(g_cublas_handles[g_main_device], CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        alpha, (const char*)src0_as_f16 + i02 * src0->nb[2] + i03 * src0->nb[3], CUDA_R_16F, nb01 / sizeof(half),
                        (const char*)src1_as_f16 + i12 * src1->nb[2] / 2 + i13 * src1->nb[3] / 2, CUDA_R_16F, nb11 / sizeof(float),
                        beta, (char*)dst_t + i12 * nbd2 + i13 * nbd3, cu_data_type, ne01,
                        cu_compute_type,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
#else
#ifdef GGML_USE_MUSA
    GGML_ASSERT(false);
#else // !GGML_USE_MUSA
    if (r2 == 1 && r3 == 1 && ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1)) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
            cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const char*)src0_f16, CUDA_R_16F, nb01 / nb00, nb02 / nb00,  // strideA
                (const char*)src1_f16, CUDA_R_16F, nb11 / nb10, nb12 / nb10,  // strideB
                beta, (char*)dst_t, cu_data_type, ne01, nb2 / nb0,   // strideC
                ne12 * ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else {
        // use cublasGemmBatchedEx
        const int ne23 = ne12 * ne13;

        ggml_cuda_pool_alloc<const void*> ptrs_src(ctx.pool(), 2 * ne23);
        ggml_cuda_pool_alloc<      void*> ptrs_dst(ctx.pool(), 1 * ne23);
        k_compute_batched_ptrs_cuda(
            src0_f16, src1_f16, dst_t,
            ptrs_src.get(), ptrs_dst.get(),
            ne12, ne13,
            ne23,
            nb02, nb03,
            src1->type == GGML_TYPE_F16 ? nb12 : nb12 / 2,
            src1->type == GGML_TYPE_F16 ? nb13 : nb13 / 2,
            nbd2, nbd3,
            r2, r3, main_stream);
        CUBLAS_CHECK(
            cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void**)(ptrs_src.get() + 0 * ne23), CUDA_R_16F, nb01 / nb00,
                (const void**)(ptrs_src.get() + 1 * ne23), CUDA_R_16F, nb11 / nb10,
                beta, (void**)(ptrs_dst.get() + 0 * ne23), cu_data_type, ne01,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
#endif // GGML_USE_MUSA
#endif

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_ddf, ne_dst, main_stream);
    }
}


void ggml_backend_cuda::mul_mat(ggml_tensor* dst)
{
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->get_type());

    bool use_mul_mat_vec = src0->type == GGML_TYPE_F16
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % 2 == 0 && src1->ne[1] == 1;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16 = false;
    bool any_gpus_without_fp16_mma = false;

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
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_available(cc);
            any_gpus_without_fp16_mma = any_gpus_without_fp16_mma || !fp16_mma_available(cc);
        }
    }
    else {
        const int cc = ggml_cuda_info().devices[device].cc;
        use_mul_mat_q = use_mul_mat_q && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_available(cc);
        any_gpus_without_fp16_mma = any_gpus_without_fp16_mma || !fp16_mma_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (!split && use_mul_mat_vec && dst->ne[3] == 1 && (src0->ne[1] < MMV_MAX_ROWS || any_gpus_without_fp16_mma)) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        op::mul_mat_vec(this->stream(), dst);
    }
    else if (!split && src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
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

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
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

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->get_type());
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
                quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12 * ne13, src1_padded_col_size, src0->type, stream);
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
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
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

void ggml_backend_cuda::set_tensor_async(ggml_tensor* tensor, const void* data, size_t offset, size_t size)
{
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->get_type() == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, stream()));
}

void ggml_backend_cuda::get_tensor_async(const ggml_tensor* tensor, void* data, size_t offset, size_t size)
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
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
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

void ggml_backend_cuda::evaluate_and_capture_cuda_graph(ggml_cgraph* cgraph,
    [[maybe_unused]] std::vector<void*>& ggml_cuda_cpy_fn_ptrs, bool& graph_evaluated_or_captured, bool& use_cuda_graph,
    bool& cuda_graph_update_required) {

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            for (auto& node : cgraph->nodes) {
                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

#ifndef NDEBUG
                assert(node->buffer->get_type() == ggml_backend_cuda_buffer_type(device));
                for (auto& src : node->src) {
                    assert(src->buffer);
                    assert(src->buffer->get_type() == ggml_backend_cuda_buffer_type(device) ||
                        ggml_backend_buft_is_cuda_split(src->buffer->get_type()));
                }
#endif

                bool ok = compute_forward(node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured
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
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

enum ggml_status ggml_backend_cuda::graph_compute(ggml_cgraph* cgraph)
{
    ggml_cuda_set_device(device);

    // vector of pointers to CUDA cpy kernels, which are required to identify
    // kernel parameters which need updated in the graph for each token
    std::vector<void*> ggml_cuda_cpy_fn_ptrs;

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;

    if (cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }

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
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) { // Start CUDA graph capture
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

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
    if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->get_type())) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], device);
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
#if 0
    case GGML_OP_REPEAT_BACK:
        ggml_cuda_op_repeat_back(ctx, dst);
        break;
#endif
    case GGML_OP_GET_ROWS:
        op::get_rows(stream(), dst);
        break;
    case GGML_OP_GET_ROWS_BACK:
        op::get_rows_back(stream(), dst);
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
#if 0
    case GGML_OP_SUB:
        ggml_cuda_op_sub(ctx, dst);
        break;
    case GGML_OP_ACC:
        ggml_cuda_op_acc(ctx, dst);
        break;
#endif
    case GGML_OP_MUL:
        op::mul(stream(), dst);
        break;
    case GGML_OP_DIV:
        op::div(stream(), dst);
        break;
    case GGML_OP_UNARY:
        op::unary(stream(), dst);
        break;
    case GGML_OP_NORM:
        op::norm(stream(), dst);
        break;
#if 0
    case GGML_OP_GROUP_NORM:
        ggml_cuda_op_group_norm(ctx, dst);
        break;
    case GGML_OP_CONCAT:
        ggml_cuda_op_concat(ctx, dst);
        break;
    case GGML_OP_UPSCALE:
        ggml_cuda_op_upscale(ctx, dst);
        break;
    case GGML_OP_PAD:
        ggml_cuda_op_pad(ctx, dst);
        break;
#endif
    case GGML_OP_ARANGE:
        op::arange(stream(), dst);
        break;
#if 0
    case GGML_OP_TIMESTEP_EMBEDDING:
        ggml_cuda_op_timestep_embedding(ctx, dst);
        break;
    case GGML_OP_LEAKY_RELU:
        ggml_cuda_op_leaky_relu(ctx, dst);
        break;
#endif
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
    case GGML_OP_CONV_TRANSPOSE_1D:
        op::conv_transpose_1d(stream(), dst);
        break;
    case GGML_OP_POOL_2D:
        op::pool2d(stream(), dst);
        break;
#if 0
    case GGML_OP_SUM:
        ggml_cuda_op_sum(ctx, dst);
        break;
    case GGML_OP_SUM_ROWS:
        ggml_cuda_op_sum_rows(ctx, dst);
        break;
    case GGML_OP_ARGSORT:
        ggml_cuda_op_argsort(ctx, dst);
        break;
    case GGML_OP_FLASH_ATTN_EXT:
        ggml_cuda_flash_attn_ext(ctx, dst);
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_cuda_cross_entropy_loss(ctx, dst);
        break;
#endif
    case GGML_OP_RWKV_WKV6:
        op::rwkv_wkv6(stream(), dst);
        break;
    case GGML_OP_GATED_LINEAR_ATTN:
        op::gated_linear_attn(stream(), dst);
        break;
#if 0
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        ggml_cuda_cross_entropy_loss_back(ctx, dst);
        break;
    case GGML_OP_OPT_STEP_ADAMW:
        ggml_cuda_opt_step_adamw(ctx, dst);
        break;
#endif
    default:
        return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

ggml_backend_cuda::~ggml_backend_cuda()
{
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

