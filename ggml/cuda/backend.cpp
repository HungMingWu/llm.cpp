module;
#include <assert.h>
#include <bit>
#include <memory>
#include <span>
#include "block.h"
#include "common.h"
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

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool* pool = nullptr;
    T* ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool& pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool& pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T* alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T*)pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T* alloc(ggml_cuda_pool& pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T* get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc&) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc&&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc&) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc&&) = delete;
};

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
    const int64_t stride00 = ne00 / ggml_blck_size(src0->type);

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA &&
        cc < GGML_CUDA_CC_OFFSET_AMD && src1_ncols == ne11;
    const mmq_args args = { src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stride00, src1_padded_row_size, src1_ncols, ne11, nrows_dst, use_stream_k };
#if 0
    switch (src0->type) {
    case GGML_TYPE_Q4_0:
        mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
        break;
    case GGML_TYPE_Q4_1:
        mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
        break;
    case GGML_TYPE_Q5_0:
        mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
        break;
    case GGML_TYPE_Q5_1:
        mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
        break;
    case GGML_TYPE_Q8_0:
        mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
        break;
    case GGML_TYPE_Q2_K:
        mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
        break;
    case GGML_TYPE_Q3_K:
        mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
        break;
    case GGML_TYPE_Q4_K:
        mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
        break;
    case GGML_TYPE_Q5_K:
        mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
        break;
    case GGML_TYPE_Q6_K:
        mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ2_XXS:
        mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ2_XS:
        mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ2_S:
        mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ3_XXS:
        mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ3_S:
        mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ1_S:
        mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ4_XS:
        mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
        break;
    case GGML_TYPE_IQ4_NL:
        mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
#endif
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

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
    case GGML_TYPE_Q4_0:
        mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
#if 0
    case GGML_TYPE_Q4_1:
        mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q5_0:
        mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q5_1:
        mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q8_0:
        mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q2_K:
        mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q3_K:
        mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q4_K:
        mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q5_K:
        mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_Q6_K:
        mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ2_XXS:
        mul_mat_vec_iq2_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ2_XS:
        mul_mat_vec_iq2_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ2_S:
        mul_mat_vec_iq2_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ3_XXS:
        mul_mat_vec_iq3_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ1_S:
        mul_mat_vec_iq1_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ1_M:
        mul_mat_vec_iq1_m_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ4_NL:
        mul_mat_vec_iq4_nl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ4_XS:
        mul_mat_vec_iq4_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
    case GGML_TYPE_IQ3_S:
        mul_mat_vec_iq3_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
        break;
#endif
    default:
        GGML_ABORT("fatal error");
        break;
    }
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
        //ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
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
            // TODO
#if 0
            size_t src_1_ddq_size = nrows1 * src1_padded_col_size * q8_1_ts / q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc) * sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(pool(id), src_1_ddq_size);

            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12 * ne13, src1_padded_col_size, src0->type, stream);
                CUDA_CHECK(cudaGetLastError());
            }
#endif
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
                //TODO
                if (0) { //quantize_src1 == quantize_mmq_q8_1_cuda) {
                    //src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                }
                else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size * q8_1_ts / q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                char* src0_dd_i = dev[id].src0_dd + (i0 / i02_divisor) * (ne01 * ne00 * src0_ts) / src0_bs;
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
                    // TODO
#if 0
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0 + src1_ncols, stream));
#endif
                }
                else {
                    GGML_ABORT("fatal error");
                }
                // TODO
#if 0
                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02 / i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }
#endif
                // do the computation
                op(*this, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // TODO
#if 0
                // copy dst to host or other device if necessary
                if (!dst_on_device) {
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
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
#endif
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
#if 0
    case GGML_OP_REPEAT:
        ggml_cuda_op_repeat(ctx, dst);
        break;
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
#if 0
    case GGML_OP_CPY:
        ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
        break;

    case GGML_OP_ADD:
    case GGML_OP_ADD1: // TODO: more efficient implementation
        ggml_cuda_op_add(ctx, dst);
        break;
    case GGML_OP_SUB:
        ggml_cuda_op_sub(ctx, dst);
        break;
    case GGML_OP_ACC:
        ggml_cuda_op_acc(ctx, dst);
        break;
    case GGML_OP_MUL:
        ggml_cuda_op_mul(ctx, dst);
        break;
    case GGML_OP_DIV:
        ggml_cuda_op_div(ctx, dst);
        break;
#endif
    case GGML_OP_UNARY:
        op::unary(stream(), dst);
        break;
#if 0
    case GGML_OP_NORM:
        ggml_cuda_op_norm(ctx, dst);
        break;
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
    case GGML_OP_RMS_NORM:
        ggml_cuda_op_rms_norm(ctx, dst);
        break;
#endif
    case GGML_OP_MUL_MAT:
        if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
            GGML_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
            return false;
        }
        else {
            mul_mat(dst);
        }
        break;
#if 0
    case GGML_OP_MUL_MAT_ID:
        ggml_cuda_mul_mat_id(ctx, dst);
        break;
    case GGML_OP_OUT_PROD:
        ggml_cuda_out_prod(ctx, dst);
        break;
    case GGML_OP_SCALE:
        ggml_cuda_op_scale(ctx, dst);
        break;
    case GGML_OP_SQR:
        ggml_cuda_op_sqr(ctx, dst);
        break;
    case GGML_OP_SQRT:
        ggml_cuda_op_sqrt(ctx, dst);
        break;
    case GGML_OP_SIN:
        ggml_cuda_op_sin(ctx, dst);
        break;
    case GGML_OP_COS:
        ggml_cuda_op_cos(ctx, dst);
        break;
    case GGML_OP_CLAMP:
        ggml_cuda_op_clamp(ctx, dst);
        break;
#endif
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        break;
#if 0
    case GGML_OP_DIAG_MASK_INF:
        ggml_cuda_op_diag_mask_inf(ctx, dst);
        break;
    case GGML_OP_SOFT_MAX:
        ggml_cuda_op_soft_max(ctx, dst);
        break;
    case GGML_OP_ROPE:
        ggml_cuda_op_rope(ctx, dst);
        break;
#endif
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
    case GGML_OP_RWKV_WKV6:
        ggml_cuda_op_rwkv_wkv6(ctx, dst);
        break;
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

