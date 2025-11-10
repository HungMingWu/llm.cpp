#include <float.h>
#include <bit>
#include "cuda_func.h"
#include "common.cuh"
#include "table.h"
#include "dequantize.cuh"
#include "cpy-utils.cuh"

#define GGML_ASSERT(...)

static constexpr size_t CUDA_CPY_BLOCK_SIZE = 64;
static constexpr size_t CUDA_CPY_TILE_DIM_2D = 32; // 2D tile dimension for transposed blocks
static constexpr size_t CUDA_CPY_BLOCK_NM = 8;     // block size of 3rd dimension if available
static constexpr size_t CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows

template <typename T>
concept block_v =
    std::is_same_v<T, block_q8_0> ||
    std::is_same_v<T, block_q5_1> ||
    std::is_same_v<T, block_q4_0> ||
    std::is_same_v<T, block_q4_1> ||
    std::is_same_v<T, block_iq4_nl> ||
    std::is_same_v<T, block_q5_0>;

template <typename src_t, typename dst_t>
concept simplest_case_v = !block_v<src_t> && !block_v<dst_t>;

template <typename src_t, typename dst_t>
static __device__ void copy_block(const src_t* xi, dst_t* dsti) {
    *dsti = ggml_cuda_cast<dst_t>(*xi);
}

template <block_v src_t>
static __device__ void copy_block(const src_t* xi, float* dsti) {
    if constexpr (std::is_same_v<src_t, block_q8_0>) {
        const float d = __half2float(std::bit_cast<half>(xi->d));
#pragma unroll
        for (int j = 0; j < block_q8_0::block_size; j++) {
            dsti[j] = xi->qs[j] * d;
        }
    }
    else {
#pragma unroll
        for (int j = 0; j < src_t::block_size / 2; j++) {
            float2 dq;
            dequantize(xi, 0, j, dq);
            *(dsti + j) = dq.x;
            *(dsti + j + src_t::block_size / 2) = dq.y;
        }
    }
};

static __device__ void copy_block(const float* xi, block_q4_0* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q4_1* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q5_0* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q5_1* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_q8_0* dsti) {
    quantize_block(xi, dsti);
}

static __device__ void copy_block(const float* xi, block_iq4_nl* dsti) {
    quantize_block(xi, dsti);
}

template <typename src_t, typename dst_t>
static __global__ void cpy(dup_context ctx) {
    const int64_t i = [=] {
        int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if constexpr (block_v<src_t>) i *= src_t::block_size;
        else if constexpr (block_v<dst_t>) i *= dst_t::block_size;
        return i;
    }();
    if (i >= ctx.ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i / (ctx.ne00 * ctx.ne01 * ctx.ne02);
    const int64_t i02 = (i - i03 * ctx.ne00 * ctx.ne01 * ctx.ne02) / (ctx.ne00 * ctx.ne01);
    const int64_t i01 = (i - i03 * ctx.ne00 * ctx.ne01 * ctx.ne02 - i02 * ctx.ne01 * ctx.ne00) / ctx.ne00;
    const int64_t i00 = i - i03 * ctx.ne00 * ctx.ne01 * ctx.ne02 - i02 * ctx.ne01 * ctx.ne00 - i01 * ctx.ne00;
    const int64_t i13 = i / (ctx.ne10 * ctx.ne11 * ctx.ne12);
    const int64_t i12 = (i - i13  * ctx.ne10 * ctx.ne11 *ctx. ne12) / (ctx.ne10 * ctx.ne11);
    const int64_t i11 = (i - i13 * ctx.ne10 * ctx.ne11 * ctx.ne12 - i12 * ctx.ne10 * ctx.ne11) / ctx.ne10;
    const int64_t i10 = i - i13 * ctx.ne10 * ctx.ne11 * ctx.ne12 - i12 * ctx.ne10 * ctx.ne11 - i11 * ctx.ne10;;
    
    const int64_t x_offset = [=] {
        if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
            return (i00 / src_t::block_size) * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
        else {
            return i00 * ctx.nb00 + i01 * ctx.nb01 + i02 * ctx.nb02 + i03 * ctx.nb03;
        }
    }();
    const int64_t dst_offset = [=] {
        if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
            return (i10 / dst_t::block_size) * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
        } else {
            return i10 * ctx.nb10 + i11 * ctx.nb11 + i12 * ctx.nb12 + i13 * ctx.nb13;
        }
    }();
    copy_block(
        (const src_t*)((const char*)ctx.src_d + x_offset), 
        (dst_t*)((char *)ctx.dst_d + dst_offset));
}

template <typename T>
static __global__ void cpy_flt_transpose(const char* cx, char* cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
    const int nb12, const int nb13) {

    const T* src = reinterpret_cast<const T*>(cx);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    const int x = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int y = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset
    const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    __shared__ float tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM; ++i) {

        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM + i;
        if (imat >= nmat)
            break;

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (x < ne01 && y + j < ne00) {
                const int row = threadIdx.y + j;
                const int col = threadIdx.x * sizeof(float) / sizeof(T);
                T* tile2 = reinterpret_cast<T*>(tile[row]);
                tile2[col] = src[imat * n + (y + j) * ne01 + x];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (ty + j < ne01 && tx < ne00) {
                const int col = (threadIdx.y + j) * sizeof(float) / sizeof(T);
                const T* tile2 = reinterpret_cast<const T*>(tile[threadIdx.x]);
                dst[imat * n + (ty + j) * ne00 + tx] = tile2[col];
            }
        }
    }
}

template<typename src_t, typename dst_t>
static __global__ void cpy_flt_contiguous(const char* cx, char* cdst, const int64_t ne) {
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const src_t* x = (const src_t*)cx;
    dst_t* dst = (dst_t*)cdst;

    dst[i] = ggml_cuda_cast<dst_t>(x[i]);
}

template<typename src_t, typename dst_t>
static void ggml_cpy_flt_contiguous_cuda(
    const char* cx, char* cdst, const int64_t ne,
    cudaStream_t stream) {

    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_flt_contiguous<src_t, dst_t> << <num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream >> >
        (cx, cdst, ne);
}

template <typename src_t, typename dst_t, bool transposed = false>
static void ggml_cpy_cuda(const dup_context& ctx, cudaStream_t stream) {
    // copy context by value
    if constexpr (transposed) {
        GGML_ASSERT(ctx.ne == ctx.ne00 * ctx.ne01 * ctx.ne02);  // ne[3] is 1 assumed
        int ne00n, ne01n, ne02n;
        if (ctx.nb00 <= ctx.nb02) { // most likely safe to handle nb00 = nb02 case here
            ne00n = ctx.ne00;
            ne01n = ctx.ne01;
            ne02n = ctx.ne02;
        }
        else if (ctx.nb00 > ctx.nb02) {
            ne00n = ctx.ne00;
            ne01n = ctx.ne01 * ctx.ne02;
            ne02n = 1;
        }

        dim3 dimGrid((ne01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
            (ne00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
            (ctx.ne / (ne01n * ne00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM);
        dim3 dimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
        cpy_flt_transpose<dst_t> << <dimGrid, dimBlock, 0, stream >> >
            (static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, ne00n, ne01n, ne02n,
                ctx.nb00, ctx.nb01, ctx.nb02, ctx.nb03,
                ctx.ne10, ctx.ne11, ctx.ne12, ctx.nb10, ctx.nb11, ctx.nb12, ctx.nb13);
    }
    else {
        if constexpr (simplest_case_v<src_t, dst_t>) {
            const int num_blocks = (ctx.ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
            cpy<src_t, dst_t> << <num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream >> > (ctx);
        }
        else if constexpr (std::is_same_v<src_t, float> && block_v<dst_t>) {
            GGML_ASSERT(ctx.ne % dst_t::block_size == 0);
            const int num_blocks = ctx.ne / dst_t::block_size;
            cpy<src_t, dst_t> << <num_blocks, 1, 0, stream >> > (ctx);
        }
        else if constexpr (block_v<src_t> && std::is_same_v<dst_t, float>) {
            const int num_blocks = ctx.ne;
            cpy<src_t, dst_t> << <num_blocks, 1, 0, stream >> > (ctx);
        }
    }
}

void dup_cuda(const dup_context& ctx, cudaStream_t stream)
{
    if (ctx.src_type == ctx.dst_type && ctx.contiguous) {
        GGML_ASSERT(ctx.src_length == ctx.dst_length);
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        }
        else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            CUDA_CHECK(cudaMemcpyAsync(ctx.dst_d, ctx.src_d, ctx.src_length, cudaMemcpyDeviceToDevice, stream));
        }
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_F32) {
        if (ctx.can_be_transposed) {
            ggml_cpy_cuda<float, float, true>(ctx, stream);
        }
        else {
            ggml_cpy_cuda<float, float, false>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_BF16) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<float, nv_bfloat16>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<float, nv_bfloat16>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_F16) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<float, half>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<float, half>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_Q8_0) {
        ggml_cpy_cuda<float, block_q8_0>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_Q8_0 && ctx.dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q8_0, float>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_Q4_0) {
        ggml_cpy_cuda<float, block_q4_0>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_Q4_0 && ctx.dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q4_0, float>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_Q4_1) {
        ggml_cpy_cuda<float, block_q4_1>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_Q4_1 && ctx.dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q4_1, float>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_Q5_0) {
        ggml_cpy_cuda<float, block_q5_0>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_Q5_0 && ctx.dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q5_0, float>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_cuda<float, block_iq4_nl>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_Q5_1) {
        ggml_cpy_cuda<float, block_q5_1>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_Q5_1 && ctx.dst_type == GGML_TYPE_F32) {
        ggml_cpy_cuda<block_q5_1, float>(ctx, stream);
    }
    else if (ctx.src_type == GGML_TYPE_F16 && ctx.dst_type == GGML_TYPE_F16) {
        if (ctx.can_be_transposed) {
            ggml_cpy_cuda<half, half, true>(ctx, stream);
        }
        else {
            ggml_cpy_cuda<half, half, false>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_F16 && ctx.dst_type == GGML_TYPE_BF16) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<half, nv_bfloat16>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<half, nv_bfloat16>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_F16 && ctx.dst_type == GGML_TYPE_F32) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<half, float>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<half, float>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_BF16 && ctx.dst_type == GGML_TYPE_BF16) {
        if (ctx.can_be_transposed) {
            ggml_cpy_cuda<nv_bfloat16, nv_bfloat16, true>(ctx, stream);
        }
        else {
            ggml_cpy_cuda<nv_bfloat16, nv_bfloat16, false>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_BF16 && ctx.dst_type == GGML_TYPE_F16) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<nv_bfloat16, half>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<nv_bfloat16, half>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_BF16 && ctx.dst_type == GGML_TYPE_F32) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<nv_bfloat16, float>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<nv_bfloat16, float>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_F32 && ctx.dst_type == GGML_TYPE_I32) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<float, int32_t>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<float, int32_t>(ctx, stream);
        }
    }
    else if (ctx.src_type == GGML_TYPE_I32 && ctx.dst_type == GGML_TYPE_F32) {
        if (ctx.contiguous) {
            ggml_cpy_flt_contiguous_cuda<int32_t, float>(static_cast<const char*>(ctx.src_d), static_cast<char*>(ctx.dst_d), ctx.ne, stream);
        }
        else {
            ggml_cpy_cuda<int32_t, float>(ctx, stream);
        }
    }
    else {
        assert(false);
        //GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
          //  ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}
