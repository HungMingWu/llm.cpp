#include "internal_ds.h"
#include "table.h"
#include "common.cuh"
#include "convert.cuh"
#include "block.h"
#include <bit>

#define GGML_UNUSED(x)  (void)(x)
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
static constexpr size_t CUDA_Q8_0_NE_ALIGN = 2048;

template <typename src_t, typename dst_t>
static __global__ void convert_unary(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    const src_t* x = (src_t*)vx;
    y[i] = x[i];
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    convert_unary<src_t> << <num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream >> > (vx, y, k);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void* __restrict__ vx, dst_t* __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_iq1_m* x = (const block_iq1_m*)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*)x[i].scales;
    iq1m_scale_t scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = __half2float(std::bit_cast<half>(scale)) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[i].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t* q = (const int8_t*)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4 * ib + il] | (((x[i].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m << <nb, 32, 0, stream >> > (vx, y);
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type)
{
    switch (type) {
#if 0
    case GGML_TYPE_Q4_0:
        return dequantize_row_q4_0_cuda;
    case GGML_TYPE_Q4_1:
        return dequantize_row_q4_1_cuda;
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
    case GGML_TYPE_Q8_0:
        return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
    case GGML_TYPE_Q2_K:
        return dequantize_row_q2_K_cuda;
    case GGML_TYPE_Q3_K:
        return dequantize_row_q3_K_cuda;
    case GGML_TYPE_Q4_K:
        return dequantize_row_q4_K_cuda;
    case GGML_TYPE_Q5_K:
        return dequantize_row_q5_K_cuda;
    case GGML_TYPE_Q6_K:
        return dequantize_row_q6_K_cuda;
    case GGML_TYPE_IQ2_XXS:
        return dequantize_row_iq2_xxs_cuda;
    case GGML_TYPE_IQ2_XS:
        return dequantize_row_iq2_xs_cuda;
    case GGML_TYPE_IQ2_S:
        return dequantize_row_iq2_s_cuda;
    case GGML_TYPE_IQ3_XXS:
        return dequantize_row_iq3_xxs_cuda;
    case GGML_TYPE_IQ1_S:
        return dequantize_row_iq1_s_cuda;
#endif
    case GGML_TYPE_IQ1_M:
        return dequantize_row_iq1_m_cuda;
#if 0
    case GGML_TYPE_IQ4_NL:
        return dequantize_row_iq4_nl_cuda;
    case GGML_TYPE_IQ4_XS:
        return dequantize_row_iq4_xs_cuda;
    case GGML_TYPE_IQ3_S:
        return dequantize_row_iq3_s_cuda;
#endif
    case GGML_TYPE_F16:
        return convert_unary_cuda<half>;
    case GGML_TYPE_BF16:
        return convert_unary_cuda<nv_bfloat16>;
    default:
        return nullptr;
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void* __restrict__ vx, half* __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN / sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN * blockIdx.x;
    const int* x0 = ((int*)vx) + blockIdx.x * nint;
    half2* y2 = (half2*)(y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0 * sizeof(block_q8_0) / block_q8_0::block_size + sizeof(int) * (ix0 + threadIdx.x) >= k * sizeof(block_q8_0) / block_q8_0::block_size) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2 * WARP_SIZE) {
        if (need_check && i0 + iy + 2 * threadIdx.x >= k) {
            return;
        }

        const half* b0 = ((const half*)vals) + (sizeof(block_q8_0) / sizeof(half)) * ((iy + 2 * threadIdx.x) / block_q8_0::block_size);
        const half    d = *b0;
        const char2  qs = ((const char2*)(b0 + 1))[threadIdx.x % (block_q8_0::block_size / 2)];

        y2[iy / 2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(y);
    GGML_UNUSED(k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

static void dequantize_block_q8_0_f16_cuda(const void* __restrict__ vx, half* __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check> << <num_blocks, WARP_SIZE, 0, stream >> > (vx, y, k);
    }
    else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check> << <num_blocks, WARP_SIZE, 0, stream >> > (vx, y, k);
    }
}

using dequantize_kernel_t = void (*)(const void* vx, const int64_t ib, const int iqs, dfloat2& v);

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + 2 * CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block<qk, qr, dequantize_kernel> << <num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream >> > (vx, y, k);
}

static __device__ __forceinline__ void dequantize_q8_0(const void* vx, const int64_t ib, const int iqs, dfloat2& v) {
    const block_q8_0* x = (const block_q8_0*)vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)2 * (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int64_t ib = i / qk; // block index
    const int64_t iqs = (i % qk) / qr; // quant index
    const int64_t iybs = i - i % qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void* __restrict__ vx, dst_t* __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ir = tid % 8;
    const int64_t ib = 8 * i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t* y = yy + 256 * i + 32 * ir + 4 * il;

    const block_q4_0* x = (const block_q4_0*)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8 * d;

    const uint8_t* q = x->qs + 4 * il;

    for (int l = 0; l < 4; ++l) {
        y[l + 0] = d * (q[l] & 0xF) + dm;
        y[l + 16] = d * (q[l] >> 4) + dm;
    }
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0 << <nb, 32, 0, stream >> > (vx, y, nb32);
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type)
{
    switch (type) {
    case GGML_TYPE_Q4_0:
        return dequantize_row_q4_0_cuda;
#if 0
    case GGML_TYPE_Q4_1:
        return dequantize_row_q4_1_cuda;
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
#endif
    case GGML_TYPE_Q8_0:
        if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= GGML_CUDA_CC_PASCAL) {
            return dequantize_block_q8_0_f16_cuda;
        }
        return dequantize_block_cuda<block_q8_0::block_size, QR8_0, dequantize_q8_0>;
#if 0
    case GGML_TYPE_Q2_K:
        return dequantize_row_q2_K_cuda;
    case GGML_TYPE_Q3_K:
        return dequantize_row_q3_K_cuda;
    case GGML_TYPE_Q4_K:
        return dequantize_row_q4_K_cuda;
    case GGML_TYPE_Q5_K:
        return dequantize_row_q5_K_cuda;
    case GGML_TYPE_Q6_K:
        return dequantize_row_q6_K_cuda;
    case GGML_TYPE_IQ2_XXS:
        return dequantize_row_iq2_xxs_cuda;
    case GGML_TYPE_IQ2_XS:
        return dequantize_row_iq2_xs_cuda;
    case GGML_TYPE_IQ2_S:
        return dequantize_row_iq2_s_cuda;
    case GGML_TYPE_IQ3_XXS:
        return dequantize_row_iq3_xxs_cuda;
    case GGML_TYPE_IQ1_S:
        return dequantize_row_iq1_s_cuda;
    case GGML_TYPE_IQ1_M:
        return dequantize_row_iq1_m_cuda;
    case GGML_TYPE_IQ4_NL:
        return dequantize_row_iq4_nl_cuda;
    case GGML_TYPE_IQ4_XS:
        return dequantize_row_iq4_xs_cuda;
    case GGML_TYPE_IQ3_S:
        return dequantize_row_iq3_s_cuda;
    case GGML_TYPE_F32:
        return convert_unary_cuda<float>;
#endif
    default:
        return nullptr;
    }
}