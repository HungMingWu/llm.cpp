#include "internal_ds.h"
#include "table.h"
#include "common.cuh"
#include "convert.cuh"
#include "block.h"
#include <bit>
#include <assert.h>
#include <type_traits>

#define GGML_UNUSED(x)  (void)(x)
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
static constexpr size_t CUDA_Q8_0_NE_ALIGN = 2048;

template <typename src_t, typename dst_t>
static __global__ void convert_unary(
    const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = blockIdx.y;
    const int64_t i02 = blockIdx.z % ne02;
    const int64_t i03 = blockIdx.z / ne02;
    const src_t* x = (const src_t*)vx;
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01 + i00;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00 + i00;
    y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void* vx, dst_t* y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, ne01, ne02 * ne03);
    convert_unary<src_t> << <num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream >> >
        (vx, y, ne00, ne01, ne02, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const src_t* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq1_m* x, dst_t* yy) {
    const int64_t i = blockIdx.x;
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
static __global__ void dequantize_block(const block_q2_K* x, dst_t* yy) {
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t n = tid / 32;
    const int64_t l = tid - 32 * n;
    const int64_t is = 8 * n + l / 16;

    const uint8_t q = x[i].qs[32 * n + l];
    dst_t* y = yy + i * QK_K + 128 * n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l + 0] = dall * (x[i].scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is + 0] >> 4);
    y[l + 32] = dall * (x[i].scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is + 2] >> 4);
    y[l + 64] = dall * (x[i].scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is + 4] >> 4);
    y[l + 96] = dall * (x[i].scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is + 6] >> 4);
}

template<typename dst_t>
static __global__ void dequantize_block(const block_q3_K* x, dst_t* yy) {
    const int64_t i = blockIdx.x;
    const int64_t r = threadIdx.x / 4;
    const int64_t tid = r / 2;
    const int64_t is0 = r % 2;
    const int64_t l0 = 16 * is0 + 4 * (threadIdx.x % 4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4 * n;

    uint8_t m = 1 << (4 * n + j);
    int64_t is = 8 * n + 2 * j + is0;
    int shift = 2 * j;

    int8_t us = is < 4 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 8] >> 0) & 3) << 4) :
        is < 8 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 4] >> 2) & 3) << 4) :
        is < 12 ? (x[i].scales[is - 8] >> 4) | (((x[i].scales[is + 0] >> 4) & 3) << 4) :
        (x[i].scales[is - 8] >> 4) | (((x[i].scales[is - 4] >> 6) & 3) << 4);
    float d_all = __half2float(x[i].d);
    float dl = d_all * (us - 32);

    dst_t* y = yy + i * QK_K + 128 * n + 32 * j;
    const uint8_t* q = x[i].qs + 32 * n;
    const uint8_t* hm = x[i].hmask;

    for (int l = l0; l < l0 + 4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    }
    else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_q4_K* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ir = tid % 8;
    const int64_t is = 2 * il;
    const int64_t n = 4;

    dst_t* y = yy + i * QK_K + 64 * il + n * ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t* q = x[i].qs + 32 * il + n * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_q5_K* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 16;   // il is in 0...3
    const int64_t ir = tid % 16;   // ir is in 0...15
    const int64_t is = 2 * il;     // is is in 0...6

    dst_t* y = yy + i * QK_K + 64 * il + 2 * ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t* ql = x[i].qs + 32 * il + 2 * ir;
    const uint8_t* qh = x[i].qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm = 1 << (2 * il);
    y[0] = d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1;
    y[1] = d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block(const block_q6_K* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip = tid / 32;   // ip is 0 or 1
    const int64_t il = tid - 32 * ip; // 0...32
    const int64_t is = 8 * ip + il / 16;

    dst_t* y = yy + i * QK_K + 128 * ip + il;

    const float d = __half2float(x[i].d);

    const uint8_t* ql = x[i].ql + 64 * ip + il;
    const uint8_t   qh = x[i].qh[32 * ip + il];
    const int8_t* sc = x[i].scales + is;

    y[0] = d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq1_s* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = __half2float(x[i].d) * (2 * ((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t* q = (const int8_t*)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4 * ib + il] | (((x[i].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq2_xxs* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint16_t* q2 = x[i].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*)q2;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq2_xs* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint16_t* q2 = x[i].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[il] & 511));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq2_s* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*)(iq2s_grid + (x[i].qs[4 * ib + il] | ((x[i].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq3_xxs* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t* q3 = x[i].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*)(x[i].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        y[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq4_nl* vx, dst_t* yy) {

    const int64_t i = blockIdx.x;
    const block_iq4_nl* x = vx + i * (QK_K / block_iq4_nl::block_size);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = __half2float(x[ib].d);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j + 16] = d * kvalues_iq4nl[q4[j] >> 4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq4_xs* x, dst_t* yy) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 4 * il;
    const uint8_t* q4 = x[i].qs + 16 * ib + 4 * il;
    const float d = __half2float(x[i].d) * ((((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j + 16] = d * kvalues_iq4nl[q4[j] >> 4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_mxfp4* vx, dst_t* yy) {

    const int64_t i = blockIdx.x;
    const block_mxfp4* x = vx + i * (QK_K / block_mxfp4::block_size);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * kvalues_mxfp4[q4[j] & 0xf] * 0.5f;
        y[j + 16] = d * kvalues_mxfp4[q4[j] >> 4] * 0.5f;
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_iq3_s* x, dst_t* yy) {

    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8; // 0...3
    const int64_t ib = tid % 8; // 0...7
    dst_t* y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t* qs = x[i].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 0] | ((x[i].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 1] | ((x[i].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = __half2float(x[i].d) * (1 + 2 * ((x[i].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[i].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        y[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}

template <typename src_t, typename dst_t>
static void dequantize_row_cuda(const src_t* x, dst_t* y, const int64_t k, cudaStream_t stream) {
    if constexpr (
        std::is_same_v<src_t, block_iq1_m> ||
        std::is_same_v<src_t, block_q2_K> ||
        std::is_same_v<src_t, block_q3_K> ||
        std::is_same_v<src_t, block_q4_K> ||
        std::is_same_v<src_t, block_q5_K> ||
        std::is_same_v<src_t, block_q6_K> ||
        std::is_same_v<src_t, block_iq2_xxs> ||
        std::is_same_v<src_t, block_iq2_xs> ||
        std::is_same_v<src_t, block_iq2_s> ||
        std::is_same_v<src_t, block_iq3_xxs> ||
        std::is_same_v<src_t, block_iq1_s> ||
        std::is_same_v<src_t, block_iq4_nl> ||
        std::is_same_v<src_t, block_iq4_xs> ||
        std::is_same_v<src_t, block_iq3_s>
        ) {
        const int nb = (k + QK_K - 1) / QK_K;
        constexpr auto value = [] {
            if constexpr (std::is_same_v<src_t, block_q2_K>) return 64;
            if constexpr (std::is_same_v<src_t, block_q3_K>) return 64;
            if constexpr (std::is_same_v<src_t, block_q5_K>) return 64;
            if constexpr (std::is_same_v<src_t, block_q6_K>) return 64;
            return 32;
        }();
        dequantize_block << <nb, value, 0, stream >> > (x, y);
    }
    else if constexpr (
        std::is_same_v<src_t, block_q4_0> ||
        std::is_same_v<src_t, block_q4_1>) {
        const int nb32 = k / 32;
        const int nb = (k + 255) / 256;
        dequantize_block << <nb, 32, 0, stream >> > (x, y, nb32);
    }
    else {
        assert(false);
    }
}

static __device__ __forceinline__ void dequantize(const block_q4_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = __half2float(x[ib].d);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, { 8.0f, 8.0f });
    v = __hmul2(v, { d, d });
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q4_1* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = __low2half(std::bit_cast<float>(x[ib].dm));
    const dfloat m = __high2half(std::bit_cast<float>(x[ib].dm));

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
    v = __hadd2(v, { m, m });
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q5_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = __half2float(x[ib].d);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, { 16.0f, 16.0f });
    v = __hmul2(v, { d, d });
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize(const block_q5_1* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
    v = __hadd2(v, { m, m });
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

template <int qr, typename src_t, typename dst_t>
static __global__ void dequantize_block(const src_t* __restrict__ x, dst_t* __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t s01, const int64_t s02, const int64_t s03) {
    const int qk = src_t::block_size;
    const int64_t i00 = 2 * (int64_t(blockDim.x) * blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = blockIdx.y;
    const int64_t i02 = blockIdx.z % ne02;
    const int64_t i03 = blockIdx.z / ne02;

    const int64_t ibx0 = i03 * s03 + i02 * s02 + i01 * s01;

    const int64_t ib = ibx0 + i00 / qk; // block index
    const int64_t iqs = (i00 % qk) / qr; // quant index
    const int64_t iybs = i00 - i00 % qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    dequantize(x, ib, iqs, v);

    const int64_t iy0 = ((i03 * ne02 + i02) * ne01 + i01) * ne00 + iybs + iqs;
    y[iy0 + 0] = ggml_cuda_cast<dst_t>(v.x);
    y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
}

template <int qr, typename src_t, typename dst_t>
static void dequantize_block_cuda(const src_t* x, dst_t* y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const dim3 num_blocks((ne00 + 2 * CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * CUDA_DEQUANTIZE_BLOCK_SIZE), ne01, ne02 * ne03);
    dequantize_block<qr> << <num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream >> >
        (x, y, ne00, ne01, ne02, s01, s02, s03);
}

template <int qr, typename src_t, typename dst_t>
static void dequantize_block_cont_cuda(const src_t* __restrict__ x, dst_t* __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int qk = src_t::block_size;
    dequantize_block_cuda<qr>(x, y, k, 1, 1, 1, k / qk, k / qk, k / qk, stream);
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

static __device__ __forceinline__ void dequantize(const block_q8_0* x, const int64_t ib, const int iqs, dfloat2& v) {
    const dfloat d = __half2float(x[ib].d);

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, { d, d });
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

void to_fp32_cuda(ggml_type type, const void* x, float* y, int64_t k, cudaStream_t stream)
{
    switch (type) {
    case GGML_TYPE_Q4_0:
        return dequantize_row_cuda(static_cast<const block_q4_0*>(x), y, k, stream);
    case GGML_TYPE_Q4_1:
        return dequantize_row_cuda(static_cast<const block_q4_1*>(x), y, k, stream);
    case GGML_TYPE_Q5_0:
        return dequantize_block_cont_cuda<QR5_0>(static_cast<const block_q5_0*>(x), y, k, stream);
    case GGML_TYPE_Q5_1:
        return dequantize_block_cont_cuda<QR5_1>(static_cast<const block_q5_1*>(x), y, k, stream);
    case GGML_TYPE_Q8_0:
        return dequantize_block_cont_cuda<QR8_0>(static_cast<const block_q8_0*>(x), y, k, stream);
    case GGML_TYPE_Q2_K:
        return dequantize_row_cuda(static_cast<const block_q2_K*>(x), y, k, stream);
    case GGML_TYPE_Q3_K:
        return dequantize_row_cuda(static_cast<const block_q3_K*>(x), y, k, stream);
    case GGML_TYPE_Q4_K:
        return dequantize_row_cuda(static_cast<const block_q4_K*>(x), y, k, stream);
    case GGML_TYPE_Q5_K:
        return dequantize_row_cuda(static_cast<const block_q5_K*>(x), y, k, stream);
    case GGML_TYPE_Q6_K:
        return dequantize_row_cuda(static_cast<const block_q6_K*>(x), y, k, stream);
    case GGML_TYPE_IQ2_XXS:
        return dequantize_row_cuda(static_cast<const block_iq2_xxs*>(x), y, k, stream);
    case GGML_TYPE_IQ2_XS:
        return dequantize_row_cuda(static_cast<const block_iq2_xs*>(x), y, k, stream);
    case GGML_TYPE_IQ2_S:
        return dequantize_row_cuda(static_cast<const block_iq2_s*>(x), y, k, stream);
    case GGML_TYPE_IQ3_XXS:
        return dequantize_row_cuda(static_cast<const block_iq3_xxs*>(x), y, k, stream);
    case GGML_TYPE_IQ1_S:
        return dequantize_row_cuda(static_cast<const block_iq1_s*>(x), y, k, stream);
    case GGML_TYPE_IQ1_M:
        return dequantize_row_cuda(static_cast<const block_iq1_m*>(x), y, k, stream);
    case GGML_TYPE_IQ4_NL:
        return dequantize_row_cuda(static_cast<const block_iq4_nl*>(x), y, k, stream);
    case GGML_TYPE_IQ4_XS:
        return dequantize_row_cuda(static_cast<const block_iq4_xs*>(x), y, k, stream);
    case GGML_TYPE_IQ3_S:
        return dequantize_row_cuda(static_cast<const block_iq3_s*>(x), y, k, stream);
    case GGML_TYPE_MXFP4:
        return dequantize_row_cuda(static_cast<const block_mxfp4*>(x), y, k, stream);
    case GGML_TYPE_F16:
        return convert_unary_cont_cuda(static_cast<const half*>(x), y, k, stream);
    case GGML_TYPE_BF16:
        return convert_unary_cont_cuda(static_cast<const nv_bfloat16*>(x), y, k, stream);
    default:
        assert(false);
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_q4_0* vx, dst_t* yy, int nb32) {

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

    const block_q4_0* x = vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8 * d;

    const uint8_t* q = x->qs + 4 * il;

    for (int l = 0; l < 4; ++l) {
        y[l + 0] = d * (q[l] & 0xF) + dm;
        y[l + 16] = d * (q[l] >> 4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block(const block_q4_1* vx, dst_t* yy, int nb32) {

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

    const block_q4_1* x = vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t* q = x->qs + 4 * il;

    for (int l = 0; l < 4; ++l) {
        y[l + 0] = d.x * (q[l] & 0xF) + d.y;
        y[l + 16] = d.x * (q[l] >> 4) + d.y;
    }
}

void to_fp16_cuda(ggml_type type, const void* x, half* y, int64_t k, cudaStream_t stream)
{
    switch (type) {
    case GGML_TYPE_Q4_0:
        return dequantize_row_cuda(static_cast<const block_q4_0*>(x), y, k, stream);
    case GGML_TYPE_Q4_1:
        return dequantize_row_cuda(static_cast<const block_q4_1*>(x), y, k, stream);
    case GGML_TYPE_Q5_0:
        return dequantize_block_cont_cuda<QR5_0>(static_cast<const block_q5_0*>(x), y, k, stream);
    case GGML_TYPE_Q5_1:
        return dequantize_block_cont_cuda<QR5_1>(static_cast<const block_q5_1*>(x), y, k, stream);
    case GGML_TYPE_Q8_0:
        if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= GGML_CUDA_CC_PASCAL) {
            return dequantize_block_q8_0_f16_cuda(x, y, k, stream);
        }
        return dequantize_block_cont_cuda<QR8_0>(static_cast<const block_q8_0*>(x), y, k, stream);
    case GGML_TYPE_Q2_K:
        return dequantize_row_cuda(static_cast<const block_q2_K*>(x), y, k, stream);
    case GGML_TYPE_Q3_K:
        return dequantize_row_cuda(static_cast<const block_q3_K*>(x), y, k, stream);
    case GGML_TYPE_Q4_K:
        return dequantize_row_cuda(static_cast<const block_q4_K*>(x), y, k, stream);
    case GGML_TYPE_Q5_K:
        return dequantize_row_cuda(static_cast<const block_q5_K*>(x), y, k, stream);
    case GGML_TYPE_Q6_K:
        return dequantize_row_cuda(static_cast<const block_q6_K*>(x), y, k, stream);
    case GGML_TYPE_IQ2_XXS:
        return dequantize_row_cuda(static_cast<const block_iq2_xxs*>(x), y, k, stream);
    case GGML_TYPE_IQ2_XS:
        return dequantize_row_cuda(static_cast<const block_iq2_xs*>(x), y, k, stream);
    case GGML_TYPE_IQ2_S:
        return dequantize_row_cuda(static_cast<const block_iq2_s*>(x), y, k, stream);
    case GGML_TYPE_IQ3_XXS:
        return dequantize_row_cuda(static_cast<const block_iq3_xxs*>(x), y, k, stream);
    case GGML_TYPE_IQ1_S:
        return dequantize_row_cuda(static_cast<const block_iq1_s*>(x), y, k, stream);
    case GGML_TYPE_IQ1_M:
        return dequantize_row_cuda(static_cast<const block_iq1_m*>(x), y, k, stream);;
    case GGML_TYPE_IQ4_NL:
        return dequantize_row_cuda(static_cast<const block_iq4_nl*>(x), y, k, stream);
    case GGML_TYPE_IQ4_XS:
        return dequantize_row_cuda(static_cast<const block_iq4_xs*>(x), y, k, stream);
    case GGML_TYPE_IQ3_S:
        return dequantize_row_cuda(static_cast<const block_iq3_s*>(x), y, k, stream);
    case GGML_TYPE_MXFP4:
        return dequantize_row_cuda(static_cast<const block_mxfp4*>(x), y, k, stream);
    case GGML_TYPE_F32:
        return convert_unary_cont_cuda(static_cast<const float*>(x), y, k, stream);
    default:
        return GGML_ABORT("Fatal error");
    }
}

void to_bf16_cuda(ggml_type type, const void* x, nv_bfloat16* y, int64_t k, cudaStream_t stream)
{
    switch (type) {
    case GGML_TYPE_F32:
        return convert_unary_cont_cuda(static_cast<const float*>(x), y, k, stream);
    case GGML_TYPE_F16:
        return convert_unary_cont_cuda(static_cast<const half*>(x), y, k, stream);
    default:
        return GGML_ABORT("Fatal error");
    }
}

void convert_to_nc_cuda(ggml_type type, const void* x, half* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    switch (type) {
    case GGML_TYPE_F32:
        return convert_unary_cuda<float>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q4_0:
        return dequantize_block_cuda<QR4_0>(static_cast<const block_q4_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q4_1:
        return dequantize_block_cuda<QR4_1>(static_cast<const block_q4_1*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QR5_0>(static_cast<const block_q5_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QR5_1>(static_cast<const block_q5_1*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q8_0:
        return dequantize_block_cuda<QR8_0>(static_cast<const block_q8_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_BF16:
        return convert_unary_cuda<nv_bfloat16>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    default:
        return GGML_ABORT("Fatal error");
    }
}

void convert_to_nc_cuda(ggml_type type, const void* x, nv_bfloat16* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    switch (type) {
    case GGML_TYPE_F32:
        return convert_unary_cuda<float, nv_bfloat16>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q4_0:
        return dequantize_block_cuda<QR4_0>(static_cast<const block_q4_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q4_1:
        return dequantize_block_cuda<QR4_1>(static_cast<const block_q4_1*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QR5_0>(static_cast<const block_q5_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QR5_1>(static_cast<const block_q5_1*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q8_0:
        return dequantize_block_cuda<QR8_0>(static_cast<const block_q8_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_F16:
        return convert_unary_cuda<half, nv_bfloat16>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    default:
        return GGML_ABORT("Fatal error");
    }
}

void convert_to_nc_cuda(ggml_type type, const void* x, float* y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    switch (type) {
    case GGML_TYPE_F16:
        return convert_unary_cuda<half, float>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q4_0:
        return dequantize_block_cuda<QR4_0>(static_cast<const block_q4_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q4_1:
        return dequantize_block_cuda<QR4_1>(static_cast<const block_q4_1*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QR5_0>(static_cast<const block_q5_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QR5_1>(static_cast<const block_q5_1*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_Q8_0:
        return dequantize_block_cuda<QR8_0>(static_cast<const block_q8_0*>(x), y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    case GGML_TYPE_BF16:
        return convert_unary_cuda<nv_bfloat16, float>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
    default:
        return GGML_ABORT("Fatal error");
    }
}
