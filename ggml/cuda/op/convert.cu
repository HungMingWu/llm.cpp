#include "cuda_func.h"
#include "table.h"
#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "block.h"
#include <bit>
#include <assert.h>
#include <type_traits>
#include "launch.cuh"
#include "helper.h"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template <typename src_t, typename dst_t>
void convert_unary_cuda(const convert_context& ctx, const void* vx, dst_t* y, cudaStream_t stream) {
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(vx), ctx.src_ne, ctx.src_nb);
    int64_t dst_ne[4] = { ctx.src_ne[0], ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::experimental::mdspan dst_data(y, dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]);
    launch_functor(stream, std::make_tuple(dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]),
        [=] __device__(int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
            dst_data(i3, i2, i1, i0) = ggml_cuda_cast<dst_t>(src_data(i3, i2, i1, i0));
        }
    );
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_q2_K* x, dst_t* y) {
    const int64_t n = threadIdx / 32;
    const int64_t l = threadIdx - 32 * n;
    const int64_t is = 8 * n + l / 16;

    const uint8_t q = x->qs[32 * n + l];
    const int64_t offset = 128 * n;

    float dall = __low2half(x->dm);
    float dmin = __high2half(x->dm);
    y[l + offset] = ggml_cuda_cast<dst_t>(dall * (x->scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (x->scales[is + 0] >> 4));
    y[l + offset + 32] = ggml_cuda_cast<dst_t>(dall * (x->scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (x->scales[is + 2] >> 4));
    y[l + offset + 64] = ggml_cuda_cast<dst_t>(dall * (x->scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (x->scales[is + 4] >> 4));
    y[l + offset + 96] = ggml_cuda_cast<dst_t>(dall * (x->scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (x->scales[is + 6] >> 4));
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_q3_K* x, dst_t* y) {
    const int64_t r = threadIdx / 4;
    const int64_t tid = r / 2;
    const int64_t is0 = r % 2;
    const int64_t l0 = 16 * is0 + 4 * (threadIdx % 4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4 * n;

    uint8_t m = 1 << (4 * n + j);
    int64_t is = 8 * n + 2 * j + is0;
    int shift = 2 * j;

    int8_t us = is < 4 ? (x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4) :
        is < 8 ? (x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4) :
        is < 12 ? (x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4) :
        (x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4);
    float d_all = __half2float(x->d);
    float dl = d_all * (us - 32);

    const int64_t offset = 128 * n + 32 * j;
    const uint8_t* q = x->qs + 32 * n;
    const uint8_t* hm = x->hmask;

    for (int l = l0; l < l0 + 4; ++l) y[l + offset] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
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

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_q4_K* x, dst_t* y) {
    // assume 32 threads
    const int64_t il = threadIdx / 8;
    const int64_t ir = threadIdx % 8;
    const int64_t is = 2 * il;
    const int64_t n = 4;

    const int64_t offset = 64 * il + n * ir;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);

    const uint8_t* q = x->qs + 32 * il + n * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + offset] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + offset + 32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_q5_K* x, dst_t* y) {
    // assume 64 threads - this is very slightly better than the one below
    const int64_t il = threadIdx / 16;   // il is in 0...3
    const int64_t ir = threadIdx % 16;   // ir is in 0...15
    const int64_t is = 2 * il;     // is is in 0...6

    const int64_t offset = 64 * il + 2 * ir;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);

    const uint8_t* ql = x->qs + 32 * il + 2 * ir;
    const uint8_t* qh = x->qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm = 1 << (2 * il);
    y[offset + 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    y[offset + 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[offset + 32] = ggml_cuda_cast<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
    y[offset + 33] = ggml_cuda_cast<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_q6_K* x, dst_t* y) {
    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip = threadIdx / 32;   // ip is 0 or 1
    const int64_t il = threadIdx - 32 * ip; // 0...32
    const int64_t is = 8 * ip + il / 16;
    const int64_t offset = 128 * ip + il;
    const float d = __half2float(x->d);

    const uint8_t* ql = x->ql + 64 * ip + il;
    const uint8_t   qh = x->qh[32 * ip + il];
    const int8_t* sc = x->scales + is;

    y[offset + 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[offset + 32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[offset + 64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[offset + 96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

template<typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq1_s* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;

    const float delta = x->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = __half2float(x->d) * (2 * ((x->qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t* q = (const int8_t*)grid32;
    grid32[0] = iq1s_grid_gpu[x->qs[4 * ib + il] | (((x->qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[offset + j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq1_m* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*)x->scales;
    iq1m_scale_t scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = __half2float(std::bit_cast<half>(scale)) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x->qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t* q = (const int8_t*)grid32;
    grid32[0] = iq1s_grid_gpu[x->qs[4 * ib + il] | (((x->qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[offset + j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq2_s* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;

    const uint8_t* grid = (const uint8_t*)(iq2s_grid + (x->qs[4 * ib + il] | ((x->qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = __half2float(x->d) * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x->qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[offset + j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq2_xs* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;

    const uint16_t* q2 = x->qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[il] & 511));
    const float d = __half2float(x->d) * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[offset + j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq2_xxs* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;

    const uint16_t* q2 = x->qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*)q2;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x->d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[offset + j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq3_s* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;

    const uint8_t* qs = x->qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 0] | ((x->qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * il + 1] | ((x->qh[ib] << (7 - 2 * il)) & 256)));
    const float d = __half2float(x->d) * (1 + 2 * ((x->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x->signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + offset] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + offset + 4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq3_xxs* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 8 * il;
    const uint8_t* q3 = x->qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*)(x->qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x->d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + offset] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + offset + 4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq4_nl* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 4 * il;

    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = __half2float(x[ib].d);
    for (int j = 0; j < 4; ++j) {
        y[j + offset] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + offset + 16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_iq4_xs* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 4 * il;

    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = __half2float(x->d) * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + offset] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + offset + 16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}

template <typename dst_t>
static __device__ void dequantize_block(const int64_t threadIdx, const block_mxfp4* x, dst_t* y) {
    const int64_t il = threadIdx / 8; // 0...3
    const int64_t ib = threadIdx % 8; // 0...7
    const int64_t offset = 32 * ib + 4 * il;

    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j + offset] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf] * 0.5f);
        y[j + offset + 16] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] >> 4] * 0.5f);
    }
}

template <typename src_t, int qr, typename dst_t>
void dequantize_block_cuda(const convert_context &ctx, const void* x, dst_t* y, cudaStream_t stream) {
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(x), ctx.src_ne, ctx.src_nb);
    int64_t dst_ne[4] = { ctx.src_ne[0], ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::experimental::mdspan dst_data(y, dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]);
    const int qk = src_t::block_size;
    launch_functor(stream, std::make_tuple(ctx.src_ne[3], ctx.src_ne[2], ctx.src_ne[1], ctx.src_ne[0] / 2),
        [=] __device__(int64_t i03, int64_t i02, int64_t i01, int64_t i00) {
            i00 *= 2;

            const int64_t ib = i00 / qk; // block index
            const int64_t iqs = (i00 % qk) / qr; // quant index
            const int64_t iybs = i00 - i00 % qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk / 2;

            // dequantize
            float2 v;
            dequantize(&src_data(i03, i02, i01, ib), iqs, v);

            dst_data(i03, i02, i01, iybs + iqs) = ggml_cuda_cast<dst_t>(v.x);
            dst_data(i03, i02, i01, iybs + iqs + y_offset) = ggml_cuda_cast<dst_t>(v.y);
        }
    );
}

template <typename src_t, typename dst_t>
void dequantize_block_cuda(const convert_context& ctx, const void* x, dst_t* y, cudaStream_t stream) {
    assert(ctx.src_ne[0] % QK_K == 0);
    int64_t src_ne[4] = { ctx.src_ne[0] / QK_K, ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    if constexpr (std::is_same_v<src_t, block_iq4_nl>) {
        src_ne[0] /= block_iq4_nl::block_size;
    }
    else if constexpr (std::is_same_v<src_t, block_mxfp4>) {
        src_ne[0] /= block_mxfp4::block_size;
    }
    else {
        src_ne[0] /= QK_K;
    }
    auto src_data = make_strided_mdspan(static_cast<const src_t*>(x), src_ne, ctx.src_nb);
    int64_t dst_ne[4] = { ctx.src_ne[0], ctx.src_ne[1], ctx.src_ne[2], ctx.src_ne[3] };
    std::experimental::mdspan dst_data(y, dst_ne[3], dst_ne[2], dst_ne[1], dst_ne[0]);
    constexpr auto threads = [] {
        if constexpr (std::is_same_v<src_t, block_q2_K>) return 64;
        if constexpr (std::is_same_v<src_t, block_q3_K>) return 64;
        if constexpr (std::is_same_v<src_t, block_q5_K>) return 64;
        if constexpr (std::is_same_v<src_t, block_q6_K>) return 64;
        return 32;
    }();

    launch_functor_with_threads(stream, std::make_tuple(src_ne[3], src_ne[2], src_ne[1], ctx.src_ne[0] / QK_K), threads, 0,
        [=] __device__(int64_t i03, int64_t i02, int64_t i01, int64_t i00, int64_t threadIdx) {
            const int64_t real_i00 = [=]() {
                if constexpr (std::is_same_v<src_t, block_iq4_nl>) {
                    return i00 * (QK_K / block_iq4_nl::block_size);
                }
                else if constexpr (std::is_same_v<src_t, block_mxfp4>) {
                    return i00 * (QK_K / block_mxfp4::block_size);
                }
                else {
                    return i00;
                }
            }();
            dequantize_block(threadIdx, &src_data(i03, i02, i01, real_i00), &dst_data(i03, i02, i01, i00 * QK_K));
        }
    );
}

template <typename dst_t>
static void convert_to(const convert_context& ctx, const void* x, dst_t* y, cudaStream_t stream)
{
    switch (ctx.src_type) {
    case internal::GGML_TYPE_F16:
        return convert_unary_cuda<half>(ctx, x, y, stream);
    case internal::GGML_TYPE_BF16:
        return convert_unary_cuda<nv_bfloat16>(ctx, x, y, stream);
    case internal::GGML_TYPE_F32:
        return convert_unary_cuda<float>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q4_0:
        return dequantize_block_cuda<block_q4_0, QR4_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q4_1:
        return dequantize_block_cuda<block_q4_1, QR4_1>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q5_0:
        return dequantize_block_cuda<block_q5_0, QR5_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q5_1:
        return dequantize_block_cuda<block_q5_1, QR5_1>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q8_0:
        return dequantize_block_cuda<block_q8_0, QR8_0>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q2_K:
        return dequantize_block_cuda<block_q2_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q3_K:
        return dequantize_block_cuda<block_q3_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q4_K:
        return dequantize_block_cuda<block_q4_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q5_K:
        return dequantize_block_cuda<block_q5_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_Q6_K:
        return dequantize_block_cuda<block_q6_K>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ1_S:
        return dequantize_block_cuda<block_iq1_s>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ1_M:
        return dequantize_block_cuda<block_iq1_m>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ2_S:
        return dequantize_block_cuda<block_iq2_s>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ2_XS:
        return dequantize_block_cuda<block_iq2_xs>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ2_XXS:
        return dequantize_block_cuda<block_iq2_xxs>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ3_S:
        return dequantize_block_cuda<block_iq3_s>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ3_XXS:
        return dequantize_block_cuda<block_iq3_xxs>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ4_NL:
        return dequantize_block_cuda<block_iq4_nl>(ctx, x, y, stream);
    case internal::GGML_TYPE_IQ4_XS:
        return dequantize_block_cuda<block_iq4_xs>(ctx, x, y, stream);
    case internal::GGML_TYPE_MXFP4:
        return dequantize_block_cuda<block_mxfp4>(ctx, x, y, stream);
    default:
        assert(false);
        return GGML_ABORT("Fatal error");
    }
}

void convert_to_cuda(const convert_context& ctx, const void* x, half* y, cudaStream_t stream) {
    return convert_to(ctx, x, y, stream);
}

void convert_to_cuda(const convert_context& ctx, const void* x, nv_bfloat16* y, cudaStream_t stream) {
    return convert_to(ctx, x, y, stream);
}

void convert_to_cuda(const convert_context& ctx, const void* x, float* y, cudaStream_t stream) {
    return convert_to(ctx, x, y, stream);
}
