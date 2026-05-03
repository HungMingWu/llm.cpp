#pragma once
#include <float.h>
#include "common.cuh"
#include "mdspan_helper.h"
#include "reduce.cuh"
#include "float2_overload.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int /*cc*/) {
    return 128;
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

template <int D>
constexpr size_t getnthreads_KQ_q()
{
    if constexpr (ggml_use_hip_v) {
#ifdef RDNA
        return 2;
#else
        return 4;
#endif // RDNA
    }
    else {
        return D / 4 < 32 ? D / 4 : 32;
    }
}

template <typename type_V, typename Ret, typename T>
static __device__ __forceinline__ Ret get_dequantize_value(const T& value) {
    return value;
}

template <>
__device__ __forceinline__ half2 get_dequantize_value<nv_bfloat16, half2, float2>(const float2& value) {
    return __float22half2_rn(value);
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, typename type_K, typename type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
    flash_attn_ext_context ctx,
    [[maybe_unused]] const char* __restrict__ K,
    [[maybe_unused]] const char* __restrict__ V,
    [[maybe_unused]] const int* __restrict__ KV_max,
    [[maybe_unused]] float* __restrict__ dst,
    [[maybe_unused]] float2* __restrict__ dst_meta,
    [[maybe_unused]] const float scale,
    [[maybe_unused]] const float m0,
    [[maybe_unused]] const float m1,
    [[maybe_unused]] const uint32_t n_head_log2,
    [[maybe_unused]] const int32_t ne00, [[maybe_unused]] const uint3   ne01, [[maybe_unused]] const int32_t ne02, [[maybe_unused]] const int32_t ne03,
    [[maybe_unused]] const int32_t nb01, [[maybe_unused]] const int32_t nb02, [[maybe_unused]] const int32_t nb03,
    [[maybe_unused]] const int32_t ne10, [[maybe_unused]] const int32_t ne11, [[maybe_unused]] const int32_t ne12, [[maybe_unused]] const int32_t ne13,
    [[maybe_unused]] const int32_t nb11, [[maybe_unused]] const int32_t nb12, [[maybe_unused]] const int64_t nb13,
    [[maybe_unused]] const int32_t nb21, [[maybe_unused]] const int32_t nb22, [[maybe_unused]] const int64_t nb23,
    [[maybe_unused]] const int32_t ne31, [[maybe_unused]] const int32_t ne32, [[maybe_unused]] const int32_t ne33,
    [[maybe_unused]] const int32_t nb31, [[maybe_unused]] const int32_t nb32, [[maybe_unused]] const int64_t nb33) {

    // Skip unused kernel variants for faster compilation:
    constexpr bool emit_no_device_code_v = [=]() -> bool {
        if (!flash_attn_available_v) return true;
        return use_logit_softcap && !(D == 128 || D == 256);
    }();
    if constexpr (emit_no_device_code_v) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int nthreads_KQ_q = getnthreads_KQ_q<D>();
    constexpr int nthreads_V_q = (D / 4 < 32 ? D / 4 : 32);

    constexpr int nthreads = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = (std::is_same_v<type_K, half> || std::is_same_v<type_V, nv_bfloat16>) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (std::is_same_v<type_V, half> || std::is_same_v<type_V, nv_bfloat16>) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (std::is_same_v<type_V, half> || std::is_same_v<type_V, nv_bfloat16>) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr bool Q_q8_1 = !std::is_same_v<type_K, half> && !std::is_same_v<type_V, nv_bfloat16>;
    using t1 = std::conditional_t<v_dot2_f32_f16_available_v, half, float>;

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    auto Q_data = make_strided_mdspan(static_cast<const float*>(ctx.Q.data), ctx.Q.ne, ctx.Q.nb);
    const float* Q = &Q_data(sequence, head, ic0, 0);
    K += nb13 * sequence + nb12 * (head / gqa_ratio);
    V += nb23 * sequence + nb22 * (head / gqa_ratio);

    const char* __restrict__ mask = (const char*)ctx.mask.data;
    const half* maskh = (const half*)(mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const float slope = get_alibi_slope(ctx.max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ = ncols * D;
    constexpr int ne_combine = nwarps * V_cols_per_iter * D;
    using t2 = std::conditional_t<v_dot2_f32_f16_available_v, half2, float2>;
    mdarray<t2, ncols, D / 2 / nthreads_V> VKQ;
    __shared__ t1 KQ[std::max(ne_KQ, ne_combine)];

    auto block = cooperative_groups::this_thread_block();
    auto tile_warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);
    auto tile_KQ = cooperative_groups::tiled_partition<nthreads_KQ>(block);

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX / 2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    __align__(16) mdarray<t2, ncols, D / 2 / nthreads_KQ> Q_reg; // Will be initialized completely.
    mdarray<int, ncols, std::max<size_t>(1, D / (sizeof(int) * nthreads_KQ))> Q_i32;
    mdarray<float2, ncols, std::max<size_t>(1, D / (sizeof(int) * nthreads_KQ))> Q_ds;
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int* tmp_q_i32 = (int*)&KQ[j * D];
            float2* tmp_q_ds = (float2*)(tmp_q_i32 + D / sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D / sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D / sizeof(int)) || i < int(D / sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D / QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            }
            else {
                const float* Q_f = (const float*)(&Q[j]);
                constexpr int nthreads_quantize = D / sizeof(int) < WARP_SIZE ? D / sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D / sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<nthreads_quantize>
                        (Q_f + i0 * sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0 / QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int* tmp_q_i32 = (int*)&KQ[j * D];
            float2* tmp_q_ds = (float2*)(tmp_q_i32 + D / sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D / sizeof(int)) / nthreads_KQ; i0++) {
                const int i = i0 * nthreads_KQ + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32(j, i0) = tmp_q_i32[i];
                Q_ds(j, i0) = tmp_q_ds[i / QI8_1];
            }
        }

        __syncthreads();
    }
    else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2* Q_j = (const float2*)(&Q[j]);
            const auto scale_vec = make_vec2<t2>(scale, scale);
#pragma unroll
            for (int i0 = 0; i0 < D / 2 / nthreads_KQ; i0 += cpy_ne) {
                const int i = i0 * nthreads_KQ + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) * cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne / 2; i1++) {
                        Q_reg(j, i0 + i1) =
                            make_vec2<t2>(Q_j[i + i1].x, Q_j[i + i1].y) * scale_vec;
                        Q_reg(j, i0 + cpy_ne / 2 + i1) =
                            make_vec2<t2>(Q_j[i + cpy_ne / 2 + i1].x, Q_j[i + cpy_ne / 2 + i1].y) * scale_vec;
                    }
                }
            }
        }
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;
    K += blockIdx.y * nthreads * nb11;
    V += blockIdx.y * nthreads * nb21;
    maskh += blockIdx.y * nthreads;
    for (int k_VKQ_0 = blockIdx.y * nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y * nthreads,
        // Increment pointers after each loop:
        K += gridDim.y * nthreads * nb11, V += gridDim.y * nthreads * nb21, maskh += gridDim.y * nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y * WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_fattn_vec_KQ<D, nthreads_KQ>((type_K*)(K + i_KQ * nb11), &Q_reg(j, 0), &Q_i32(j, 0), &Q_ds(j, 0));
                sum = cooperative_groups::reduce(tile_KQ, sum, cooperative_groups::plus<float>());

                if (use_logit_softcap) {
                    sum = ctx.logit_softcap * tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope * __half2float(maskh[j * ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + KQ_reg[j];
            KQ[j * nthreads + tid] = KQ_reg[j];

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2 / nthreads_V; i_VKQ_0++) {
                VKQ(j, i_VKQ_0) = VKQ(j, i_VKQ_0) * make_vec2<t2>(KQ_max_scale, KQ_max_scale);
            }
        }

        if constexpr (!ggml_use_hip_v) {
            __syncwarp();
        }

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y * WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            using storage_t = std::conditional_t<std::is_same_v<type_V, nv_bfloat16>, float2, t2>;
            storage_t tmp[V_rows_per_thread / 2];

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2 / nthreads_V; i_VKQ_0 += V_rows_per_thread / 2) {
                dequantize_V<V_rows_per_thread>((type_V*)(V + k * nb21), tmp,
                    2 * i_VKQ_0 * nthreads_V + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V) * V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread / 2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ(j, i_VKQ_0 + i_VKQ_1) = VKQ(j, i_VKQ_0 + i_VKQ_1) +
                            get_dequantize_value<type_V, t2>(tmp[i_VKQ_1]) * make_vec2<t2>(KQ[j * nthreads + k], KQ[j * nthreads + k]);
                    }
                }
            }
        }
    }

    if (const float* sinks = (const float*)ctx.sinks.data; sinks && blockIdx.y == 0) {
        const float sink = sinks[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2 / nthreads_V; i_VKQ_0++) {
                VKQ(j, i_VKQ_0) = VKQ(j, i_VKQ_0) * make_vec2<t2>(KQ_max_scale, KQ_max_scale);
            }
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = cooperative_groups::reduce(tile_warp, kqmax_new, cooperative_groups::greater<float>());
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2 / nthreads_V; i_VKQ_0++) {
            VKQ(j_VKQ, i_VKQ_0) = VKQ(j_VKQ, i_VKQ_0) * make_vec2<t2>(kqmax_scale, kqmax_scale);
        }

        t2* VKQ_tmp = (t2*)KQ + threadIdx.y * (V_cols_per_iter * D / 2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V) * (D / 2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2 / nthreads_V; i_VKQ_0 += V_rows_per_thread / 2) {
            const int i_VKQ = i_VKQ_0 * nthreads_V + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V) * (V_rows_per_thread / 2);

#pragma unroll
            for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread / 2; i_VKQ_1++) {
                VKQ_tmp[i_VKQ + i_VKQ_1] = VKQ(j_VKQ, i_VKQ_0 + i_VKQ_1);
            }
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = cooperative_groups::reduce(tile_warp, KQ_sum[j_VKQ], cooperative_groups::plus<float>());
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = cooperative_groups::reduce(tile_warp, KQ_sum[j_VKQ], cooperative_groups::plus<float>());

            std::mdspan KQ_mdspan(KQ, nwarps, V_cols_per_iter, D);
#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ_mdspan(w, v, i0 + tid));
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * int(ne01.z) + ic0 + j_VKQ) * ne02 + head) * gridDim.y + blockIdx.y) * D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols - 1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence * int(ne01.z) + ic0 + tid) * ne02 + head) * gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, typename type_K, typename type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(const flash_attn_ext_context& ctx) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = std::is_same_v<type_K, half>;
    const bool need_f16_V = std::is_same_v<type_V, half>;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, typename type_K, typename type_V>
void ggml_cuda_flash_attn_ext_vec_case(const flash_attn_ext_context& ctx) {
    if (ctx.Q.ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (ctx.logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (ctx.logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
    }
    else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx);
    }
}
