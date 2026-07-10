module;
#include "common.h"
#include <bit>
#include <span>
#include "op/cuda_func.h"

static constexpr int64_t MMF_ROWS_PER_BLOCK = 32;

module ggml;
import :cuda.utils;

namespace utils
{
    bin_bcast_context create_bcast_context(const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst)
    {
        bin_bcast_context ctx{
            .dst_d = dst->data,
            .src0_type = std::bit_cast<internal::ggml_type>(src0->type),
            .src1_type = std::bit_cast<internal::ggml_type>(src1->type),
            .dst_type = std::bit_cast<internal::ggml_type>(dst->type),
			.src0_ne = {src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]},
            .src0_nb = {src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]},
            .src1_ne = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]},
            .src1_nb = {src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]},
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]},
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]}
        };
        for (size_t i = 0; i < dst->src.size(); i++)
            ctx.src_data[i] = dst->src[i]->data;
        return ctx;
    }

    bool should_use_mmf(ggml_type type, int cc, int warp_size, std::span<const int64_t> src0_ne,
        std::span<const size_t> src0_nb, int64_t src1_ncols, bool mul_mat_id)
    {
        if (ggml_is_quantized(type)) return false;

        const size_t ts = ggml_type_size(type);

        if (src0_ne[0] % (warp_size * (4 / ts)) != 0) {
            return false;
        }

        if (src0_nb[0] != ts) {
            return false;
        }

        // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
        for (auto nb : src0_nb.subspan(1)) {
            if (nb % (2 * ts) != 0) {
                return false;
            }
        }
        if (src0_ne[1] % mmf_get_rows_per_block(cc) != 0) {
            return false;
        }

        if (GGML_CUDA_CC_IS_CDNA3(cc) && type == GGML_TYPE_BF16) {
            return false;
        }

        if (mul_mat_id) {
            if (src0_ne[1] <= 1024 && src1_ncols > 512) {
                return false;
            }
            else if (src0_ne[1] > 1024 && src1_ncols > 128) {
                return false;
            }
        }
        else {
            if (GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 8) {
                return false;
            }
            else if (GGML_CUDA_CC_IS_CDNA2(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
                //TODO: truse CDNA2 as CDNA1, tune the perf when CDNA2 is available.
                return false;
            }
            else if (GGML_CUDA_CC_IS_CDNA1(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
                return false;
            }
            else if (src1_ncols > 16) {
                return false;
            }
        }

        switch (type) {
        case GGML_TYPE_F32:
            return ampere_mma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_BF16:
            return ampere_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        default:
            return false;
        }
    }

    bool should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
        if constexpr (force_enable_cuda_blas_v) {
            return false;
        }

        bool mmq_supported;

        switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
        }

        if (!mmq_supported) {
            return false;
        }

        // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
        {
            const int    id = ggml_cuda_get_device();
            const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
            if (smpbo < 48 * 1024) {
                return false;
            }
        }

        if (turing_mma_available(cc)) {
            return true;
        }

        if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
            return false;
        }

        if constexpr (ggml_cuda_force_mmq_v) return true;

        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
        }

        if (amd_mfma_available(cc)) {
            // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
            // performs better but is currently suffering from a crash on this architecture.
            // TODO: Revisit when hipblaslt is fixed on CDNA3
            if (GGML_CUDA_CC_IS_CDNA3(cc)) {
                return true;
            }
            if (n_experts > 64 || ne11 <= 128) {
                return true;
            }
            if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
                return true;
            }
            if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
                return true;
            }
            return false;
        }

        if (amd_wmma_available(cc)) {
            if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                // High expert counts are almost always better on MMQ due to
                //     the synchronization overhead in the cuBLAS/hipBLAS path:
                // https://github.com/ggml-org/llama.cpp/pull/18202
                if (n_experts >= 64) {
                    return true;
                }

                // For some quantization types MMQ can have lower peak TOPS than hipBLAS
                //     so it's only faster for sufficiently small batch sizes:
                switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
                }
            }

            // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
            // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
            return true;
        }

        // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
        // for dense matrices; keep MMQ only for MoE, where the
        // hipBLAS path is much slower.
        if (cc == GGML_CUDA_CC_VEGA) {
            return n_experts > 0;
        }

        return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    bool should_use_mmvf(ggml_type type, int cc, std::span<const int64_t> src0_ne, std::span<const size_t> src0_nb, int64_t ne11) {
        if (src0_ne[0] % 2 != 0) {
            return false;
        }

        const size_t ts = ggml_type_size(type);
        if (src0_nb[0] != ts) {
            return false;
        }

        // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
        for (auto nb : src0_nb.subspan(1)) {
            if (nb % (2 * ts) != 0) {
                return false;
            }
        }

        switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
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
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
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
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
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

    bool should_use_mmv(ggml_type type, int cc, std::span<const int64_t> src0_ne, int64_t ne11) {
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
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
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

    bool should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
        if (!ggml_is_quantized(type)) {
            return false;
        }
        // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
        // Only list quant-types MMQ supports, others would fall back to cuBLAS.
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
            switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
            switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 5;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
            switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            if (GGML_CUDA_CC_IS_CDNA1(cc)) {
                switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
                }
            }
            switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        return ne11 <= MMVQ_MAX_BATCH_SIZE;
    }

    // WMMA flash attention requires FP16 matrix instructions to be available for ggml code.
    static bool ggml_cuda_should_use_wmma_fattn(const int cc) {
#if defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
        return false;
#else
        if ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA) ||
            GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_MTHREADS(cc)) {
            return true;
        }
        else if (GGML_CUDA_CC_IS_CDNA(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
            return true;
#else
            return false;
#endif // defined(GGML_HIP_ROCWMMA_FATTN) (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
        }
        else if (GGML_CUDA_CC_IS_RDNA4(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN) && ROCWMMA_VERSION_MAJOR > 1
            return true;
#else
            return false;
#endif // defined(GGML_HIP_ROCWMFMA_FATTN) && ROCWMMA_VERSION_MAJOR > 1
        }
        else {
            return false;
        }
#endif // defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
    }

    static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
        switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            if constexpr (not ggml_cuda_fa_all_quants_v) {
                return false;
            }
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
        }
    }

    best_fattn_kernel ggml_cuda_get_best_fattn_kernel([[maybe_unused]] const int device, [[maybe_unused]] const ggml_tensor* dst) {
        if constexpr (!flash_attn_available_v) {
            return BEST_FATTN_KERNEL_NONE;
        }

        const ggml_tensor* KQV = dst;
        const ggml_tensor* Q = dst->src[0];
        const ggml_tensor* K = dst->src[1];
        const ggml_tensor* V = dst->src[2];
        const ggml_tensor* mask = dst->src[3];

        const int gqa_ratio = Q->ne[2] / K->ne[2];
        assert(Q->ne[2] % K->ne[2] == 0);

        float max_bias = 0.0f;
        memcpy(&max_bias, (const float*)KQV->op_params + 1, sizeof(float));

        static constexpr int64_t FATTN_KQ_STRIDE = 256;

        // The effective batch size for the kernel can be increased by gqa_ratio.
        // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
        bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
        for (const ggml_tensor* t : { Q, K, V, mask }) {
            if (t == nullptr || ggml_is_quantized(t->type)) {
                continue;
            }
            for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
                if (t->nb[i] % 16 != 0) {
                    gqa_opt_applies = false;
                    break;
                }
            }
        }

        const int cc = ggml_cuda_info().devices[device].cc;

        switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
        }

        if constexpr (not ggml_cuda_fa_all_quants_v) {
            if (K->type != V->type) {
                return BEST_FATTN_KERNEL_NONE;
            }
        }

        if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
            return BEST_FATTN_KERNEL_NONE;
        }

        if (mask && mask->ne[2] != 1) {
            return BEST_FATTN_KERNEL_NONE;
        }

        // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
        // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
        const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

        // If Turing tensor cores are available, use them:
        if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
            if (can_use_vector_kernel) {
                if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                    if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
                else {
                    if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                        if (Q->ne[1] <= 2) {
                            return BEST_FATTN_KERNEL_VEC;
                        }
                    }
                    else {
                        if (Q->ne[1] == 1) {
                            return BEST_FATTN_KERNEL_VEC;
                        }
                    }
                }
                if (!gqa_opt_applies && Q->ne[1] == 1) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
            return BEST_FATTN_KERNEL_MMA_F16;
        }

        const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
        int gqa_ratio_eff = 1;
        while (gqa_ratio % (2 * gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }

        if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
            if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
            if (Q->ne[1] * gqa_ratio_eff <= 16) {
                return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
            }
            return BEST_FATTN_KERNEL_MMA_F16;
        }

        // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
        if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
            if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
                return BEST_FATTN_KERNEL_MMA_F16;
            }
            if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
                return BEST_FATTN_KERNEL_MMA_F16;
            }
            if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
                return BEST_FATTN_KERNEL_MMA_F16;
            }
        }

        // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
        if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }

        // If there are no tensor cores available, use the generic tile kernel:
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        return BEST_FATTN_KERNEL_TILE;
    }

    bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor* dst) {
        return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
    }

    size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor* dst)
    {
        assert(dst->op == GGML_OP_FLASH_ATTN_EXT);

        const ggml_tensor* K = dst->src[1];
        const ggml_tensor* V = dst->src[2];

        assert(K != nullptr);
        assert(V != nullptr);

        const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

        bool need_f16_K = false;
        bool need_f16_V = false;

        switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
        }

        flash_attn_ext_context ctx{
            .V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs)),
            .K = {
                .type = std::bit_cast<internal::ggml_type>(K->type),
                .block_size = ggml_blck_size(K->type),
                .type_size = ggml_type_size(K->type),
                .data = K->data,
                .elements = K->nelements(),
                .ne0 = K->ne[0],
                .ne1 = K->ne[1],
                .ne2 = K->ne[2],
                .ne3 = K->ne[3],
                .nb0 = K->nb[0],
                .nb1 = K->nb[1],
                .nb2 = K->nb[2],
                .nb3 = K->nb[3],
                .bs = ggml_blck_size(K->type),
                .ts = ggml_type_size(K->type),
                .element_size = ggml_element_size(K)
            },
            .V = {
                .type = std::bit_cast<internal::ggml_type>(V->type),
                .block_size = ggml_blck_size(V->type),
                .type_size = ggml_type_size(V->type),
                .data = V->data,
                .elements = V->nelements(),
                .ne0 = V->ne[0],
                .ne1 = V->ne[1],
                .ne2 = V->ne[2],
                .ne3 = V->ne[3],
                .nb0 = V->nb[0],
                .nb1 = V->nb[1],
                .nb2 = V->nb[2],
                .nb3 = V->nb[3],
                .bs = ggml_blck_size(V->type),
                .ts = ggml_type_size(V->type),
                .element_size = ggml_element_size(V)
            },
            .KQV = {
                .type = std::bit_cast<internal::ggml_type>(dst->type),
                .data = dst->data,
                .elements = dst->nelements(),
                .nbytes = dst->nbytes(),
                .nrows = ggml_nrows(dst),
                .ne0 = dst->ne[0],
                .ne1 = dst->ne[1],
                .ne2 = dst->ne[2],
                .ne3 = dst->ne[3]
            }
        };

        const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
            ggml_cuda_flash_attn_ext_get_f16_extra_data(ctx, need_f16_K, need_f16_V);

        return f16_extra.end - (uintptr_t)dst->data;
    }

    gated_delta_net_context build_gated_delta_net_context(ggml_tensor* dst)
    {
        ggml_tensor* src_q = dst->src[0];
        ggml_tensor* src_k = dst->src[1];
        ggml_tensor* src_v = dst->src[2];
        ggml_tensor* src_g = dst->src[3];
        ggml_tensor* src_beta = dst->src[4];
        ggml_tensor* src_state = dst->src[5];

        const int64_t S_v = src_v->ne[0];

        const bool kda = (src_g->ne[0] == S_v);

        assert(src_q->ne[1] == src_k->ne[1]);

        assert(ggml_is_contiguous_rows(src_q));
        assert(ggml_is_contiguous_rows(src_k));
        assert(ggml_is_contiguous_rows(src_v));
        assert(ggml_are_same_stride(src_q, src_k));
        assert(src_g->ne[0] == 1 || kda);
        assert(ggml_is_contiguous(src_g));
        assert(ggml_is_contiguous(src_beta));
        assert(ggml_is_contiguous(src_state));
        assert(src_q->ne == src_k->ne && src_q->nb == src_k->nb);

        // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
        const int K = std::bit_cast<int>(dst->op_params[0]);

        return gated_delta_net_context {
            .kda = kda,
            .q_d = (const float*)src_q->data,
            .k_d = (const float*)src_k->data,
            .v_d = (const float*)src_v->data,
            .g_d = (const float*)src_g->data,
            .b_d = (const float*)src_beta->data,
            .s_d = (const float*)src_state->data,
            .dst_d = (float*)dst->data,
            .S_v = S_v,
            .H = src_v->ne[1],
            .n_tokens = src_v->ne[2],
            .n_seqs = src_v->ne[3],
            // strides in floats (beta strides used for both g and beta offset computation)
            .qk_ne = { src_q->ne[0], src_q->ne[1], src_q->ne[2], src_q->ne[3] },
            .qk_nb = { src_q->nb[0], src_q->nb[1], src_q->nb[2], src_q->nb[3] },
            .v_ne = { src_v->ne[0], src_v->ne[1], src_v->ne[2], src_v->ne[3] },
            .v_nb = { src_v->nb[0], src_v->nb[1], src_v->nb[2], src_v->nb[3] },
            .g_ne = { src_g->ne[0], src_g->ne[1], src_g->ne[2], src_g->ne[3] },
            .g_nb = { src_g->nb[0], src_g->nb[1], src_g->nb[2], src_g->nb[3] },
            .b_ne = { src_beta->ne[0], src_beta->ne[1], src_beta->ne[2], src_beta->ne[3] },
            .b_nb = { src_beta->nb[0], src_beta->nb[1], src_beta->nb[2], src_beta->nb[3] },
            .neqk1 = src_q->ne[1],
            .rq3 = src_v->ne[3] / src_q->ne[3],
            .scale = 1.0f / sqrtf((float)S_v),
            .K = K
        };
    }

    bool ggml_cuda_lightning_indexer_supported(int /*device*/, const ggml_tensor* dst) {
        const ggml_tensor* q = dst->src[0];
        const ggml_tensor* k = dst->src[1];
        const ggml_tensor* w = dst->src[2]; // weights
        const ggml_tensor* m = dst->src[3]; // mask

        if (q->ne[0] != 128) {
            return false;
        }

        if (q->ne[1] != 64 && q->ne[1] != 32) {
            return false;
        }

        // alignment checks
        for (const ggml_tensor* t : { q, k }) {
            if (ggml_is_quantized(t->type)) {
                continue;
            }
            for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
                if (t->nb[i] % 16 != 0) {
                    return false;
                }
            }
        }

        switch (k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
        }
    }

    bool ggml_cuda_is_aligned(const ggml_tensor* tensor, const size_t alignment) {
        assert(tensor != nullptr);
        return (reinterpret_cast<uintptr_t>(tensor->data) % alignment) == 0 &&
            tensor->nb[1] % alignment == 0 &&
            tensor->nb[2] % alignment == 0 &&
            tensor->nb[3] % alignment == 0;
    }

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor* dst, const int cc) {
        const ggml_tensor* src0 = dst->src[0];
        const ggml_tensor* src1 = dst->src[1];

        if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            return true;
        }

        if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                if (dst->ne[2] <= get_mmvq_mmid_max_batch(std::bit_cast<internal::ggml_type>(src0->type), cc)) {
                    return false;
                }
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                return false;
            }
        }

        if (should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
            return false;
        }

        if (should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            return false;
        }

        return true;
    }
}